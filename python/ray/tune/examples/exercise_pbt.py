"""
lr, momentum 파라메터 튜닝하는 간단한 예제에서 출발

파라메터가 M개, 하이퍼파라메터 N개
ex) M= 2 , N = 3

PCT episode 1
  -- 튜닝이 필요한 파라메터들을 더 찾아내고 거기에 tuning을 집중하자.
PCT episode 2
  --
"""

import tensorflow as tf
# try:
#     tf.get_logger().setLevel('INFO')
# except Exception as exc:
#     print(exc)
import warnings
# warnings.simplefilter("ignore")
import os
import numpy as np
import torch
import torch.optim as optim
from torchvision import datasets
from ray.tune.examples.mnist_pytorch import train, test, ConvNet, get_data_loaders
import ray
from ray import tune
from ray.tune import track
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune.utils import validate_save_restore
import matplotlib.style as style
import matplotlib.pyplot as plt
import math


style.use("ggplot")
datasets.MNIST("data", train=True, download=True)

N_EPISODE = 5000
N_EPISODE_STEP = 2
N_PARAMS = 2
K = int(math.pow(2, N_PARAMS)) - 2


class ucb_bandit:
    '''
    Upper Confidence Bound Bandit
    ============================================
    '''
    def __init__(self, k, c, iters):
        self.k = k              # Number of arms
        self.c = c              # Exploration parameter
        self.iters = iters      # Number of iterations
        self.n = 1              # Step count
        self.k_n = np.ones(k)   # Step count for each arm
        self.mean_reward = 0    # Total mean reward
        self.reward = np.zeros(iters)
        self.k_reward = np.zeros(k-2)   # Mean reward for each arm

    def pull(self):
        # Select action according to UCB Criteria
        a = np.argmax(self.k_reward + self.c * np.sqrt(
            (np.log(self.n)) / self.k_n))
        return a

    def update(self, a, r):
        self.n += 1     # Update counts
        self.k_n[a] += 1
        self.mean_reward = self.mean_reward + (r - self.mean_reward) / self.n   # Update total
        self.k_reward[a] = self.k_reward[a] + (r - self.k_reward[a]) / self.k_n[a]  # Update results for a_k

    # def run(self):
    #     for i in range(self.iters):
    #         a = self.pull()
    #         r = np.random.normal(self.mu[a], 1)
    #         self.update(a, r)
    #         self.reward[i] = self.mean_reward

    def reset(self):    # Resets results while keeping settings
        self.n = 1
        self.k_n = np.ones(self.k)
        self.mean_reward = 0
        self.reward = np.zeros(self.iters)
        self.k_reward = np.zeros(self.k)


class PytorchTrainble(tune.Trainable):
    def _setup(self, config):
        self.device = torch.device("cpu")
        self.train_loader, self.test_loader = get_data_loaders()
        self.model = ConvNet().to(self.device)
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=config.get("lr", 0.01),
            momentum=config.get("momentum", 0.9))

    def _train(self):
        train(self.model, self.optimizer, self.train_loader, device=self.device)
        acc = test(self.model, self.test_loader, self.device)
        return {"mean_accuracy": acc}

    def _save(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
        torch.save(self.model.state_dict(), checkpoint_path)
        return checkpoint_path

    def _restore(self, checkpoint_path):
        self.model.load_state_dict(torch.load(checkpoint_path))

    def reset_config(self, new_config):
        del self.optimizer
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=new_config.get("lr", 0.01),
            momentum=new_config.get("momentum", 0.9))
        return True


class ucb_state:
    def __init__(self, n_total_arms=2):
        self.n_arms = n_total_arms
        self.n = 0
        self.selected = 0

    def bitfield(self, selected):
        bits = [int(digit) for digit in bin(selected)[2:]]  # [2:] to chop off the "0b" part
        size = len(bits)
        pad = [0] * (self.n_arms - size)
        return pad + bits


ucbstate = ucb_state()

scheduler = PopulationBasedTraining(
    time_attr="training_iteration",
    metric="mean_accuracy",
    mode="max",
    ucb=ucbstate,
    perturbation_interval=5,
    hyperparam_mutations={
        # distribution for resampling
        "lr": lambda: np.random.uniform(0.0001, 1),
        # allow perturbations within this set of categorical values
        "momentum": [0.8, 0.9, 0.99],
    }
)

# origin_dic_params = scheduler._hyperparam_mutations

numbers_of_selections = [0] * K
sums_of_reward = [0] * K
total_reward = 0  # total reward

resume = False

for e in range(N_EPISODE):
    print('--------------------------- episode started---------------------------')
    if e == 0:
        resume = False
    else:
        resume = True

    for n in range(N_EPISODE_STEP):
        print('--------------------------- episode step started---------------------------')
        selected = 0
        max_upper_bound = 0
        # scheduler._hyperparam_mutations = origin_dic_params

        for i in range(0, K):
            if numbers_of_selections[i] > 0:
                average_reward = sums_of_reward[i] / numbers_of_selections[i]
                delta_i = math.sqrt(2 * math.log(n+1) / numbers_of_selections[i])
                upper_bound = average_reward + delta_i
            else:
                upper_bound = 1e400
            if upper_bound > max_upper_bound:
                max_upper_bound = upper_bound
                selected = i

        ray.shutdown()  # Restart Ray defensively in case the ray connection is lost.
        ray.init(log_to_driver=False)

        masked_dic_params = {}
        ucbstate.selected = selected
        scheduler._ucb = ucbstate

        # for j, key in enumerate(origin_dic_params.keys()):
        #     if masks[j] == 1:
        #         masked_dic_params[key] = origin_dic_params[key]
        #     else:
        #         pass
        # scheduler._hyperparam_mutations = masked_dic_params

        numbers_of_selections[selected] += 1

        analysis = tune.run(
            PytorchTrainble,
            name="pbt_test",
            scheduler=scheduler,
            reuse_actors=True,
            verbose=1,
            stop={
                "training_iteration": 100,
            },
            resume=resume,
            num_samples=4,
            # PBT starts by training many neural networks in parallel with random hyperparameters.
            config={
                "lr": tune.uniform(0.001, 1),
                "momentum": tune.uniform(0.001, 1),
            })

        dfs = analysis.fetch_trial_dataframes()
        # This plots everything on the same plot
        # ax = None
        mean_value = 0

        for d in dfs.values():
            # ax = d.plot("training_iteration", "mean_accuracy", ax=ax, legend=False)
            mean_value += d.mean_accuracy.mean()

        # plt.xlabel("epoch");
        # plt.ylabel("Test Accuracy");

        sums_of_reward[selected] += mean_value
        total_reward += mean_value

    print(f'{e} episode, {selected} selected, {sums_of_reward} sum_of_rewards')

