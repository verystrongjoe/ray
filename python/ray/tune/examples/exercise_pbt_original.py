"""
https://github.com/ray-project/tutorial/blob/master/tune_exercises/exercise_3_pbt.ipynb
참고!!!
"""

import tensorflow as tf
try:
    tf.get_logger().setLevel('INFO')
except Exception as exc:
    print(exc)
import warnings
warnings.simplefilter("ignore")

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
style.use("ggplot")

datasets.MNIST("~/data", train=True, download=True)


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


scheduler = PopulationBasedTraining(
    time_attr="training_iteration",
    metric="mean_accuracy",
    mode="max",
    perturbation_interval=5,
    hyperparam_mutations={
        # distribution for resampling
        "lr": lambda: np.random.uniform(0.0001, 1),
        # allow perturbations within this set of categorical values
        "momentum": [0.8, 0.9, 0.99],
    }
)

ray.shutdown()  # Restart Ray defensively in case the ray connection is lost.
ray.init(log_to_driver=False)

analysis = tune.run(
    PytorchTrainble,
    name="pbt_test",
    scheduler=scheduler,
    reuse_actors=True,
    checkpoint_freq=20,
    verbose=1,
    stop={
        "training_iteration": 500,
    },
    num_samples=8,
    # PBT starts by training many neural networks in parallel with random hyperparameters.
    config={
        "lr": tune.uniform(0.001, 1),
        "momentum": tune.uniform(0.001, 1),
    })


# Plot by wall-clock time
dfs = analysis.fetch_trial_dataframes()
# This plots everything on the same plot
ax = None
for d in dfs.values():
    ax = d.plot("training_iteration", "mean_accuracy", ax=ax, legend=False)

plt.xlabel("epoch"); plt.ylabel("Test Accuracy");