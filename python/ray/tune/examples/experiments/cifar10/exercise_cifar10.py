"""

https://towardsdatascience.com/deep-reinforcement-learning-and-hyperparameter-tuning-df9bf48e4bd2

keras
https://github.com/ray-project/ray/pull/1729/files
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
from ray.tune.examples.mnist_pytorch import train, test, ConvNet, get_data_loaders

import ray
from ray import tune
from ray.tune import run, sample_from, run_experiments
from ray.tune import track
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune.utils import validate_save_restore
from ray.tune import grid_search, run, sample_from



from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.datasets import cifar10
from tensorflow.python.keras.layers import Input, Dense, Dropout, Flatten
from tensorflow.python.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator


import matplotlib.style as style
import matplotlib.pyplot as plt
style.use("ggplot")
import random
import math
import heapq
import pickle
import argparse
from ray.tune import register_trainable
from ray.tune import Trainable

parser = argparse.ArgumentParser(description='PCB with Parameters')
parser.add_argument("-n_experiments", "--n_experiments", type=int, help="Number of experiments", default=2)
parser.add_argument("-n_workers", "--n_workers", type=int, help="Number of workers", default=4)
parser.add_argument("-ucb", "--ucb", action="store_true", help="turn on ucb")
parser.add_argument("-perturbation_interval", "--perturbation_interval", type=int, help="Perturbation Interval", default=3)
# parser.add_argument("-experiments", "--experiments", type=str, help="Experiments")
parser.add_argument("-training_iteration", "--training_iteration", type=int, help="Training Iteration", default=500)
parser.add_argument("-save_dir", "--save_dir", type=str, help="Training Iteration", default='data_500')

args = parser.parse_args()
print(f"args.n_experiments : {args.n_experiments}")
print(f"args.n_workers : {args.n_workers}")
print(f"args.perturbation_interval : {args.perturbation_interval}")
print(f"args.training_iteration : {args.training_iteration}")
print(f"args.ucb : {args.ucb}")
print(f"args.save_dir : {args.save_dir}")

############################################
# 실험 파라메터
############################################
N_EXPERIMENTS = args.n_experiments
TRAINING_ITERATION = args.training_iteration
# N_EPISODE_STEP = 5
DEFAULT_ACTION = 0
N_PARAMS = 3
K = int(math.pow(2, N_PARAMS))
NUM_WORKERS = args.n_workers
PERTUBATION_INTERVAL = args.perturbation_interval

IS_UCB = False
if args.ucb:
    IS_UCB = True

METRIC_NAME = 'mean_accuracy'
SAVE_DIR = args.save_dir

EXPERIMENT_NAME = 'pbt-cifar10-ucb-v1.0'
#############################################
# 실험 과정 및 결과 metrics
# - accuracy or reward over training iteration (plot)
# - cumulative selected count of bandit over training iteration (plot)
# - cumulative rewards of bandit over training iteration (plot)

# cartpole도 동일하게 실험 비교 (cartpole은 5개의 bandit)
# 위 실험이 너무 간단하게 끝나면 learning rate 낮추어 고정하고 나머지 조합으로 해보고
# 그래도 안되면 humanioid 로 실험 변경
# 적어도 2개의 실험 결과를 가져갈것
# 추후계획도 고민
#############################################
CUMULATIVE_REWARDS_EACH_BANDIT = [[]] * K
CUMULATIVE_SELECTED_COUNT_EACH_BANDIT = [[]] * K


num_classes = 10



class Cifar10Model(Trainable):

    def _read_data(self):
        # The data, split between train and test sets:
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

        # Convert class vectors to binary class matrices.
        y_train = tf.keras.utils.to_categorical(y_train, num_classes)
        y_test = tf.keras.utils.to_categorical(y_test, num_classes)

        x_train = x_train.astype('float32')
        x_train /= 255
        x_test = x_test.astype('float32')
        x_test /= 255

        return (x_train, y_train), (x_test, y_test)

    def _build_model(self, input_shape):
        x = Input(shape=(32, 32, 3))
        y = x
        y = Convolution2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu',
                          kernel_initializer='he_normal')(y)
        y = Convolution2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu',
                          kernel_initializer='he_normal')(y)
        y = MaxPooling2D(pool_size=2, strides=2, padding='same')(y)

        y = Convolution2D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu',
                          kernel_initializer='he_normal')(y)
        y = Convolution2D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu',
                          kernel_initializer='he_normal')(y)
        y = MaxPooling2D(pool_size=2, strides=2, padding='same')(y)

        y = Convolution2D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu',
                          kernel_initializer='he_normal')(y)
        y = Convolution2D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu',
                          kernel_initializer='he_normal')(y)
        y = MaxPooling2D(pool_size=2, strides=2, padding='same')(y)

        y = Flatten()(y)
        y = Dropout(self.config['dropout'])(y)
        y = Dense(units=10, activation='softmax', kernel_initializer='he_normal')(y)

        model = Model(inputs=x, outputs=y, name='model1')
        return model

    def _setup(self, config):
        self.train_data, self.test_data = self._read_data()
        x_train = self.train_data[0]
        model = self._build_model(x_train.shape[1:])

        opt = tf.keras.optimizers.Adadelta()
        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy'])
        self.model = model

    def _train(self):
        x_train, y_train = self.train_data
        x_test, y_test = self.test_data

        aug_gen = ImageDataGenerator(
            featurewise_center=False,             # set input mean to 0 over the dataset
            samplewise_center=False,              # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,   # divide each input by its std
            zca_whitening=False,                  # apply ZCA whitening
            rotation_range=0,                     # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,                # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,               # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,                 # randomly flip images
            vertical_flip=False,                  # randomly flip images
        )

        aug_gen.fit(x_train)
        gen = aug_gen.flow(x_train, y_train, batch_size=self.config['batch_size'])
        self.model.fit_generator(generator=gen,
                                 steps_per_epoch=50000 // self.config['batch_size'],
                                 epochs=self.config['epochs'],
                                 validation_data=None)

        # loss, accuracy
        _, accuracy = self.model.evaluate(x_test, y_test, verbose=0)
        return {'timesteps_this_iter' : 10,
                              'mean_accuracy': accuracy}

    def _save(self, checkpoint_dir):
        file_path = checkpoint_dir + '/model'
        self.model.save_weights(file_path)
        return file_path

    def _restore(self, path):
        self.model.load_weights(path)

    def _stop(self):
        # If need, save your model when exit.
        # saved_path = self.model.save(self.logdir)
        # print('save model at: ', saved_path)
        pass



############################################
# 아래는 UCB State 추가함
############################################

class ucb_state:
    def __init__(self, n_params=2, n_episode_iteration=5, optimal_exploration=True, default_action =0):
        self.n_params = n_params
        self.n = 0
        self.selected = 0
        self.K = int(math.pow(2, n_params))
        self.rewards = np.asarray([0.] * self.K)
        self.num_of_selections = np.asarray([0] * self.K)
        self.default_action = default_action
        self.n_episode_iteration = n_episode_iteration
        self.check_reflected_reward = True
        self.last_update_n_refleceted_reward = 0
        self.last_score = 0

        if optimal_exploration:
            for i in range(self.K):
                self.rewards[i] = np.sum(self.bitfield(i))

    def bitfield(self, selected):
        bits = [int(digit) for digit in bin(selected)[2:]]  # [2:] to chop off the "0b" part
        size = len(bits)
        pad = [0] * (self.n_params - size)
        return pad + bits

    # 요청마다 selected가 바뀌면 리워드 수집이 용이하지 않음. 그러므로 
    # 이터레이션을 지정받게 해서 그 주기만큼의 리워드를 보고 perturb를 몇번 해서 그간의 metric의 증가율을 보는것으로 평가
    def pull(self):
        self.n = self.n + 1
        if self.n != 0 and self.n > self.last_update_n_refleceted_reward and self.n < self.last_update_n_refleceted_reward + self.n_episode_iteration:
            return self.selected
        # elif self.n == self.n_episode_iteration + 1:
        else:
            selected = self.default_action
            # self.check_reflected_reward = False
            self.max_upper_bound = 0

            for i in range(1, self.K):
                if self.num_of_selections[i] > 0:
                    average_reward = self.rewards[i] / self.num_of_selections[i]
                    delta_i = math.sqrt(2 * math.log(self.n + 1) / self.num_of_selections[i])
                    upper_bound = average_reward + delta_i
                else:
                    upper_bound = 1e400
                if upper_bound > self.max_upper_bound:
                    self.max_upper_bound = upper_bound
                    selected = i
            self.num_of_selections[selected] = self.num_of_selections[selected] + 1
            self.selected = selected
            return selected
        # else:
        #     raise Exception(f"{self.last_update_n_refleceted_reward} 이후 metric 업데이트가 필요합니다.")

    # 스코어(accuracy or reward or any metric)을 저장한다.
    # reflected_reward에서는 지난 리워드 반영 체크해야함 
    # pull() 하기전에 reward를 반영해줘야함
    
    def reflect_reward(self, episode_reward):
        assert self.n != 0 and self.n % self.n_episode_iteration == 0
        # 이전 score와의 차이를 저장
        self.rewards[self.selected] += episode_reward - self.last_score
        self.last_score = episode_reward
        self.last_update_n_refleceted_reward = self.n
        # self.check_reflected_reward = True

    def is_need_to_reflect_reward(self):
        if self.n != 0 and self.n % self.n_episode_iteration == 0:
            return True
        else:
            return False

############################################


def experiment():

    ucbstate = None

    if IS_UCB:
        ucbstate = ucb_state(n_params=N_PARAMS)

    os.makedirs(SAVE_DIR, exist_ok=True)

    scheduler = PopulationBasedTraining(
        time_attr="training_iteration",
        metric=METRIC_NAME,
        mode="max",
        ucb=ucbstate,
        perturbation_interval=PERTUBATION_INTERVAL,
        hyperparam_mutations={
            "dropout": lambda: np.random.uniform(0, 1),
            "lr": lambda: np.random.uniform(0.0003, 0.003),
            "batch_size": lambda: random.choice([64, 128, 256, 512])
        }
    )

    ray.shutdown()  # Restart Ray defensively in case the ray connection is lost.
    # ray.init(log_to_driver=False, local_mode=True)
    ray.init(log_to_driver=False)

    # register_trainable('train_cifar10', Cifar10Model)

    analysis = tune.run(
        Cifar10Model,
        name=EXPERIMENT_NAME,
        scheduler=scheduler,
        # reuse_actors=True,
        checkpoint_freq=20,
        verbose=1,
        stop={
            "training_iteration": TRAINING_ITERATION,
        },
        num_samples=NUM_WORKERS,
        # resources_per_trial={"cpu": 10, "gpu": 0.25},
        # PBT starts by training many neural networks in parallel with random hyperparameters.
        config={
            "epochs": 1,
            "batch_size": 64,
            # "lr": grid_search([10**-4, 10**-5]),
            "lr": 1e-5,  # 1e-4,
            "decay": sample_from(lambda spec: spec.config.lr / 100.0),
            # "dropout": grid_search([0.25, 0.5]),
            "dropout": 0.5,
            "num_gpus":0.25,
        })

    # Plot by wall-clock time
    dfs = analysis.fetch_trial_dataframes()

    ## Save pickle
    with open(f"{SAVE_DIR}/{EXPERIMENT_NAME}_trials.pickle", "wb") as fw:
        pickle.dump(dfs, fw)

    # This plots everything on the same plot
    ax = None
    for d in dfs.values():
        ax = d.plot("training_iteration", METRIC_NAME, ax=ax, legend=False)

    if METRIC_NAME == 'mean_accuracy':
        a = np.asarray([list(dfs.values())[i].mean_accuracy.max() for i in range(NUM_WORKERS)])
    elif METRIC_NAME == 'episode_reward_mean':
        a = np.asarray([list(dfs.values())[i].episode_reward_mean.max() for i in range(NUM_WORKERS)])

    topk = heapq.nlargest(3, range(len(a)), a.__getitem__)
    sum = 0
    for i in topk:
        sum+= a[i]
    avg_top_k = sum /3

    plt.xlabel("epoch"); plt.ylabel("Test Accuracy");
    # plt.show()
    plt.savefig(f'{SAVE_DIR}/{EXPERIMENT_NAME}_accuracy.png')

    if IS_UCB:
        # bar chart
        fig, axs = plt.subplots(1, 2, figsize=(9,3))
        axs[0].bar(range(len(ucbstate.num_of_selections)-1), ucbstate.num_of_selections[1:])
        axs[1].bar(range(len(ucbstate.rewards)-1), ucbstate.rewards[1:])

        print(ucbstate.rewards)
        print(ucbstate.num_of_selections)

        ## Save pickle
        with open(f"{SAVE_DIR}/{EXPERIMENT_NAME}_bandit.pickle", "wb") as fw:
            pickle.dump(ucbstate, fw)

        plt.savefig(f'{SAVE_DIR}/{EXPERIMENT_NAME}_bandit_final.png')
        # plt.show()

    return avg_top_k


if __name__ == '__main__':

    final_results = []

    # for u in [True, False]:
    for u in [False]:
        list_accuracy = []
        for i in range(N_EXPERIMENTS):
            K = int(math.pow(2, N_PARAMS))
            IS_UCB = u
            IDX = i
            EXPERIMENT_NAME = f'pbt-cifar10-{IS_UCB}-{IDX}'
            list_accuracy.append(experiment())

        ## Save pickle
        with open(f"{SAVE_DIR}/{EXPERIMENT_NAME}_results.pickle", "wb") as fw:
            pickle.dump(list_accuracy, fw)
        print(f'{EXPERIMENT_NAME} list of accuracy : {list_accuracy}')
        print(f'average accuracy over {N_EXPERIMENTS} experiments ucb {u} : {np.average(list_accuracy)}')
        final_results.append(np.average(list_accuracy))

    print('============================final_result============================')
    print('UCB True: ', final_results[0])
    print('UCB False: ', final_results[1])
