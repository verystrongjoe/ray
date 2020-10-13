#!/usr/bin/env python
"""Example of using PBT with RLlib.

Note that this requires a cluster with at least 8 GPUs in order for all trials
to run concurrently, otherwise PBT will round-robin train the trials which
is less efficient (or you can set {"gpu": 0} to use CPUs for SGD instead).

Note that Tune in general does not need 8 GPUs, and this is just a more
computationally demainding example.
"""

import random
import ray
from ray.tune import run, sample_from, run_experiments
from ray.tune.schedulers import PopulationBasedTraining
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

N_EPISODE = 5000
N_EPISODE_STEP = 1
N_PARAMS = 2
K = int(math.pow(2, N_PARAMS)) - 2


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


if __name__ == "__main__":

    ucbstate = ucb_state()

    # Postprocess the perturbed config to ensure it's still valid
    def explore(config):
        # ensure we collect enough timesteps to do sgd
        # if config["train_batch_size"] < config["sgd_minibatch_size"] * 2:
        #     config["train_batch_size"] = config["sgd_minibatch_size"] * 2
        # ensure we run at least one sgd iter
        # if config["num_sgd_iter"] < 1:
        #     config["num_sgd_iter"] = 1
        return config


    numbers_of_selections = [0] * K
    sums_of_reward = [0] * K
    total_reward = 0  # total reward

    resume = False

    pbt = PopulationBasedTraining(
        time_attr="time_total_s",
        metric="episode_reward_mean",
        mode="max",
        ucb=ucbstate,
        perturbation_interval=120,
        resample_probability=0.25,
        # Specifies the mutations of these hyperparams
        hyperparam_mutations={
            # "lr": [1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
            "train_batch_size": lambda: random.randint(32, 256),
            # "buffer_size": [5000, 10000, 50000],
            # "prioritized_replay": [True, False],
            "target_network_update_freq": lambda: random.randint(10, 1000)
        },
        custom_explore_fn=explore)

    for e in range(N_EPISODE):
        print(f'--------------------------- {e} episode started---------------------------')
        if e == 0:
            resume = False
        else:
            resume = False

        for n in range(N_EPISODE_STEP):
            print(f'--------------------------- episode step {n}---------------------------')
            selected = 0
            max_upper_bound = 0
            # scheduler._hyperparam_mutations = origin_dic_params

            for i in range(0, K):
                if numbers_of_selections[i] > 0:
                    average_reward = sums_of_reward[i] / numbers_of_selections[i]
                    delta_i = math.sqrt(2 * math.log(n + 1) / numbers_of_selections[i])
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

            numbers_of_selections[selected] += 1

            score = run_experiments({
                "pbt_cartpole_run_experiments": {
                    "run": "DQN",
                    "env": "CartPole-v0",
                    "stop": {"episodes_this_iter": 5*(e+1)},
                    "config": {
                        "num_workers": 5,
                        "num_gpus": 0,
                        # "num_cpus": 128,
                        "lr": 1e-4,
                        "train_batch_size": sample_from(
                            lambda spec: random.choice([32, 64, 128, 256])),
                        "buffer_size": 10000,
                        "target_network_update_freq": sample_from(
                            lambda spec: random.choice([10, 100, 500, 1000]))
                    },
                    "num_samples": 5,
                    # "resources_per_trial": {
                    #                           "cpu": 1,
                    #                           "gpu": 0,
                    #                           # "extra_cpu": lambda spec : spec.config.num_workers
                    #                       },
                },

            },
                scheduler=pbt,
                resume=resume
            )
            mean_value = np.mean([t.last_result['episode_reward_mean'] for t in score])
            sums_of_reward[selected] += mean_value
            total_reward += mean_value

            print(f'{e} episode, {selected} selected, {sums_of_reward} sum_of_rewards')

    # local_mode = True ---> single process!!!
    # ray.init(num_cpus=128, local_mode=True)

    # fail_fast = True --->
    # run(
    #     "DQN",
    #     stop={"training_iteration": iters},
    #     name="pbt_cartpole_test",
    #     scheduler=pbt,
    #     verbose=0,
    #     # 여기에 샘플수 조정해서 하면 멀티 프로세싱 처리가 가능하다.왜 num_worker로 조정된다는건지 인터넷에선..
    #     num_samples=32,
    #     config={
    #         "env": "CartPole-v0",
    #         # "num_workers": 64,
    #         # "num_gpus": 0,
    #         # 'resources': {
    #         #     'cpu': 64},
    #         "lr": 1e-4,
    #         "train_batch_size":
    #             sample_from(
    #                 lambda spec: random.choice([32, 64, 128, 256])),
    #         "buffer_size": 10000,
    #         "target_network_update_freq": sample_from(
    #                 lambda spec: random.choice([10, 100, 500, 1000]))
    #     })


