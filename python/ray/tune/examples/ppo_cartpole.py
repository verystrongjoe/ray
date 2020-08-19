import ray
from ray.tune.tune import run_experiments
from ray.tune.suggest.variant_generator import grid_search
import random

experiment = {
    'cartpole-ppo': {
        'run': 'DQN',
        'env': 'CartPole-v0',
        # 'resources_per_trial': {
        #     'cpu': 1,
        #     'gpu': 0},
        'stop': {
            'episode_reward_mean': 200,
            'time_total_s': 180
        },
        'config': {
            'num_sgd_iter': grid_search([1, 4]),
            'num_workers': 2,
            # 'use_critic' : random.choice([True, False]),
            # "train_batch_size": random.choice([1024, 2048, 4096]),
            "lr": [1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
            # 'sgd_minibatch_size': grid_search([128, 256, 512])
        }
    },
    # put additional experiments to run concurrently here
}

ray.init()
run_experiments(experiment)

