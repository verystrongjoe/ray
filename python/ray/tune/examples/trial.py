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

def my_func(config, reporter):  # add the reporter parameter
    import time, numpy as np
    i = 0
    while True:
        reporter(timesteps_total=i, mean_accuracy=i ** config["alpha"])
        i += config["beta"]
        time.sleep(.01)


tune.register_trainable("my_func", my_func)
ray.init()

tune.run_experiments({
    "my_experiment": {
        "run": "my_func",
        "stop": { "mean_accuracy": 100 },
        "config": {
            "alpha": tune.grid_search([0.2, 0.4, 0.6]),
            "beta": tune.grid_search([1, 2]),
        }
    }
})