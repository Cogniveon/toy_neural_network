import numpy as np
from numpy.typing import ArrayLike


def mse_loss(gt: ArrayLike, predictions: ArrayLike):
    return np.mean(np.power(gt - predictions, 2))


def mse_loss_prime(gt: ArrayLike, predictions: ArrayLike):
    return 2 * (predictions - gt) / np.size(gt)
