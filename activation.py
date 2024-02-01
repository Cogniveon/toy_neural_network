import numpy as np
from numpy.typing import ArrayLike

from layer import Layer


class TanH(Layer):
    def __init__(self):
        super().__init__()

    def _tanh(self, x):
        return np.tanh(x)

    def _inverse_tanh(self, x):
        return 1 - np.tanh(x) ** 2

    def forward(self, input):
        self.input = input
        return self._tanh(self.input)

    def backward(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, self._inverse_tanh(self.input))
