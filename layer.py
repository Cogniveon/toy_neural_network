import numpy as np
from numpy.typing import ArrayLike


class Layer:
    def __init__(self):
        self.input: ArrayLike = None
        self.output: ArrayLike = None

    def forward(self, input: ArrayLike):
        raise NotImplementedError(
            f'Layer [{type(self).__name__}] is missing the required "forward" function'
        )

    def backward(self, output_gradient: ArrayLike, learning_rate: float):
        raise NotImplementedError(
            f'Layer [{type(self).__name__}] is missing the required "backward" function'
        )


class DenseLayer(Layer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)

    def forward(self, input):
        self.input = input
        return np.dot(self.weights, input) + self.bias

    def backward(self, output_gradient, learning_rate):
        weights_gradient = np.dot(output_gradient, self.input.T)
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient
        return np.dot(self.weights.T, output_gradient)
