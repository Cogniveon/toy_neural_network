import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from activation import TanH
from functions import mse_loss, mse_loss_prime
from layer import DenseLayer, Layer


def train(network, X, Y, epochs=10000, learning_rate=0.1):
    with tqdm(range(epochs), desc="training") as iterator:
        for e in iterator:
            error = 0

            for x, y in zip(X, Y):
                output = x

                for layer in network:
                    output = layer.forward(output)

                error += mse_loss(y, output)

                grad = mse_loss_prime(y, output)
                for layer in reversed(network):
                    grad = layer.backward(grad, learning_rate)

            error /= len(X)
            # print("%d/%d, error=%f" % (e + 1, epochs, error))
            iterator.set_postfix(error=("%.6f" % error))

    return network


def plot_xor_graph(network):
    points = []
    for x in np.linspace(0, 1, 20):
        for y in np.linspace(0, 1, 20):
            z = [[x], [y]]
            for layer in network:
                z = layer.forward(z)
            points.append([x, y, z[0, 0]])

    points = np.array(points)

    fig = plt.figure()
    plt.title("XOR")
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 2], cmap="winter")
    plt.show()


if __name__ == "__main__":
    network: list[Layer] = [DenseLayer(2, 3), TanH(), DenseLayer(3, 1), TanH()]
    X = np.reshape(
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ],
        (4, 2, 1),
    )
    Y = np.reshape(
        [
            [0],
            [1],
            [1],
            [0],
        ],
        (4, 1, 1),
    )

    train(network, X, Y)
    plot_xor_graph(network)
