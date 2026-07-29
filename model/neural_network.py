import numpy as np


class NeuralNetwork:

    def __init__(self, input_size):

        np.random.seed(42)

        self.W1 = (
            np.random.randn(input_size, 32)
            * np.sqrt(2 / input_size)
        )
        self.b1 = np.zeros((1, 32))

        self.W2 = (
            np.random.randn(32, 16)
            * np.sqrt(2 / 32)
        )
        self.b2 = np.zeros((1, 16))

        self.W3 = (
            np.random.randn(16, 8)
            * np.sqrt(2 / 16)
        )
        self.b3 = np.zeros((1, 8))

        self.W4 = (
            np.random.randn(8, 1)
            * np.sqrt(2 / 8)
        )
        self.b4 = np.zeros((1, 1))


    def relu(self, x):
        return np.where(x > 0, x, 0.01 * x)


    def forward(self, X):

        Z1 = X @ self.W1 + self.b1
        A1 = self.relu(Z1)

        Z2 = A1 @ self.W2 + self.b2
        A2 = self.relu(Z2)

        Z3 = A2 @ self.W3 + self.b3
        A3 = self.relu(Z3)

        out = A3 @ self.W4 + self.b4

        return (
            Z1,
            A1,
            Z2,
            A2,
            Z3,
            A3,
            out
        )