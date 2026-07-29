import numpy as np


def binary_cross_entropy(pred, y):

    eps = 1e-8

    return -np.mean(
        y * np.log(pred + eps)
        +
        (1 - y) * np.log(1 - pred + eps)
    )


def leaky_relu_derivative(x):

    return np.where(
        x > 0,
        1,
        0.01
    )


def backward(
        model,
        X,
        y,
        Z1,
        A1,
        Z2,
        A2,
        Z3,
        A3,
        pred,
):
    # x -> w1 -> z1 -> a1 -> w2 -> z2 -> a2 -> w3 -> z3 -> a3 -> w4 -> out

    m = X.shape[0]


    # sigmoid + BCE
    dZ4 = pred - y


    dW4 = (A3.T @ dZ4) / m

    db4 = np.sum(
        dZ4,
        axis=0,
        keepdims=True
    ) / m



    dE3 = (
        dZ4 @ model.W4.T
    ) * leaky_relu_derivative(Z3)


    dW3 = (A2.T @ dE3) / m

    db3 = np.sum(
        dE3,
        axis=0,
        keepdims=True
    ) / m



    dE2 = (
        dE3 @ model.W3.T
    ) * leaky_relu_derivative(Z2)


    dW2 = (A1.T @ dE2) / m

    db2 = np.sum(
        dE2,
        axis=0,
        keepdims=True
    ) / m



    dE1 = (
        dE2 @ model.W2.T
    ) * leaky_relu_derivative(Z1)


    dW1 = (X.T @ dE1) / m

    db1 = np.sum(
        dE1,
        axis=0,
        keepdims=True
    ) / m


    return [
        dW1,
        db1,

        dW2,
        db2,

        dW3,
        db3,

        dW4,
        db4
    ]



def train(
        model,
        X,
        y,
        epochs=1000,
        lr=0.002
):

    best_loss = float("inf")
    best = None


    for epoch in range(epochs):


        Z1,A1,Z2,A2,Z3,A3,pred = model.forward(X)


        loss = binary_cross_entropy(
            pred,
            y
        )


        grads = backward(
            model,
            X,
            y,
            Z1,
            A1,
            Z2,
            A2,
            Z3,
            A3,
            pred
        )


        model.W1 -= lr * grads[0]
        model.b1 -= lr * grads[1]


        model.W2 -= lr * grads[2]
        model.b2 -= lr * grads[3]


        model.W3 -= lr * grads[4]
        model.b3 -= lr * grads[5]


        model.W4 -= lr * grads[6]
        model.b4 -= lr * grads[7]



        if loss < best_loss:

            best_loss = loss

            best = [
                model.W1.copy(),
                model.b1.copy(),

                model.W2.copy(),
                model.b2.copy(),

                model.W3.copy(),
                model.b3.copy(),

                model.W4.copy(),
                model.b4.copy()
            ]



        if epoch % 100 == 0:

            print(
                "epoch:",
                epoch,
                "loss:",
                loss
            )


    (
        model.W1,
        model.b1,

        model.W2,
        model.b2,

        model.W3,
        model.b3,

        model.W4,
        model.b4

    ) = best


    return model