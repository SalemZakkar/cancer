import numpy as np



def mse(pred, y):

    return np.mean(
        (pred - y) ** 2
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

    m = X.shape[0]


    d_pred = (2 / m) * (pred - y)

    # print(d_pred)

    # input -> W1S -> A1S -> W2S -> A2S -> W3S -> A3S -> W4S -> A4S -> output


    dW4 = A3.T @ d_pred

    db4 = np.sum(
        d_pred,
        axis=0,
        keepdims=True
    )





    dE3 = ((d_pred @ model.W4.T) * leaky_relu_derivative(Z3))
    dW3 = A2.T @ dE3

    db3 = np.sum(
        dE3,
        axis=0,
        keepdims=True
    )


    dE2 = (dE3 @ model.W3.T) * leaky_relu_derivative(Z2)

    dW2 = A1.T @ dE2

    db2 = np.sum(
        dE2,
        axis=0,
        keepdims=True
    )


    dE1 = (dE2 @ model.W2.T) * leaky_relu_derivative(Z1)

    dW1 = X.T @ dE1

    db1 = np.sum(
        dE1,
        axis=0,
        keepdims=True
    )

    


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


        Z1,A1,Z2,A2,Z3,A3,pred = model.forward(
            X
        )


        loss = mse(
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
            pred,
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
                epoch,
                loss,
                "lr:",
                lr
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