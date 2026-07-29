import numpy as np



def report(
    model,
    prediction,
    y,
    A1,
    A2,
    A3
):

    print("\n========== FINAL LOSS ==========")


    loss = np.mean(
        (prediction - y) ** 2
    )


    print("Loss:")
    print(loss)




    print("\n========== WEIGHT MAGNITUDE ==========")


    print(
        "W1 mean:",
        np.mean(np.abs(model.W1))
    )


    print(
        "W2 mean:",
        np.mean(np.abs(model.W2))
    )


    print(
        "W3 mean:",
        np.mean(np.abs(model.W3))
    )


    print(
        "W4 mean:",
        np.mean(np.abs(model.W4))
    )






    print("\n========== PREDICTION VS REAL ==========")


    print("Predictions:")
    print(prediction[:10])


    print("\nTargets:")
    print(y[:10])






    print("\n========== STATISTICS ==========")


    print(
        "Prediction mean:",
        prediction.mean()
    )


    print(
        "Prediction std:",
        prediction.std()
    )


    print(
        "Y mean:",
        y.mean()
    )


    print(
        "Y std:",
        y.std()
    )







    print("\n========== RELU STATUS ==========")


    print(
        "Layer 1 dead ratio:",
        np.sum(A1 == 0) / A1.size
    )


    print(
        "Layer 2 dead ratio:",
        np.sum(A2 == 0) / A2.size
    )


    print(
        "Layer 3 dead ratio:",
        np.sum(A3 == 0) / A3.size
    )







    print("\n========== ERROR ==========")


    errors = prediction - y


    print(
        "MAE:",
        np.mean(np.abs(errors))
    )


    print(
        "MAX ERROR:",
        np.max(np.abs(errors))
    )