import pandas as pd
import numpy as np

from utils.preprocess import clean , scale , scaleX ,scaleY , realY 
from model.neural_network import NeuralNetwork
from trainer.trainer import train
from diagnostics.diagnostics import report

from sklearn.model_selection import train_test_split


df = pd.read_csv(
    "data.csv"
)


X , y , cdf = clean(df)


X_train, X_test, y_train, y_test ,cdf_train, cdf_test = train_test_split(
    X,
    y,
    cdf,
    test_size=0.01,
    random_state=42
)
# # np.set_printoptions(
# #     threshold=50,
# #     precision=2,
# #     suppress=False
# # )
# # for value in y_test[0]:
# #     print(value)


# # print()

print(
    "Train:",
    X_train.shape
)

print(
    "Test:",
    X_test.shape
)

X_train , y_train = scale(X_train , y_train) 

model = NeuralNetwork(
    input_size=X_train.shape[1]
)

train(
    model,
    X_train,
    y_train,
    lr = 0.01,
    epochs = 1
)




# # report(
# #     model,
# #     prediction,
# #     y_test,
# #     A1,
# #     A2,
# #     A3
# # )

for i in range(len(X_test)):

    Z1,A1,Z2,A2,Z3,A3,prediction = model.forward(
    scaleX(X_test[i])
)

    print()

    print(
        "Sample:",
        i
    )

    print(
        "Predicted:",
        realY(prediction[0][0])
    )


    print(
        "Actual:",
        y_test[i][0]
    )


    print(
        "Difference:",
        round(
            abs(
                y_test[i][0]
                -
                realY(prediction[0][0])
            ),
            2
        )
    )

    print("-----------------------------------------------------------")