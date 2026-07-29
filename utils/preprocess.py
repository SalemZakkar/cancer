import numpy as np
import pandas as pd


X_mean = None
X_std = None

y_mean = None
y_std = None

numericNum = None


def clean(df):

    global numericNum

    df = df.copy()

    df.dropna()

    required = ["radius_mean"	,"texture_mean"	,"perimeter_mean"	,"area_mean"	,"smoothness_mean"]
    numericNum = len(required)
    df[required] = df[required].astype(float)
    df.dropna()
    df["diagnosis"] = df["diagnosis"].map({
    "M": 1,
    "B": 0
})
    X = df[required].values
    y = df['diagnosis'].values.reshape(-1,1)

    return X , y , df


# =====================================
# SCALE
# =====================================

def scale(X, y):

    X = X.copy()
    y = y.copy()

    global X_mean, X_std
    global y_mean, y_std
    global numericNum
    X_num = X[:, :numericNum]
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std == 0] = 1
    X = (
        X_num - X_mean
    ) / X_std
    y_mean = y.mean()
    y_std = y.std()
    if y_std == 0:
        y_std = 1

    y = (
        y - y_mean
    ) / y_std
    return X, y



def scaleX(data):
    global X_mean, X_std
    global numericNum
    
    # Split data
    data = data.copy()
    # Convert to float
    data = (data - X_mean) / X_std
    
    # Combine back
    scaled_data = data
    
    return scaled_data

def scaleY(data):

    global y_mean, y_std

    return (
        data - y_mean
    ) / y_std



def realY(scaledPrice):

    global y_mean, y_std

    return (
        scaledPrice * y_std
        +
        y_mean
    )