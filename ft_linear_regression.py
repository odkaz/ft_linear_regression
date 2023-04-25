import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

def estimate_price(bias, weight, x):
    y =  bias + weight * x
    return y

def get_bias(bias, weight, milage, price):
    m = len(milage)
    sum = 0
    for i in range(m):
        sum += (estimate_price(bias, weight, milage[i]) - price[i])
    return sum / m

def get_weight(bias, weight, milage, price):
    m = len(milage)
    sum = 0
    for i in range(m):
        sum += (estimate_price(bias, weight, milage[i]) - price[i]) * milage[i]
    return sum / m

def mean_(y):
    m = len(y)
    sum = 0
    for i in range(m):
        sum = y[i]
    return sum / m

def get_stddev(data):
    variance = data - mean_(data)
    square = variance * variance
    sum = np.sum(square)
    res = math.sqrt(sum / (len(data) - 1))
    return res

def data_standardization(data):
    mean = mean_(data)
    standard = get_stddev(data)
    res = (data - mean) / standard
    return res

def unnormalize_weight(weight, data):
    standard = get_stddev(data)
    return weight / standard

def unnormalize_bias(theta0, theta1, data):
    mean = mean_(data)
    standard = get_stddev(data)
    return theta0 - ((theta1 * mean) / standard)

def get_train_data():
    url = './data.csv'
    df = pd.read_csv(url)
    km = df["km"].to_list()
    price = df["price"].to_list()
    x_train = np.array(km)
    y_train = np.array(price)
    return x_train, y_train

def linear_regression():
    x_train, y_train = get_train_data()
    learning_rate = 0.001
    iterate = 100000
    theta0, theta1 = 0,0
    x_norm = data_standardization(x_train)

    for _ in range(0, iterate):
        tmp0, tmp1 = theta0, theta1
        theta0 -= learning_rate * get_bias(tmp0, tmp1, x_norm, y_train)
        theta1 -= learning_rate * get_weight(tmp0, tmp1, x_norm, y_train)

    theta0 = unnormalize_bias(theta0, theta1, x_train)
    theta1 = unnormalize_weight(theta1, x_train)

    return theta0, theta1
