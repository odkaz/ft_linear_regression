# from math import sqrt
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# def mse_(y, y_hat):
#     m = len(y)
#     sum = 0
#     for i in range(m):
#         sum += (y_hat[i] - y[i]) ** 2
#     return sum / m

# def rmse_(y, y_hat):
#     mse = mse_(y, y_hat)
#     return sqrt(mse)

# def mae_(y, y_hat):
#     m = len(y)
#     sum = 0
#     for i in range(m):
#         sum += abs(y_hat[i] - y[i])
#     return sum / m

# def mean_(y):
#     m = len(y)
#     sum = 0
#     for i in range(m):
#         sum = y[i]
#     return sum / m

# def r2score_(y, y_hat):
#     m = len(y)
#     y_mean = mean_(y)
#     sum_up, sum_btm = 0, 0
#     for i in range(m):
#         sum_up += (y_hat[i] - y) ** 2
#         sum_btm += (y[i] - y_mean) ** 2
#     return 1 - (sum_up / sum_btm)

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

    res_y = [estimate_price(theta0, theta1, x) for x in x_train]
    plt.plot(x_train, res_y, c='b', label='Our Prediction')
    plt.scatter(x_train, y_train, marker='x', c='r', label='Actual Values')
    plt.show()

    return theta0, theta1


def main():
    t0, t1 = linear_regression()




    res = {'t0': t0, 't1': t1}

if __name__ == '__main__':
    main()
