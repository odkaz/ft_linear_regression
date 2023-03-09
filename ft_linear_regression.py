import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

def data_normalisation(data):
    low = min(data)
    high = max(data)
    res = []
    for x in data:
        norm = (x - low) / (high - low)
        res.append(norm)
    return res, low

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
    x_norm, norm = data_normalisation(x_train)

    for _ in range(0, iterate):
        tmp0, tmp1 = theta0, theta1
        theta0 -= learning_rate * get_bias(tmp0, tmp1, x_norm, y_train)
        theta1 -= learning_rate * get_weight(tmp0, tmp1, x_norm, y_train)

    res_y = [estimate_price(theta0, theta1, x) for x in x_norm]
    plt.plot(x_train, res_y, c='b', label='Our Prediction')
    plt.scatter(x_train, y_train, marker='x', c='r', label='Actual Values')
    plt.show()
    return theta0, theta1
