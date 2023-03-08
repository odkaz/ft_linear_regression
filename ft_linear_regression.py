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

def linear_regression():
    df = pd.read_csv('./data.csv')
    km = df["km"].to_list()
    price = df["price"].to_list()
    x_train = np.array(km)
    y_train = np.array(price)
    learning_rate = 0.001
    iterate = 10
    theta0, theta1 = 00,0

    for _ in range(0, iterate):
        tmp0, tmp1 = theta0, theta1
        theta0 -= learning_rate * get_bias(tmp0, tmp1, x_train, y_train)
        theta1 -= learning_rate * get_weight(tmp0, tmp1, x_train, y_train)
        print('theta0', theta0, 'theta1', theta1)

    res_y = [estimate_price(theta0, theta1, x) for x in x_train]
    plt.plot(x_train, res_y, c='b', label='Our Prediction')
    plt.scatter(x_train, y_train, marker='x', c='r', label='Actual Values')
    plt.show()
    return theta0, theta1

def main():
    t0, t1 = linear_regression()

if __name__ == '__main__':
    main()
