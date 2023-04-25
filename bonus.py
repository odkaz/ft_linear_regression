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

def unnormalize_bias(b, w, data):
    mean = mean_(data)
    standard = get_stddev(data)
    return b - ((w * mean) / standard)

def get_train_data():
    url = './data.csv'
    df = pd.read_csv(url)
    km = df["km"].to_list()
    price = df["price"].to_list()
    x_train = np.array(km)
    y_train = np.array(price)
    return x_train, y_train

def compute_cost(x, y, w, b):
   
    m = x.shape[0] 
    cost = 0
    
    for i in range(m):
        f_wb = w * x[i] + b
        cost = cost + (f_wb - y[i])**2
    total_cost = 1 / (2 * m) * cost

    return total_cost

from celluloid import Camera

def linear_regression():
    x_train, y_train = get_train_data()
    learning_rate = 0.01
    iterate = 10000
    b, w = 0,0
    x_norm = data_standardization(x_train)
    J_history = []
    p_history = []

    for _ in range(0, iterate):
        tmp0, tmp1 = b, w
        b -= learning_rate * get_bias(tmp0, tmp1, x_norm, y_train)
        w -= learning_rate * get_weight(tmp0, tmp1, x_norm, y_train)

        if _< iterate:      # prevent resource exhaustion 
            J_history.append( compute_cost(x_norm, y_train, w , b))
            p_history.append([w, b])

    b = unnormalize_bias(b, w, x_train)
    w = unnormalize_weight(w, x_train)

    res_y = [estimate_price(b, w, x) for x in x_train]
    # plt.plot(x_train, res_y, c='b', label='Our Prediction')
    # plt.scatter(x_train, y_train, marker='x', c='r', label='Actual Values')
    # plt.show()



    fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12,4))
    camera = Camera(fig)

    # ax1.plot(x_train, res_y, c='b', label='Our Prediction')
    # ax1.scatter(x_train, y_train, marker='x', c='r', label='Actual Values')
    show = 1
    buf = 10000


    for i in range(0, iterate):
        if (i == show):
            animate_y = [estimate_price(p_history[i][1], p_history[i][0], x) for x in x_norm]


            ax1.plot(x_train, animate_y, c='r', label='Our Prediction')

            if (i < buf):
                ax2.scatter(i, J_history[i], marker='o', c='r', label='cost')
            # ax2.scatter(i, J_history[i], marker='o', c='r', label='cost')

            show*= 2
            camera.snap()



    ax1.plot(x_train, res_y, c='b', label='Our Prediction')
    ax1.scatter(x_train, y_train, marker='x', c='g', label='Actual Values')
    ax2.plot(J_history[:buf])

    # ax2.plot(buf + np.arange(len(J_history[buf:])), J_history[buf:])
    ax1.set_title("Price vs. Milage");  ax2.set_title("Cost vs. Iteration")
    ax1.set_ylabel('Price')            ;  ax2.set_ylabel('Cost') 
    ax1.set_xlabel('Milage')  ;  ax2.set_xlabel('iteration step')



    animation = camera.animate()
    animation.save('celluloid_legends.gif', writer = 'imagemagick')
    plt.show()

    return b, w


def main():
    t0, t1 = linear_regression()




    res = {'t0': t0, 't1': t1}

if __name__ == '__main__':
    main()
