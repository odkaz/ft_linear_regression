from math import sqrt

def mse_(y, y_hat):
    m = len(y)
    sum = 0
    for i in range(m):
        sum += (y_hat[i] - y[i]) ** 2
    return sum / m

def rmse_(y, y_hat):
    mse = mse_(y, y_hat)
    return sqrt(mse)

def mae_(y, y_hat):
    m = len(y)
    sum = 0
    for i in range(m):
        sum += abs(y_hat[i] - y[i])
    return sum / m

def mean_(y):
    m = len(y)
    sum = 0
    for i in range(m):
        sum = y[i]
    return sum / m

def r2score_(y, y_hat):
    m = len(y)
    y_mean = mean_(y)
    sum_up, sum_btm = 0, 0
    for i in range(m):
        sum_up += (y_hat[i] - y) ** 2
        sum_btm += (y[i] - y_mean) ** 2
    return 1 - (sum_up / sum_btm)

def main():
    pass