import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import copy


def compute_model_output(x, w, b):
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b
    return f_wb

def compute_cost(x, y, w, b):
    m = x.shape[0]

    cost_sum = 0
    for i in range(m):
        f_wb = w * x[i] + b
        cost = (f_wb - y[i]) ** 2
        cost_sum += cost
    total_cost = (1 / (2 * m)) * cost_sum
    return total_cost

def compute_gradient(x, y, w, b):
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0
    print('w: ', w, ' b: ', b)
    print('res', w * x[0] + b)
    for i in range(m):
        f_wb = w * x[i] + b
        dj_dw_i = (f_wb - y[i]) * x[i]
        dj_db_i = f_wb - y[i]
        dj_db += dj_db_i
        dj_dw += dj_dw_i
    dj_dw = dj_dw / m
    dj_db = dj_db / m
    print('dw: ', dj_dw, ' db: ', dj_db)

    return dj_dw, dj_db


def gradient_descent(x, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function):
    w = copy.deepcopy(w_in)
    J_history = []
    p_history = []
    b = b_in
    w = w_in
    for i in range(num_iters):
        dj_dw, dj_db = gradient_function(x, y, w, b)
        b = b - alpha * dj_db
        w = w - alpha * dj_dw
        # print('b : ', b, ' w: ', w)
        if i < 10000:
            J_history.append(cost_function(x, y , w, b))
            p_history.append([w, b])
        
        # if (i % math.ceil(num_iters / 10) == 0):
        #     print(f"Iteration {i:4}: Cost {J_history[-1]:0.2e} ",
        #     f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  ",
        #     f"w: {w: 0.3e}, b:{b: 0.5e}")

    return w, b, J_history, p_history







df = pd.read_csv('./data.csv')
km = df["km"].to_list()
price = df["price"].to_list()
x_train = np.array(km)
y_train = np.array(price)
w = -0.024
b = 9000
w_init = 0
b_init = 0
iterations = 10
tmp_alpha = 1.0e-2

w_final, b_final, J_hist, p_hist = gradient_descent(x_train, y_train, w_init, b_init, tmp_alpha, iterations, compute_cost, compute_gradient)
print(f"(w,b) found by gradient descent: ({w_final:12.8f},{b_final:12.8f})")
f_wb = compute_model_output(x_train, w_final, b_final)


# gradient_descent
# fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))
# ax1.plot(J_hist[:10])
# ax2.plot(10 + np.arange(len(J_hist[10:])), J_hist[10:])
# ax1.set_title("Cost vs. iteration(start)");  ax2.set_title("Cost vs. iteration (end)")
# ax1.set_ylabel('Cost')            ;  ax2.set_ylabel('Cost') 
# ax1.set_xlabel('iteration step')  ;  ax2.set_xlabel('iteration step')

# print(f"1300 sqft house prediction {w_final*75000 + b_final:0.1f} Thousand dollars")
# print(f"1200 sqft house prediction {w_final*125000 + b_final:0.1f} Thousand dollars")
# print(f"2000 sqft house prediction {w_final*175000 + b_final:0.1f} Thousand dollars")




plt.plot(x_train, f_wb, c='b', label='Our Prediction')
# plt_gradients(x_train, y_train, compute_cost, compute_gradient)
plt.scatter(x_train, y_train, marker='x', c='r', label='Actual Values')

# plt.title("Car prices")
# plt.ylabel('price (USD)')
# plt.xlabel('distance ran(km)')
# plt.legend()

plt.show()
