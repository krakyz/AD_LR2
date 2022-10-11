import numpy as np
import matplotlib.pyplot as plt
import gspread


def model(a, b, x):
    return a * x + b


def loss_function(a, b, x, y):
    num = len(x)
    prediction = model(a, b, x)
    return (0.5 / num) * (np.square(prediction - y)).sum()


def optimize(a, b, x, y, Lr):
    num = len(x)
    prediction = model(a, b, x)
    da = (1.0 / num) * ((prediction - y) * x).sum()
    db = (1.0 / num) * (prediction - y).sum()
    a = a - Lr * da
    b = b - Lr * db
    return a, b


def iterate(a, b, x, y, times, Lr):
    for i in range(times):
        a, b = optimize(a, b, x, y, Lr)
    return a, b


gc = gspread.service_account(filename='avian-influence-365213-de6d57216284.json')
sh = gc.open('LR2_T2')

x = [3, 21, 22, 34, 54, 34, 55, 67, 89, 99]
x = np.array(x)
y = [2, 22, 24, 65, 79, 82, 55, 130, 150, 99]
y = np.array(y)
Lr = 0.000001

times = [1, 2, 3, 4, 5, 6, 10, 100, 1000, 10000]

for i in range(1, len(times) + 1):
    a, b = np.random.rand(1), np.random.rand(1)
    a, b = iterate(a, b, x, y, times[i - 1], Lr)
    prediction = model(a, b, x)

    sh.sheet1.update(('A' + str(i)), str(i))
    sh.sheet1.update(('B' + str(i)), str(a[0]))
    sh.sheet1.update(('C' + str(i)), str(b[0]))
    sh.sheet1.update(('D' + str(i)), str(loss_function(a, b, x, y)))
    print(loss_function(a, b, x, y))
