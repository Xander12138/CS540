import numpy as np
from matplotlib import pyplot as plt
import csv
import math


# Feel free to import other packages, if needed.
# As long as they are supported by CSL machines.


def get_dataset(filename):
    dataset = []
    with open(filename, newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            line = []
            for i in range(1, len(row)):
                line.append(float(list(row.items())[i][1]))
            dataset.append(line)
    return np.array(dataset)


def print_stats(dataset, col):
    mean = 0
    for num in dataset[:, col]:
        mean += float(num)
    mean = mean / dataset.shape[0]
    temp = []
    for num in dataset[:, col]:
        temp.append((float(num) - mean) ** 2)
    std = math.sqrt(sum(temp) / (len(dataset) - 1))
    print(len(dataset))
    print(round(mean, 2))
    print(round(std, 2))


def regression(dataset, cols, betas):
    y_hat = dataset[:, 0]
    mse = 0.0
    for col in range(len(cols)):
        mse += betas[col + 1] * dataset[:, cols[col]]
    mse = sum((mse + betas[0] - y_hat) ** 2) / len(y_hat)
    return mse


def gradient_descent(dataset, cols, betas):
    y = dataset[:, 0]
    grads = []
    for k in range(len(betas)):
        if k == 0:
            mse = 0.0
            for col in range(len(cols)):
                mse += betas[col + 1] * dataset[:, cols[col]]
            grad = sum((mse + betas[0] - y) * 2 / len(y))
        else:
            mse = 0.0
            for col in range(len(cols)):
                mse += betas[col + 1] * dataset[:, cols[col]]
            grad = sum((mse + betas[0] - y) * 2 * dataset[:, cols[k - 1]] / len(y))
        grads.append(grad)
    return grads


def iterate_gradient(dataset, cols, betas, T, eta):
    for i in range(T):
        desc = gradient_descent(dataset, cols, betas)
        betas = betas - np.dot(eta, desc)
        mse = regression(dataset, cols, betas)
        print(i+1, " ", round(mse, 2), end=" ")
        for beta in betas:
            print(f"{beta:.2f}", end=' ')
        print("")
    pass


def compute_betas(dataset, cols):
    x = []
    y = dataset[:, 0]

    for i in range(len(dataset)):
        line = [1]
        for col in cols:
            line.append(dataset[i][col])
        x.append(line)

    x = np.array(x, dtype='float')
    beta = (np.linalg.pinv(x.T.dot(x)).dot(x.T)).dot(y)
    mse = regression(dataset, cols, beta)
    return (mse, *beta)


def predict(dataset, cols, features):
    betas = compute_betas(dataset, cols)
    result = 0
    for i in range(len(cols)):
        result += features[i] * betas[i + 2]
    return result + betas[1]


def synthetic_datasets(betas, alphas, X, sigma):
    pass


def plot_mse():
    from sys import argv
    if len(argv) == 2 and argv[1] == 'csl':
        import matplotlib
        matplotlib.use('Agg')


if __name__ == '__main__':
    ### DO NOT CHANGE THIS SECTION ###
    plot_mse()