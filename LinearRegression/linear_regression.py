import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

np.random.seed(411)

housing = datasets.load_boston()
diabetes = datasets.load_diabetes()
n_samp = diabetes.data.shape[0]
n_feat = diabetes.data.shape[1]

def linear_regression(X, y):
    X_bias = add_bias(X)
    w = l2_loss_no_regularizer(X_bias, y)
    return w

def linear_regression_regularizer(X, y, l):
    w = l2_loss_l2_regularizer(add_bias(X), y, l)
    return w

def l2_loss_no_regularizer(X, y):
    # w^* = (X^T X)^-1 X^T y
    w = np.linalg.solve(np.transpose(X).dot(X), np.transpose(X).dot(y))
    return w

def l2_loss_l2_regularizer(X, y, l):
    # w^* = (X^T X + \lambda I)^-1 X^T y
    I = np.dot(l, np.identity(X.shape[1]))

    # We don't regularize the bias term
    I[0][0] = 0
    
    left = np.add(np.transpose(X).dot(X), I)
    w = np.linalg.solve(left, np.transpose(X).dot(y))
    return w

def add_bias(X):
    bias_X = np.empty((X.shape[0], X.shape[1] + 1))
    # Add bias column
    bias_X[:,0] = 1
    bias_X[:,1:] = X
    return bias_X

def split_data(X, y):
    split_size = int(X.shape[0] * 0.7)
    indices = np.random.choice(X.shape[0], split_size, replace = False)

    train_data = X[indices,:]
    train_target = y[indices]

    test_data = np.delete(X, indices, 0)
    test_target = np.delete(y, indices, 0)
    
    return (train_data, train_target, test_data, test_target)

def fit_polynomial(X, order):
    X_poly = np.empty((X.shape[0], X.shape[1] * order))
    for i in range(X.shape[0]):
        new_row = np.zeros(X.shape[1] * order)    
        for j in range(X.shape[1]):
            for k in range(order):
                new_row[j*order + k] = X[i][j] ** (k + 1)
        X_poly[i] = new_row
    return X_poly

def predict(X, y, w):
    X_bias = add_bias(X)
    y_hat = X_bias.dot(w)
    
    return np.mean((y - y_hat)**2)

def visualize(X, y):
    plt.figure(figsize=(20, 5))
    feature_count = X.shape[1]
    
    # i: index
    for i in range(feature_count):
        plt.subplot(3, 5, i + 1)
        plt.scatter(X[:,i], y)
    plt.tight_layout()
    plt.show()

def main():
    # Hyper-Parameters
    l = 0.00001
    deg = 4

    # Load data
    d = housing.data
    t = housing.target
    
    data = split_data(d, t)
    p_data = split_data(fit_polynomial(d, deg), t)

    #visualize(data[0], data[1])

    weights = linear_regression(data[0], data[1])
    p_weights = linear_regression_regularizer(p_data[0], p_data[1], l)

    mse = predict(data[2], data[3], weights)
    mse_poly = predict(p_data[2], p_data[3], p_weights)

    print("Non-Polynomial mse: {}".format(mse))
    print("Polynomial({}) mse (lambda={}): {}".format(deg, l, (mse_poly)))
          
if __name__ == "__main__":
    main()
