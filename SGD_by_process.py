import os
import time
import numpy as np
from sklearn.metrics import mean_squared_error


"""
Conducts Linear Regression but initially transforms data using polynomial basis functions
Takes in an (N * 1) matrix, converts it into a (N * h) matrix
Performs linear regression on the (N*h) matrix resulting in h weights - betas
But this time linear regression is conducted through iterative gradient descent
Specifically stochastic gradient descent where we just choose a single sample from the the dataset
MSE as you iterate through the algorithm is shown
Returns the predictions only
"""


def lin_reg_poly_sgd(X_poly, y, h, alpha, n):
    beta_hat_poly = np.random.rand(h)
    for i in range(n):
        idx = np.random.randint(0, X_poly.shape[0])
        X_sample = X_poly[idx, :]
        y_sample = y[idx]
        y_hat_sample_poly = X_sample @ beta_hat_poly
        beta_hat_poly = beta_hat_poly - alpha * (
            X_sample.T * (y_hat_sample_poly - y_sample)
        )

        y_hat_poly = X_poly @ beta_hat_poly
        if i % 100000 == 0:
            print(
                "MSE in iteration",
                i,
                ": ",
                mean_squared_error(y, y_hat_poly),
                "in process id",
                os.getpid(),
            )
    return beta_hat_poly
