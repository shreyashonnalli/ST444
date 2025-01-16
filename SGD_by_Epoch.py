import os
import numpy as np
from sklearn.metrics import mean_squared_error

def shuffle_data(X_poly, y):
    """
    Shuffles the data for each processor independently (shuffled once at the start).

    Parameters:
    - X_poly: (N * h) matrix of input features.
    - y: Target variable (N * 1).

    Returns:
    - X_poly_shuffled: Shuffled matrix of input features for each process.
    - y_shuffled: Shuffled target variable for each process.
    """
    shuffled_indices = np.random.permutation(len(X_poly))
    X_poly_shuffled = X_poly[shuffled_indices]
    y_shuffled = y[shuffled_indices]
    
    return X_poly_shuffled, y_shuffled

def sgd_by_epoch_process(X_poly_shuffled, y_shuffled, h, alpha, shared_weights):
    """
    Performs one epoch of SGD for each processor.

    Parameters:
    - X_poly_shuffled: (N * h) shuffled matrix of input features.
    - y_shuffled: (N * 1) shuffled target variable.
    - h: Number of basis functions (degree + 1).
    - alpha: Learning rate.
    - n: Number of iterations in one epoch.
    - shared_weights: Current weights.

    Returns:
    - updated_weights: Weights after one epoch.
    """
    updated_weights = shared_weights.copy()

    for i in range(len(y_shuffled)):
        idx = i % len(X_poly_shuffled)
        X_sample = X_poly_shuffled[idx, :]
        y_sample = y_shuffled[idx]
        
        y_hat_sample_poly = X_sample @ updated_weights
        updated_weights = updated_weights - alpha * (X_sample.T * (y_hat_sample_poly - y_sample))

    return updated_weights
