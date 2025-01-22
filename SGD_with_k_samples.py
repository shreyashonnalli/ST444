import numpy as np
from multiprocessing import Pool
import time
import pandas as pd
import random

def polynomial_basis_function_transformation(X, h):
    powers = np.arange(h)
    X_poly = np.power(X, powers)
    return X_poly


def compute_gradient(args):
    """
    Compute the gradient of the loss function for a mini-batch of data.
    """
    X_batch, y_batch, alpha = args
    X_poly = polynomial_basis_function_transformation(X_batch, 4)
    m = len(y_batch)  # Mini-batch size
    predictions = X_poly.dot(alpha)  # Linear regression predictions
    errors = predictions - y_batch
    gradient = (1 / m) * X_poly.T.dot(errors)
    return gradient

def parallel_sgd_with_k_samples(X, y, alpha, learning_rate, epochs, num_threads, k):
    """
    Implements parallelized stochastic gradient descent with k samples.
    """
    mse_history = []
    time_taken = []
    n_samples = len(y)
    pool = Pool(processes=num_threads)  # Create a pool of workers
    start_time = time.time()

    for epoch in range(epochs):
        
        
        # Shuffle data
        indices = np.random.choice(len(y), size=(k*num_threads), replace=True)
        random.shuffle(indices)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        X_poly = polynomial_basis_function_transformation(X_shuffled, 4)

        # Split data into mini-batches
        mini_batches = [
            (X_shuffled[i:i+k], y_shuffled[i:i+k], alpha)
            for i in range(0, (k*num_threads), k)
        ]

        # Compute gradients in parallel
        gradients = pool.map(compute_gradient, mini_batches)

        # Average the gradients across all threads
        avg_gradient = np.mean(gradients, axis=0)

        # Update parameters (alpha)
        alpha -= learning_rate * avg_gradient

        # Calculate MSE
        mse = np.mean((X_poly.dot(alpha) - y_shuffled) ** 2) # MSE using the updated parameters
        mse_history.append(mse)

        # Calculate time taken for the epoch
        end_time = time.time()
        elapsed_time = end_time - start_time
        time_taken.append(elapsed_time)



    pool.close()
    pool.join()
    return mse_history, time_taken


def estimation(n, X, y, learning_rate, epochs, num_threads, k):
    df_mse_rows = []
    df_time_rows = []
    for _ in range(n):
        alpha = np.random.rand(4)
        mse, time = parallel_sgd_with_k_samples(X, y, alpha, learning_rate, epochs, num_threads, k)  
        df_mse_rows.append(mse)  
        df_time_rows.append(time)
    df_mse = pd.DataFrame(df_mse_rows)
    df_time = pd.DataFrame(df_time_rows)
    df_mse_avg = df_mse.mean(axis=0)
    df_time_avg = df_time.mean(axis=0)
    return df_mse_avg, df_time_avg
