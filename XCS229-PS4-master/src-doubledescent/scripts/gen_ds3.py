import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

sigma = 0.5  # noise ~ N(0, sigma^2)
d = 500 # dimension of x
eta = 1/np.sqrt(d) # theta ~ N(0, eta^2*I)
n_list = [200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800]

DATA_DIR = '../'
np.random.seed(22)

def generate_gaussian(theta, n_examples):
    """Generate dataset for l2 linear regression."""

    x = np.random.normal(0, 1, [n_examples, d])
    y = np.matmul(x, theta) + np.random.normal(0, sigma, [n_examples, 1])

    return x, y

if __name__ == '__main__':
    theta = np.random.normal(0, eta, [d, 1])
    """
    x, y = generate_gaussian(theta, n_list[-1])
    for n in n_list:
        gaussian_df = pd.DataFrame({'x': [i for i in x[:n]], 'y': [i[0] for i in y[:n]]})
        gaussian_df.to_csv(os.path.join(DATA_DIR, f'train{n}.csv'), index=False)
    """
    x, y = generate_gaussian(theta, 2000)
    gaussian_df = pd.DataFrame({'x': [i for i in x], 'y': [i[0] for i in y]})
    gaussian_df.to_csv(os.path.join(DATA_DIR, f'validation.csv'), index=False)
