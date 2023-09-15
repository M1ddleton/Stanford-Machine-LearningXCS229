import matplotlib.pyplot as plt
import numpy as np

def load_dataset(csv_path, label_col='y'):
    """Load dataset from a CSV file.

    Args:
         csv_path: Path to CSV file containing dataset.
         label_col: Name of column to use as labels (should be 'y' or 't').

    Returns:
        xs: Numpy array of x-values (inputs).
        ys: Numpy array of y-values (labels).
    """

    # Validate label_col argument
    allowed_label_cols = ('y', 't')
    if label_col not in allowed_label_cols:
        raise ValueError('Invalid label_col: {} (expected {})'
                         .format(label_col, allowed_label_cols))

    # Load headers
    with open(csv_path, 'r') as csv_fh:
        headers = csv_fh.readline().strip().split(',')

    # Load features and labels
    x_cols = [i for i in range(len(headers)) if headers[i].startswith('x')]
    l_cols = [i for i in range(len(headers)) if headers[i] == label_col]
    inputs = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=x_cols)
    labels = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=l_cols)

    if inputs.ndim == 1:
        inputs = np.expand_dims(inputs, -1)

    return inputs, labels


def generate_data_linear(n, d):
    np.random.seed(0)
    beta_star = np.random.normal(size=d) / np.sqrt(d)
    
    X = np.random.normal(size=(n,d))
    Y = X.dot(beta_star)
    
    X_val = np.random.normal(size=(n,d))
    Y_val = X_val.dot(beta_star)
    return X, Y, X_val, Y_val

def generate_data_QP(n, d):
    np.random.seed(0)
    beta_star = np.ones(5)
    beta_star.resize(d)
    np.random.shuffle(beta_star)
    
    X = np.random.normal(size=(n,d))
    Y = X.dot(beta_star)
    
    X_val = np.random.normal(size=(n,d))
    Y_val = X_val.dot(beta_star)
    return X, Y, X_val, Y_val

def plot_points(x, y, save_path):
    """Plot the validation error vs. norm of the solution
    part (c) of Implicit Regularization

    Args:
        x: list of norms
        y: list of validation errors
        save_path: path to save the plot
    """
    plt.figure()
    plt.scatter(x, y)
    plt.xlabel('norm')
    plt.ylabel('validation error')
    plt.savefig(save_path)

def plot_training_and_validation_curves(logs, save_path, label):
    """Plot multiple training/validation curves

    For better visualization, we add the following trick:
        1. we only plot one point every 10 steps

    Args:
        logs: list of (steps, training error, validation error) tuple
        save_path: path to save the plot
        label: list of labels
    """
    plt.figure()
    for i in range(len(logs)):
        log = logs[i]
        plt.plot(log[0][::10], log[2][::10], label = "validation error, " + label[i])
    for i in range(len(logs)):
        log = logs[i]
        plt.plot(log[0][::10], log[1][::10], '--', label = "training error, " + label[i])
    plt.ylim([0, 0.5])
    plt.xlabel('steps')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(save_path, bbox_inches='tight')
