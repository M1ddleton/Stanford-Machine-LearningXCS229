import numpy as np
import util

# Dimension of x
d = 500
# List for lambda to plot
reg_list = [0, 1, 5, 10, 50, 250, 500, 1000]
# List of dataset sizes
n_list = [200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800]


def regression(train_path, validation_path):
    """Part (b): Double descent for unregularized linear regression.
    For a specific training set, obtain beta_hat and return validation error.

    Args:
        train_path: Path to CSV file containing training set.
        validation_path: Path to CSV file containing validation set.

    Return:
        val_err: Validation error
        beta: \hat{\beta}_0 from pdf file
        pred: prediction on validation set
    """
    x_train, y_train = util.load_dataset(train_path)
    x_validation, y_validation = util.load_dataset(validation_path)

    beta = 0
    pred = 0
    val_err = 0
    # *** START CODE HERE ***
    # Compute beta_hat
    beta = np.linalg.pinv(x_train.T @ x_train) @ x_train.T @ y_train

    # Compute predictions on validation set
    pred = x_validation @ beta

    # Compute validation error (mean squared error)
    val_err = (1 / (2 * len(y_validation))) * np.linalg.norm(x_validation @ beta - y_validation) ** 2

    # *** END CODE HERE
    return val_err, beta, pred


def ridge_regression(train_path, validation_path):
    """Part (c): Double descent for regularized linear regression.
    For a specific training set, obtain beta_hat under different l2 regularization strengths
    and return validation error.

    Args:
        train_path: Path to CSV file containing training set.
        validation_path: Path to CSV file containing validation set.

    Return:
        val_err: List of validation errors for different scaling factors of lambda in reg_list.
    """
    x_train, y_train = util.load_dataset(train_path)
    x_validation, y_validation = util.load_dataset(validation_path)

    val_err = []
    # *** START CODE HERE ***

    for reg_strength in reg_list:
        # Compute beta_hat with regularization
        beta = np.linalg.pinv(x_train.T @ x_train + reg_strength * np.eye(d)) @ x_train.T @ y_train

        # Compute predictions on validation set
        pred = x_validation @ beta

        # Compute validation error (mean squared error)
        err = (1 / (2 * len(y_validation))) * np.linalg.norm(x_validation @ beta - y_validation) ** 2
        val_err.append(err)

    # *** END CODE HERE
    return val_err


if __name__ == "__main__":
    val_errs = []
    for n in n_list:
        val_err, _, _ = regression(
            train_path="train%d.csv" % n, validation_path="validation.csv"
        )
        val_errs.append(val_err)
    util.plot(val_errs, "unreg.png", n_list)

    val_errs = []
    for n in n_list:
        val_err = ridge_regression(
            train_path="train%d.csv" % n, validation_path="validation.csv"
        )
        val_errs.append(val_err)
    val_errs = np.asarray(val_errs).T
    util.plot_all(val_errs, "reg.png", n_list)
