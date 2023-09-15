import numpy as np


class GDA:
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, step_size=0.01, max_iter=10000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y by updating
        self.theta.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        # Compute phi, mu_0, mu_1, and sigma
        n_examples = x.shape[0]
        dim = x.shape[1]
        if self.theta is None:
            self.theta = np.zeros(dim, dtype=float)
        phi = np.true_divide(len(y[y == 1.0]), n_examples)
        mu_0 = np.true_divide(np.sum(x[y == 0.0], axis=0), len(y[y == 0.0]))
        mu_1 = np.true_divide(np.sum(x[y == 1.0], axis=0), len(y[y == 1.0]))
        sigma = np.true_divide((x[y == 1.0] - mu_1).transpose().dot(x[y == 1.0] - mu_1) \
                               + (x[y == 0.0] - mu_0).transpose().dot((x[y == 0.0] - mu_0)), len(x))
        sigma_inv = np.linalg.inv(sigma)
        theta_0 = 0.5 * (np.dot(mu_0, sigma_inv).dot(mu_0) - np.dot(mu_1, sigma_inv).dot(mu_1)) - np.log(
            (1.0 - phi) / phi)
        theta = -np.dot((mu_0 - mu_1).transpose(), sigma_inv)
        self.theta = np.append(theta_0, theta)
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        predictions = []
        for x_i in x:
            theta_x = np.dot(self.theta.transpose(), x_i)
            sigmoid = 1.0 / (1.0 + np.exp(-theta_x))
            predictions.append(sigmoid)
        return np.array(predictions)
        # *** END CODE HERE
