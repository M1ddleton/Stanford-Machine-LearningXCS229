import numpy as np
import ast
import matplotlib.pyplot as plt

# Scaling for lambda to plot
reg_list = [0, 1, 5, 10, 50, 250, 500, 1000]


def from_np_array(array_string):
    array_string = ",".join(array_string.replace("[ ", "[").split())
    return np.array(ast.literal_eval(array_string))


def add_intercept(x):
    """Add intercept to matrix x.

    Args:
        x: 2D NumPy array.

    Returns:
        New matrix same as x with 1's in the 0th column.
    """
    new_x = np.zeros((x.shape[0], x.shape[1] + 1), dtype=x.dtype)
    new_x[:, 0] = 1
    new_x[:, 1:] = x

    return new_x


def load_dataset(csv_path, label_col="y", add_intercept=False):
    """Load dataset from a CSV file.

    Args:
         csv_path: Path to CSV file containing dataset.
         label_col: Name of column to use as labels (should be 'y' or 't').
         add_intercept: Add an intercept entry to x-values.

    Returns:
        xs: Numpy array of x-values (inputs).
        ys: Numpy array of y-values (labels).
    """

    def add_intercept_fn(x):
        global add_intercept
        return add_intercept(x)

    # Validate label_col argument
    allowed_label_cols = ("y", "t")
    if label_col not in allowed_label_cols:
        raise ValueError(
            "Invalid label_col: {} (expected {})".format(label_col, allowed_label_cols)
        )

    # Load headers
    csv_fh = open(csv_path, "r")
    headers = csv_fh.readline().strip().split(",")

    # Load features and labels
    x_cols = [i for i in range(len(headers)) if headers[i].startswith("x")]
    l_cols = [i for i in range(len(headers)) if headers[i] == label_col]

    lines = csv_fh.read()
    x_raw = lines.split('"')[1::2]
    y_raw = lines.split('"')[2::2]
    inputs = []
    labels = []

    for i in range(len(x_raw)):
        inputs.append(
            [float(x) for x in x_raw[i].replace("]", "").replace("[", "").split()]
        )
        labels.append(float(y_raw[i][1:-1]))

    inputs = np.asarray(inputs)
    labels = np.asarray(labels)

    if inputs.ndim == 1:
        inputs = np.expand_dims(inputs, -1)

    if add_intercept:
        inputs = add_intercept_fn(inputs)

    return inputs, labels


def plot(val_err, save_path, n_list):
    """Plot dataset size vs. val err (a single curve)

    Args:
        val_err: list of validation erro
        save_path: Path to save the plot.
        n_list: List of trainset sizes.
    """
    # Plot dataset
    plt.figure()
    plt.plot(n_list, val_err, linewidth=2, label="lambda=0")

    # Add labels and save to disk
    plt.xlabel("Num Samples")
    plt.ylabel("Validation Err")
    plt.ylim(0, 2)
    plt.legend()
    plt.savefig(save_path)


def plot_all(val_err, save_path, n_list):
    """Plot dataset size vs. val err for different reg strengths

    Args:
        val_err: Matrix of validation erros, row.
        save_path: Path to save the plot.
        n_list: List of trainset sizes.
    """
    # Plot dataset
    plt.figure()
    for i in range(len(reg_list)):
        plt.plot(n_list, val_err[i], linewidth=2, label="lambda=%0.0f" % reg_list[i])

    # Add labels and save to disk
    plt.xlabel("Num Samples")
    plt.ylabel("Validation Err")
    plt.ylim(0, 2)
    plt.legend()
    plt.savefig(save_path)
