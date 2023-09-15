import numpy as np
import util
import importlib.util

use_submission = importlib.util.find_spec('submission') is not None
if use_submission:
  from submission import GDA, LogisticRegression


# Character to replace with sub-problem letter in plot_path/save_path
WILDCARD = 'X'

def main_LogReg(train_path, valid_path, save_path):
    """Problem (1b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # Train a logistic regression classifier
    clf = LogisticRegression()
    clf.fit(x_train, y_train)

    # Plot decision boundary on top of validation set set
    x_eval, y_eval = util.load_dataset(valid_path, add_intercept=True)
    plot_path = save_path.replace('.txt', '.png')
    util.plot(x_eval, y_eval, clf.theta, plot_path)

    # Use np.savetxt to save predictions on eval set to save_path
    p_eval = clf.predict(x_eval)
    yhat = p_eval > 0.5
    print('LR Accuracy: %.2f' % np.mean( (yhat == 1) == (y_eval == 1)))
    np.savetxt(save_path, p_eval)

def main_GDA(train_path, valid_path, save_path):
    """Problem (1e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # Train a GDA classifier
    clf = GDA()
    clf.fit(x_train, y_train)

    # Plot decision boundary on validation set
    x_eval, y_eval = util.load_dataset(valid_path, add_intercept=False)
    plot_path = save_path.replace('.txt', '.png')
    util.plot(x_eval, y_eval, clf.theta, plot_path)
    x_eval = util.add_intercept(x_eval)

    # Use np.savetxt to save outputs from validation set to save_path
    p_eval = clf.predict(x_eval)
    yhat = p_eval > 0.5
    print('GDA Accuracy: %.2f' % np.mean( (yhat == 1) == (y_eval == 1)))
    np.savetxt(save_path, p_eval)

if __name__ == '__main__':
    '''
    Start of Problem 1: Linear Classifiers
    '''
    # 1b
    main_LogReg(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='logreg_pred_1.txt')
    main_LogReg(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='logreg_pred_2.txt')
    # 1e
    main_GDA(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='gda_pred_1.txt')
    main_GDA(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='gda_pred_2.txt')