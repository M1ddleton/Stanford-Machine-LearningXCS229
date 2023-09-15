#!/usr/bin/env python3
import unittest, random, sys, copy, argparse, inspect
import numpy as np
from graderUtil import graded, CourseTestRunner, GradedTestCase

# Import student submission
import submission
import util

#############################################
# HELPER FUNCTIONS FOR CREATING TEST INPUTS #
#############################################

#########
# TESTS #
#########
class Test_2a(GradedTestCase):
  def setUp(self):
    np.random.seed(42)
    _, self.t_test = util.load_dataset('test.csv', label_col='t', add_intercept=True)
  
  @graded()
  def test_0(self):
    """2a-0-basic: Fully Observed Binary Classifier (accuracy check on test set [>96%])"""
    p_test = submission.fully_observed_predictions('train.csv', 'test.csv', 'posonly_true_pred.txt', 'posonly_true_pred.png')
    self.assertTrue(p_test is not None, "full predictions are None")
    yhat = p_test > 0.5
    accuracy = np.mean((yhat == 1) == (self.t_test == 1))
    print('Fully Observed Binary Classifier Accuracy: {}'.format(accuracy * 100))
    self.assertTrue(accuracy * 100 >= 96)

class Test_2b(GradedTestCase):
  def setUp(self):
    np.random.seed(42)
    _, self.t_test = util.load_dataset('test.csv', label_col='t', add_intercept=True)
  
  @graded()
  def test_0(self):
    """2b-0-basic: Naive Method Partial Labels Binary Classifier (accuracy check on test set [>=50%])"""
    p_test, _ = submission.naive_partial_labels_predictions('train.csv', 'test.csv', 'posonly_naive_pred.txt', 'posonly_naive_pred.png')
    self.assertTrue(p_test is not None, "naive predictions are None")
    yhat = p_test > 0.5
    accuracy = np.mean((yhat == 1) == (self.t_test == 1))
    print('Fully Observed Binary Classifier Accuracy: {}'.format(accuracy * 100))
    self.assertTrue(accuracy * 100 >= 50)

class Test_2f(GradedTestCase):
  def setUp(self):
    np.random.seed(42)
    _, self.t_test = util.load_dataset('test.csv', label_col='t', add_intercept=True)
  
  @graded()
  def test_0(self):
    """2f-0-basic: Alpha estimation for binary classifier correction"""
    p_test, clf = submission.naive_partial_labels_predictions('train.csv', 'test.csv', 'posonly_naive_pred.txt', 'posonly_naive_pred.png')
    self.assertTrue(clf is not None, "Logistic Regression Classifier from naive solution is None")
    alpha = submission.find_alpha_and_plot_correction(clf,'valid.csv', 'test.csv', 'posonly_adjusted_pred.txt', 'posonly_adjusted_pred.png', p_test)
    self.assertTrue(alpha is not None, "Correct alpha is None")
    print('Alpha Correction Value: {}'.format(alpha))
    self.assertTrue(alpha > 0.16)

  @graded()
  def test_1(self):
    """2f-1-basic: Alpha Corrected Binary Classification (accuracy check on test set [>93%])"""
    p_test, clf = submission.naive_partial_labels_predictions('train.csv', 'test.csv', 'posonly_naive_pred.txt', 'posonly_naive_pred.png')
    self.assertTrue(clf is not None, "Logistic Regression Classifier from naive solution is None")
    alpha = submission.find_alpha_and_plot_correction(clf,'valid.csv', 'test.csv', 'posonly_adjusted_pred.txt', 'posonly_adjusted_pred.png', p_test)
    self.assertTrue(alpha is not None, "Correct alpha is None")

    yhat = (p_test/alpha) > 0.5
    accuracy = np.mean((yhat == 1) == (self.t_test == 1))
    print('Fully Observed Binary Classifier Accuracy: {}'.format(accuracy * 100))
    self.assertTrue(accuracy * 100 >= 93)

def getTestCaseForTestID(test_id):
  question, part, _ = test_id.split('-')
  g = globals().copy()
  for name, obj in g.items():
    if inspect.isclass(obj) and name == ('Test_'+question):
      return obj('test_'+part)

if __name__ == '__main__':
  # Parse for a specific test
  parser = argparse.ArgumentParser()
  parser.add_argument('test_case', nargs='?', default='all')
  test_id = parser.parse_args().test_case

  assignment = unittest.TestSuite()
  if test_id != 'all':
    assignment.addTest(getTestCaseForTestID(test_id))
  else:
    assignment.addTests(unittest.defaultTestLoader.discover('.', pattern='grader.py'))
  CourseTestRunner().run(assignment)