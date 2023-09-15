#!/usr/bin/env python3
import unittest, random, sys, copy, argparse, inspect
import numpy as np
import matplotlib.image as mpimg
import os
from graderUtil import graded, CourseTestRunner, GradedTestCase

# Import student submission
import submission

#############################################
# HELPER FUNCTIONS FOR CREATING TEST INPUTS #
#############################################

#########
# TESTS #
#########
class Test_3c(GradedTestCase):
  def setUp(self):
    np.random.seed(42)
    self.X = submission.normalize(submission.load_data())
    self.test_tensor_top_row = np.array([ 0.0092272,  -0.01189988, -0.00997819, -0.00536916, 0.01186695])
    self.test_tensor_middle = np.array([-8.56913366, -20.58484356,  4.36190999, 18.6723115,  -7.21102089])
    self.learning_rate = 0.05

  @graded()
  def test_0(self):
    """3c-0-basic: update_W returns correct tensor"""
    M, N = self.X.shape
    W = np.eye(N)
    rand = np.random.permutation(range(M))
    for i in rand:
        x = self.X[i]
        W = submission.update_W(W, x, self.learning_rate)
  
    self.assertTrue(W is not None, "Updated W cannot be Done")
    self.assertTrue(W[3].shape == self.test_tensor_middle.shape, "shapes of output tensor not the same")
  @graded()
  def test_1(self):
    """3c-1-basic: unmix returns correct tensor"""
    M, N = self.X.shape
    W = np.eye(N)
    S = submission.normalize(submission.unmix(self.X, W))
    self.assertTrue(S[0].shape == self.test_tensor_top_row.shape, "shapes of output tensor not the same")
    self.assertTrue(np.allclose(S[0],self.test_tensor_top_row), "unmix does not return correct tensor")

  @graded(is_hidden=True)
  def test_2(self):
    """3c-2-hidden: Checking unmixed matrix obtrained with Laplace"""
    if os.path.exists('./W_sol.txt'):
      W_solution = np.loadtxt('./W_sol.txt')
      W_student = np.loadtxt('./W.txt')
      self.assertTrue(W_solution.shape == W_student.shape, "shapes of split tensor are not the same")
      self.assertTrue(np.allclose(W_solution, W_student, atol=1e-2), "resulting unmixed 5x5 matrix do not match solution")


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