#!/usr/bin/env python3
import unittest, random, sys, copy, argparse, inspect
import numpy as np
from graderUtil import graded, CourseTestRunner, GradedTestCase
from sklearn.metrics import accuracy_score
import os
from scipy.linalg import null_space

import util

# Import student submission
import submission

#############################################
# HELPER FUNCTIONS FOR CREATING TEST INPUTS #
#############################################


#########
# TESTS #
#########
class Test_4c(GradedTestCase):
  def setUp(self):
    np.random.seed(42)

    train_path = 'ir1_train.csv'
    valid_path = 'ir1_valid.csv'
    self.X, self.Y = util.load_dataset(train_path)
    self.X_val, self.Y_val = util.load_dataset(valid_path)

  @graded(timeout=1)
  def test_0(self):
    """4c-0-basic:  Check the shape of the minimum norm solution"""

    beta_0 = submission.get_minimum_norm_solution(self.X_val, self.Y_val)

    self.assertTrue(beta_0.shape == (self.X_val.shape[-1],))

  @graded(timeout=1, is_hidden=True)
  def test_1(self):
    """4c-1-hidden:  Check the result of the minimum norm solution"""

    beta_0 = submission.get_minimum_norm_solution(self.X_val, self.Y_val)

    # *** BEGIN_HIDE ***
    # *** END_HIDE ***
    
  @graded(timeout=1)
  def test_2(self):
    """4c-2-basic:  Check the number of returned solutions and that they are different"""

    beta_0 = submission.get_minimum_norm_solution(self.X_val, self.Y_val)

    ns = null_space(self.X_val).T

    n = 3

    solutions = submission.get_different_n_solutions(beta_0, ns, n)

    self.assertTrue(len(solutions) == n)

    for i in range(n):
      self.assertFalse(np.array_equal(beta_0, solutions[i]))

  @graded(timeout=1)
  def test_3(self):
    """4c-3-hidden:  Check the returned solutions are valid"""

    beta_0 = submission.get_minimum_norm_solution(self.X_val, self.Y_val)

    ns = null_space(self.X_val).T

    n = 3

    solutions = submission.get_different_n_solutions(beta_0, ns, n)

    self.assertTrue(len(solutions) == n)

    # *** BEGIN_HIDE ***
    # *** END_HIDE ***

class Test_4f(GradedTestCase):
  def setUp(self):
    np.random.seed(42)

    train_path = 'ir2_train.csv'
    valid_path = 'ir2_valid.csv'
    self.X, self.Y = util.load_dataset(train_path)
    self.X_val, self.Y_val = util.load_dataset(valid_path)
    self.d = self.X.shape[1]

    self.submission_QP = submission.QP
    self.solution_QP = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol.QP)

  @graded(timeout=1)
  def test_0(self):
    """4f-0-basic:  Check the shape of the QP.gradient output"""

    alpha = 0.1

    model = self.submission_QP(self.d, np.ones(self.d) * alpha)

    grad = model.gradient(self.X_val, self.Y_val)

    self.assertTrue(len(grad) == 2)

    self.assertTrue(grad[0].shape == self.X_val.shape and grad[1].shape == self.X_val.shape) 


  @graded(timeout=1, is_hidden=True)
  def test_1(self):
    """4f-1-hidden:  Check the QP.gradient output matches"""

    alpha = 0.1

    student_model = self.submission_QP(self.d, np.ones(self.d) * alpha)
    sol_model = self.solution_QP(self.d, np.ones(self.d) * alpha)

    student_grad = student_model.gradient(self.X_val, self.Y_val)
    sol_grad = sol_model.gradient(self.X_val, self.Y_val)

    self.assertTrue(len(student_grad) == len(sol_grad))

    for i in range(len(sol_grad)):
      assert(np.allclose(student_grad[i], sol_grad[i]))

  @graded(timeout=1)
  def test_2(self):
    """4f-2-basic:  Check that internal variables got updated after one iteration of QP.train_GD"""

    alpha = 0.1

    model = self.submission_QP(self.d, np.ones(self.d) * alpha)

    prev_theta, prev_phi = model.theta, model.phi

    model.train_GD(self.X, self.Y, max_step=1, X_val=self.X_val, Y_val=self.Y_val)

    self.assertFalse(np.array_equal(prev_theta, model.theta))
    self.assertFalse(np.array_equal(prev_phi, model.phi))


  @graded(timeout=1, is_hidden=True)
  def test_3(self):
    """4f-3-hidden:  Check that internal variables match after one iteration of QP.train_GD"""

    alpha = 0.1

    student_model = self.submission_QP(self.d, np.ones(self.d) * alpha)
    sol_model = self.solution_QP(self.d, np.ones(self.d) * alpha)

    student_model.train_GD(self.X, self.Y, max_step=1, X_val=self.X_val, Y_val=self.Y_val)
    sol_model.train_GD(self.X, self.Y, max_step=1, X_val=self.X_val, Y_val=self.Y_val)

    assert(np.allclose(student_model.theta, sol_model.theta))
    assert(np.allclose(student_model.phi, sol_model.phi))

    
class Test_4h(GradedTestCase):
  def setUp(self):
    np.random.seed(42)

    train_path = 'ir2_train.csv'
    valid_path = 'ir2_valid.csv'
    self.X, self.Y = util.load_dataset(train_path)
    self.X_val, self.Y_val = util.load_dataset(valid_path)
    self.d = self.X.shape[1]

    self.submission_QP = submission.QP
    self.solution_QP = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol.QP)


  @graded(timeout=1)
  def test_0(self):
    """4h-0-basic:  Check that internal variables got updated after one iteration of QP.train_SGD"""

    alpha = 0.1

    model = self.submission_QP(self.d, np.ones(self.d) * alpha)

    prev_theta, prev_phi = model.theta, model.phi

    model.train_SGD(self.X, self.Y, max_step=1, X_val=self.X_val, Y_val=self.Y_val)

    self.assertFalse(np.array_equal(prev_theta, model.theta))
    self.assertFalse(np.array_equal(prev_phi, model.phi))


  @graded(timeout=1, is_hidden=True)
  def test_1(self):
    """4h-1-hidden:  Check that internal variables match after one iteration of QP.train_SGD"""

    alpha = 0.1

    student_model = self.submission_QP(self.d, np.ones(self.d) * alpha)
    sol_model = self.solution_QP(self.d, np.ones(self.d) * alpha)

    student_model.train_SGD(self.X, self.Y, max_step=1, X_val=self.X_val, Y_val=self.Y_val)
    sol_model.train_SGD(self.X, self.Y, max_step=1, X_val=self.X_val, Y_val=self.Y_val)

    assert(np.allclose(student_model.theta, sol_model.theta))
    assert(np.allclose(student_model.phi, sol_model.phi))



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