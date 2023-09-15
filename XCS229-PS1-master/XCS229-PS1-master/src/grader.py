#!/usr/bin/env python3
import unittest, random, sys, copy, argparse, inspect
from graderUtil import graded, CourseTestRunner, GradedTestCase
import numpy as np

# Import student submission
import submission
import util

#############################################
# HELPER FUNCTIONS FOR CREATING TEST INPUTS #
#############################################

#########
# TESTS #
#########

class Test_2b(GradedTestCase):
  def setUp(self):
    np.random.seed(42)
    self.train_x, self.train_y = util.load_dataset('train.csv',
                                                   add_intercept=False)

  def compare_poly_model_fit(self, k):
    solution_model = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol.LinearModel())
    train_phi_from_solution = solution_model.create_poly(k=k, x=self.train_x)
    solution_model.fit(train_phi_from_solution, self.train_y)

    model = submission.LinearModel()
    train_phi = model.create_poly(k=k, x=self.train_x)
    model.fit(train_phi, self.train_y)

    self.assertTrue(np.allclose(np.ravel(model.theta),
                                np.ravel(solution_model.theta),
                                rtol=0.2))

  @graded(is_hidden=True)
  def test_0(self):
    """2b-0-hidden:  Compare polynomial (degree=3) regression phi."""
    solution_model = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol.LinearModel)
    self.assertTrue(np.allclose(solution_model.create_poly(k=3, x=self.train_x),
                                submission.LinearModel.create_poly(k=3, x=self.train_x)))

  @graded(is_hidden=True)
  def test_1(self):
    """2b-1-hidden:  Compare polynomial (degree=3) regression fit (computed theta)."""
    self.compare_poly_model_fit(k=3)

  @graded(is_hidden=True)
  def test_2(self):
    """2b-2-hidden:  Compare polynomial regression phi."""
    solution_model = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol.LinearModel)
    self.assertTrue(np.allclose(solution_model.create_poly(k=1, x=self.train_x),
                                submission.LinearModel.create_poly(k=1, x=self.train_x)))
    self.assertTrue(np.allclose(solution_model.create_poly(k=2, x=self.train_x),
                                submission.LinearModel.create_poly(k=2, x=self.train_x)))
    self.assertTrue(np.allclose(solution_model.create_poly(k=3, x=self.train_x),
                                submission.LinearModel.create_poly(k=3, x=self.train_x)))
    self.assertTrue(np.allclose(solution_model.create_poly(k=5, x=self.train_x),
                                submission.LinearModel.create_poly(k=5, x=self.train_x)))
    self.assertTrue(np.allclose(solution_model.create_poly(k=10, x=self.train_x),
                                submission.LinearModel.create_poly(k=10, x=self.train_x)))
    self.assertTrue(np.allclose(solution_model.create_poly(k=20, x=self.train_x),
                                submission.LinearModel.create_poly(k=20, x=self.train_x)))

  @graded(is_hidden=True)
  def test_3(self):
    """2b-3-hidden:  Compare polynomial (degree=1) regression fit (computed theta)."""
    self.compare_poly_model_fit(k=1)

  @graded(is_hidden=True)
  def test_4(self):
    """2b-4-hidden:  Compare polynomial (degree=2) regression fit (computed theta)."""
    self.compare_poly_model_fit(k=2)

  @graded(is_hidden=True)
  def test_5(self):
    """2b-5-hidden:  Compare polynomial (degree=3) regression fit (computed theta)."""
    self.compare_poly_model_fit(k=3)

  @graded(is_hidden=True)
  def test_6(self):
    """2b-6-hidden:  Compare polynomial (degree=5) regression fit (computed theta)."""
    self.compare_poly_model_fit(k=5)

  @graded(is_hidden=True)
  def test_7(self):
    """2b-7-hidden:  Compare polynomial (degree=10) regression fit (computed theta)."""
    self.compare_poly_model_fit(k=10)

  @graded(is_hidden=True)
  def test_8(self):
    """2b-8-hidden:  Compare polynomial (degree=20) regression fit (computed theta)."""
    self.compare_poly_model_fit(k=20)

  @graded()
  def test_9(self):
    """2b-9-basic:  Create the plot for visually verifying the student's implementation"""
    submission.run_exp('train.csv', False, [1, 2, 3, 5, 10, 20], 'large-poly.png')
    self.assertTrue(True)

class Test_2d(GradedTestCase):
  def setUp(self):
    np.random.seed(42)
    self.train_x, self.train_y = util.load_dataset('train.csv',
                                                   add_intercept=False)
  def compare_sine_model_fit(self, k):
    solution_model = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol.LinearModel())
    train_phi_from_solution = solution_model.create_sin(k=k, x=self.train_x)
    solution_model.fit(train_phi_from_solution, self.train_y)

    model = submission.LinearModel()
    train_phi = model.create_sin(k=k, x=self.train_x)
    model.fit(train_phi, self.train_y)

    self.assertTrue(np.allclose(np.ravel(model.theta),
                                np.ravel(solution_model.theta),
                                rtol=0.2))

  @graded(is_hidden=True)
  def test_0(self):
    """2d-0-hidden:  Compare polynomial regression phi."""
    solution_model = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol.LinearModel)
    self.assertTrue(np.allclose(solution_model.create_sin(k=1, x=self.train_x),
                                submission.LinearModel.create_sin(k=1, x=self.train_x)))
    self.assertTrue(np.allclose(solution_model.create_sin(k=2, x=self.train_x),
                                submission.LinearModel.create_sin(k=2, x=self.train_x)))
    self.assertTrue(np.allclose(solution_model.create_sin(k=3, x=self.train_x),
                                submission.LinearModel.create_sin(k=3, x=self.train_x)))
    self.assertTrue(np.allclose(solution_model.create_sin(k=5, x=self.train_x),
                                submission.LinearModel.create_sin(k=5, x=self.train_x)))
    self.assertTrue(np.allclose(solution_model.create_sin(k=10, x=self.train_x),
                                submission.LinearModel.create_sin(k=10, x=self.train_x)))
    self.assertTrue(np.allclose(solution_model.create_sin(k=20, x=self.train_x),
                                submission.LinearModel.create_sin(k=20, x=self.train_x)))

  @graded(is_hidden=True)
  def test_1(self):
    """2d-1-hidden:  Compare polynomial (degree=1) regression fit (computed theta)."""
    self.compare_sine_model_fit(k=1)

  @graded(is_hidden=True)
  def test_2(self):
    """2d-2-hidden:  Compare polynomial (degree=2) regression fit (computed theta)."""
    self.compare_sine_model_fit(k=2)

  @graded(is_hidden=True)
  def test_3(self):
    """2d-3-hidden:  Compare polynomial (degree=3) regression fit (computed theta)."""
    self.compare_sine_model_fit(k=3)

  @graded(is_hidden=True)
  def test_4(self):
    """2d-4-hidden:  Compare polynomial (degree=5) regression fit (computed theta)."""
    self.compare_sine_model_fit(k=5)

  @graded(is_hidden=True)
  def test_5(self):
    """2d-5-hidden:  Compare polynomial (degree=10) regression fit (computed theta)."""
    self.compare_sine_model_fit(k=10)

  @graded(is_hidden=True)
  def test_6(self):
    """2d-6-hidden:  Compare polynomial (degree=20) regression fit (computed theta)."""
    self.compare_sine_model_fit(k=20)

  @graded()
  def test_7(self):
    """2d-7-basic:  Create the plot for visually verifying the student's implementation"""
    submission.run_exp('train.csv', True, [1, 2, 3, 5, 10, 20], 'large-sine.png')
    self.assertTrue(True)

class Test_2f(GradedTestCase):
  def setUp(self):
    np.random.seed(42)
    self.train_x, self.train_y = util.load_dataset('train.csv',
                                                   add_intercept=False)

  @graded()
  def test_0(self):
    """2f-0-basic:  Create the plot for visually verifying the student's implementation"""
    submission.run_exp('small.csv', True, [1, 2, 3, 5, 10, 20], 'small-sine.png')
    submission.run_exp('small.csv', False, [1, 2, 3, 5, 10, 20], 'small-poly.png')
    self.assertTrue(True)

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