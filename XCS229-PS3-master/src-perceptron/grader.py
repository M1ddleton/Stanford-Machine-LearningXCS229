#!/usr/bin/env python3
from mimetypes import init
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

class Test_3b(GradedTestCase):
  def setUp(self):
    np.random.seed(42)
    self.train_x, self.train_y = util.load_csv('train.csv')
    self.lr = 0.5
    self.test_state = [(np.array([-0.5]), np.array([3.1, 0.67])), (np.array([0.0]), np.array([ 4.5, -3.4])), (np.array([0.0]), np.array([1.5, -2.0])), (np.array([-0.67]), np.array([-3.2, -0.5])), (np.array([0.0]), np.array([1.5, -3.2]))]

  @graded()
  def test_0(self):
    """3b-0-basic: Initial state of perceptron"""
    init_state = submission.initial_state()
    self.assertTrue(init_state == [], "Initial state of perceptron is incorrect")

  @graded()
  def test_1(self):
    """3b-1-basic: Check prediction function for dot kernel"""
    init_state = submission.initial_state()
    self.assertTrue(init_state == [], "Initial state of perceptron is incorrect")
  
    for x_i, y_i in zip(self.train_x, self.train_y):
      next_prediction = submission.predict(self.test_state, submission.dot_kernel, x_i)
      self.assertTrue(next_prediction == 1, f"next prediction using predict function wrong: {next_prediction}")
      break
  
  @graded()
  def test_2(self):
    """3b-2-basic: Check prediction function for rbf kernel"""
    init_state = submission.initial_state()
    self.assertTrue(init_state == [], "Initial state of perceptron is incorrect")
  
    for x_i, y_i in zip(self.train_x, self.train_y):
      next_prediction = submission.predict(self.test_state, submission.rbf_kernel, x_i)
      self.assertTrue(next_prediction == 0, f"next prediction using predict function wrong: {next_prediction}")
      break
  
  @graded(is_hidden=True)
  def test_3(self):
    """3b-3-hidden: Check update_state function for dot kernel"""
    solution_update = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol.update_state)
    solution_initial_state = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol.initial_state)

    student_init_state = submission.initial_state()
    solution_init_state = solution_initial_state()

    self.assertTrue(student_init_state == [], "Initial state of perceptron is incorrect")

    for x_i, y_i in zip(self.train_x, self.train_y):
      submission.update_state(student_init_state, submission.dot_kernel, self.lr, x_i, y_i)
      solution_update(solution_init_state, submission.dot_kernel, self.lr, x_i, y_i)

    self.assertTrue(student_init_state == solution_init_state, "update_state did not lead to same updated state after all training samples.")

  @graded(is_hidden=True)
  def test_4(self):
    """3b-4-hidden: Check update_state function for rbf kernel"""
    solution_update = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol.update_state)
    solution_initial_state = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol.initial_state)

    student_init_state = submission.initial_state()
    solution_init_state = solution_initial_state()

    self.assertTrue(student_init_state == [], "Initial state of perceptron is incorrect")
    
    for x_i, y_i in zip(self.train_x, self.train_y):
      submission.update_state(student_init_state, submission.rbf_kernel, self.lr, x_i, y_i)
      solution_update(solution_init_state, submission.rbf_kernel, self.lr, x_i, y_i)

    self.assertTrue(student_init_state == solution_init_state, "update_state did not lead to same updated state after all training samples.")

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