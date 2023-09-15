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
class Test_3b(GradedTestCase):
  def setUp(self):
    np.random.seed(42)

    self.x_train, self.y_train = util.load_dataset('train.csv')
    self.x_val, self.y_val = util.load_dataset('validation.csv')

  @graded()
  def test_0(self):
    """3b-0-basic: naive logistic regression (verify correct p_val shape)"""
    student_p_val = submission.apply_logisitic_regression(self.x_train, self.y_train, self.x_val, self.y_val, 'naive')
    self.assertTrue(student_p_val.shape == (self.x_val.shape[0],))
  
  @graded(is_hidden=True)
  def test_1(self):
    """3b-1-hidden: naive logistic regression (verify correct p_val)"""
    solution_p_val = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol.apply_logisitic_regression(self.x_train, self.y_train, self.x_val, self.y_val, 'naive'))
    student_p_val = submission.apply_logisitic_regression(self.x_train, self.y_train, self.x_val, self.y_val, 'naive')
    is_close = np.allclose(solution_p_val, student_p_val, rtol=0.25, atol=0)
    self.assertTrue(is_close)

  @graded(is_hidden=True)
  def test_2(self):
    """3b-2-hidden: naive logistic regression correct accuracies"""
    solution_p_val = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol.apply_logisitic_regression(self.x_train, self.y_train, self.x_val, self.y_val, 'naive'))
    solution_accuracies = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol.calculate_accuracies(solution_p_val, self.y_val))
    student_p_val = submission.apply_logisitic_regression(self.x_train, self.y_train, self.x_val, self.y_val, 'naive')
    student_accuracies = submission.calculate_accuracies(student_p_val, self.y_val)
    print("Comparing Accuracy of Positive Examples")
    self.assertTrue(solution_accuracies[0] == student_accuracies[0])
    print("Comparing Accuracy of Negative Examples")
    self.assertTrue(solution_accuracies[1] == student_accuracies[1])
    print("Comparing Balanced Accuracy")
    self.assertTrue(solution_accuracies[2] == student_accuracies[2])
    print("Comparing Total Accuracy")
    self.assertTrue(solution_accuracies[3] == student_accuracies[3])

class Test_3d(GradedTestCase):
  def setUp(self):
    np.random.seed(42)

    self.x_train, self.y_train = util.load_dataset('train.csv')
    self.x_val, self.y_val = util.load_dataset('validation.csv')
  
  @graded()
  def test_0(self):
    """3d-0-basic: upsampling minority returns correct x_train, y_train shapes"""
    student_x_train_upsampled, student_y_train_upsampled = submission.upsample_minority_class(self.x_train, self.y_train)
    self.assertTrue(student_x_train_upsampled.shape == (2500,3))
    self.assertTrue(student_y_train_upsampled.shape == (2500,))
  
  @graded(timeout=30, is_hidden=True)
  def test_1(self):
    """3d-1-hidden: upsampled logistic regression (verify correct p_val)"""
    solution_x_train_upsampled, solution_y_train_upsampled = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol.upsample_minority_class(self.x_train, self.y_train))
    solution_p_val = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol.apply_logisitic_regression(solution_x_train_upsampled, solution_y_train_upsampled, self.x_val, self.y_val, 'upsampling'))
    student_x_train_upsampled, student_y_train_upsampled = submission.upsample_minority_class(self.x_train, self.y_train)
    student_p_val = submission.apply_logisitic_regression(student_x_train_upsampled, student_y_train_upsampled, self.x_val, self.y_val, 'upsampling')
    is_close = np.allclose(solution_p_val, student_p_val, rtol=0.25, atol=0)
    self.assertTrue(is_close)

  @graded(timeout=40, is_hidden=True)
  def test_2(self):
    """3d-2-hidden: upsampled logistic regression correct accuracies"""
    solution_x_train_upsampled, solution_y_train_upsampled = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol.upsample_minority_class(self.x_train, self.y_train))
    solution_p_val = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol.apply_logisitic_regression(solution_x_train_upsampled, solution_y_train_upsampled, self.x_val, self.y_val, 'upsampling'))
    solution_accuracies = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol.calculate_accuracies(solution_p_val, self.y_val))
    student_x_train_upsampled, student_y_train_upsampled = submission.upsample_minority_class(self.x_train, self.y_train)
    student_p_val = submission.apply_logisitic_regression(student_x_train_upsampled, student_y_train_upsampled, self.x_val, self.y_val, 'upsampling')
    student_accuracies = submission.calculate_accuracies(student_p_val, self.y_val)
    print("Comparing Accuracy of Positive Examples")
    self.assertTrue(solution_accuracies[0] == student_accuracies[0])
    print("Comparing Accuracy of Negative Examples")
    self.assertTrue(solution_accuracies[1] == student_accuracies[1])
    print("Comparing Balanced Accuracy")
    self.assertTrue(solution_accuracies[2] == student_accuracies[2])
    print("Comparing Total Accuracy")
    self.assertTrue(solution_accuracies[3] == student_accuracies[3])
    
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