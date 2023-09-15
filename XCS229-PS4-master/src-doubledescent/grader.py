#!/usr/bin/env python3
import unittest, random, sys, copy, argparse, inspect, collections
import numpy as np
from graderUtil import graded, CourseTestRunner, GradedTestCase

# Import student submission
import submission

#############################################
# HELPER FUNCTIONS FOR CREATING TEST INPUTS #
#############################################
N_ROWS = 5000

#########
# TESTS #
#########


class Test_4b(GradedTestCase):
    def setUp(self):
        np.random.seed(42)

    @graded(is_hidden=False)
    def test_0(self):
        """4b-0-basic: check dimension (`regression`)"""

        for n in submission.n_list[:1]:
            val_err, beta, pred = submission.regression(
                train_path="train%d.csv" % n, validation_path="validation.csv"
            )
            self.assertTrue(beta.shape == (500,))
            self.assertTrue(pred.shape == (2000,))

    @graded(is_hidden=True, timeout=60)
    def test_1(self):
        """4b-1-hidden: check return values (`regression`)"""
        for n in submission.n_list:
            student_val_err, student_beta, student_pred = submission.regression(
                train_path="train%d.csv" % n, validation_path="validation.csv"
            )

            (
                solution_val_err,
                solution_beta,
                solution_pred,
            ) = self.run_with_solution_if_possible(
                submission,
                lambda sub_or_sol: sub_or_sol.regression(
                    train_path="train%d.csv" % n, validation_path="validation.csv"
                ),
            )

            self.assertTrue(
                np.allclose(student_val_err, solution_val_err, atol=0.1, rtol=0.1)
            )
            self.assertTrue(
                np.allclose(student_beta, solution_beta, atol=0.1, rtol=0.1)
            )
            self.assertTrue(
                np.allclose(student_pred, solution_pred, atol=0.1, rtol=0.1)
            )


class Test_4c(GradedTestCase):
    def setUp(self):
        np.random.seed(42)

    @graded(is_hidden=True, timeout=60)
    def test_0(self):
        """4c-0-hidden: val_error check (`ridge_regression`)"""

        for n in submission.n_list[:3]:
            student_val_err = submission.ridge_regression(
                train_path="train%d.csv" % n, validation_path="validation.csv"
            )

            solution_val_err = self.run_with_solution_if_possible(
                submission,
                lambda sub_or_sol: sub_or_sol.ridge_regression(
                    train_path="train%d.csv" % n, validation_path="validation.csv"
                ),
            )

            self.assertTrue(
                np.allclose(student_val_err, solution_val_err, atol=0.1, rtol=0.1)
            )


def getTestCaseForTestID(test_id):
    question, part, _ = test_id.split("-")
    g = globals().copy()
    for name, obj in g.items():
        if inspect.isclass(obj) and name == ("Test_" + question):
            return obj("test_" + part)


if __name__ == "__main__":
    # Parse for a specific test
    parser = argparse.ArgumentParser()
    parser.add_argument("test_case", nargs="?", default="all")
    test_id = parser.parse_args().test_case

    assignment = unittest.TestSuite()
    if test_id != "all":
        assignment.addTest(getTestCaseForTestID(test_id))
    else:
        assignment.addTests(
            unittest.defaultTestLoader.discover(".", pattern="grader.py")
        )
    CourseTestRunner().run(assignment)
