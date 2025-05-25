import numpy as np

from kernel.engine.plotter import make_same_length


def test_make_same_length_true_data_shorter():
    true_data = np.array([[1, 2], [3, 4]])
    user_data = np.array([[1, 2, 3], [4, 5, 6]])
    result = make_same_length("jobid", true_data, user_data)
    expected = np.array([[1, 2, 2], [3, 4, 4]])
    np.testing.assert_array_equal(result, expected)


def test_make_same_length_true_data_longer():
    true_data = np.array([[1, 2, 3], [4, 5, 6]])
    user_data = np.array([[1, 2], [3, 4]])
    result = make_same_length("jobid", true_data, user_data)
    expected = np.array([[1, 2], [4, 5]])
    np.testing.assert_array_equal(result, expected)


def test_make_same_length_equal_lengths():
    true_data = np.array([[1, 2, 3], [4, 5, 6]])
    user_data = np.array([[7, 8, 9], [10, 11, 12]])
    result = make_same_length("jobid", true_data, user_data)
    np.testing.assert_array_equal(result, true_data)
