#
# handy_functions.py
#
"""Module to define commonly used functions."""
import sys
import numpy
import logging

# np array output values
max_line_width = int(300)
precision = int(9)
suppress_small = True

# this is a wrapper for the numpy version of repr, which uses the variables defined above


def np_repr(ndarry_object):
    return numpy.array_repr(ndarry_object, max_line_width, precision, suppress_small)


# this is a wrapper for the numpy version of print, which uses the variables defined above


def np_print(ndarry_object):
    print(np_repr(ndarry_object))


logging.basicConfig(
    level=logging.DEBUG,
)

# a generic error messages about files


def file_error(filename, action):
    print(
        "ERROR could not " + str(action) + " file: '" + str(filename) + "'",
        file=sys.stderr,
    )


# a generic warning message


def warning(*objs):
    print("WARNING: ", *objs, file=sys.stderr)


# if the system should shutdown


def crash():
    sys.exit(0)


# generic exception


class NegativeCoefficientException(Exception):
    def __init__(self, truth_array):
        self.value = truth_array

    def __str__(self):
        return repr(self.value)
