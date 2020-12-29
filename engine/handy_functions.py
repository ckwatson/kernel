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


logging.basicConfig(level=logging.DEBUG, )

# a generic error messages about files


def file_error(filename, action):
    print("ERROR could not " + str(action) + " file: \'" +
          str(filename) + "\'", file=sys.stderr)

# a generic warning message


def warning(*objs):
    print("WARNING: ", *objs, file=sys.stderr)

# if the system should shutdown


def crash():
    sys.exit(0)

# generic exception


class User(Exception):
    def __init__(self, truth_array):
        self.value = truth_array

    def __str__(self):
        return repr(self.value)


def test():

    # testing out a replacement for the print statements all over the code
    # just copy pasted code from the python website.
    import logging
    import logging.config

    # apprently this method is outdated though
    logging.config.fileConfig('logging.conf')

    # create logger
    logger = logging.getLogger('simpleExample')

    # 'application' code
    logger.debug('debug message')
    logger.info('info message')
    logger.warn('warn message')
    logger.error('error message')
    logger.critical('critical message')

    # done
    logging.basicConfig(format='%(asctime)s %(message)s',
                        datefmt='%m/%d/%Y %I:%M: %p')  # dont need seconds
    #logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

    logging.basicConfig(filename='example.log',
                        filemode='w', level=logging.DEBUG)

    # code from python website https://docs.python.org/3/howto/logging.html#logging-basic-tutorial
    # about taking command line args to change logging, might be useful in the future

    # assuming loglevel is bound to the string value obtained from the
    # command line argument. Convert to upper case to allow the user to
    # specify --log=DEBUG or --log=debug
    #numeric_level = getattr(logging, loglevel.upper(), None)
    # if not isinstance(numeric_level, int):
    #    raise ValueError('Invalid log level: %s' % loglevel)
    # logging.basicConfig(level=numeric_level, ...)


if((__name__ == "__main__")):
    test()
