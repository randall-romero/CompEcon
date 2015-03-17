__author__ = 'Randall'

def quadratic(a, b, c):
    """
    A quadratic formula

    :param a: quadratic term
    :param b: linear term
    :param c: intercept
    :return: a quadratic function
    """
    return lambda x: a*x**2 + b*x + c


