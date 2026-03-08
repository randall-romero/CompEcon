import numpy as np
from numpy.linalg import solve, norm

eps = np.sqrt(np.spacing(1.0))

def float_matrix(*args):
    """
    Takes array M (possibly a list or tuple) and converts it to a numpy array of floats
    :param *args: several arrays
    :return: numpy arrays
    """
    return (np.asarray(m).astype(float) for m in args)

def gjacobi(A,b,x0=None,maxit=1000, tol=eps):
    """
    Solves AX=b using Gauss-Jacobi iterations

    :param A: n.n numpy array
    :param b: n numpy array
    :param x0: n numpy array of starting values, default b
    :param maxit: int, maximum number of iterations
    :param tol: float, convergence tolerance
    :return: n numpy array
    """

    A, b = float_matrix(A, b)

    if x0 is None:
        x = b.copy()
    else:
        x = float_matrix(x0)

    Q = np.diag(np.diag(A)) # diagonal of A matrix
    for i in range(maxit):
        dx = solve(Q, b - A@x)
        x += dx
        if norm(dx) < tol:
            return x
    raise('Maximum iterations exceeding in gjacobi')


def gseidel(A, b, x0=None, lambda_=1.0, maxit=1000, tol=eps):
    """
    Solves AX=b using Gauss-Seidel iterations

    :param A: n.n numpy array
    :param b: n numpy array
    :param x0: n numpy array of starting values, default b
    :param lambda_: float, over-relaxation parameter
    :param maxit: int, maximum number of iterations
    :param tol: float, convergence tolerance
    :return: n numpy array
    """
    A, b = float_matrix(A, b)

    if x0 is None:
        x = b.copy()
    else:
        x = float_matrix(x0)

    Q = np.tril(A)  # lower triangle part of A
    for i in range(maxit):
        dx = solve(Q, b - A @ x)
        x += (lambda_ * dx)
        if norm(dx) < tol:
            return x
    raise ('Maximum iterations exceeding in gseidel')
