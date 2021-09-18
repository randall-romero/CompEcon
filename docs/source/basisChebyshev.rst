BasisChebyshev
==============

A Chebyshev basis is a linear combination of Chebyshev polynomials :math:`\phi(x)` used to approximate a function :math:`f(x)`:

.. math::
    f(x) \approx \sum_{i=1}^{n} c_i\phi_i(x)

where :math:`c_i` are the collocation coefficients. The approximation uses :math:`n` polynomials over the interval :math:`[a, b]`.

---------------------------

BasisChebyshev class
--------------------

.. autoclass:: compecon.BasisChebyshev
    :members: _phi1d
    :undoc-members:
    :show-inheritance:
    :special-members: __init__, __call__
