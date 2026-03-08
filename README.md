# CompEcon

## A Python version of Miranda and Fackler's CompEcon toolbox

The aim of this toolbox is to provide a Python toolbox to replicate the funcionality of [Miranda and Fackler's CompEcon toolbox](https://github.com/PaulFackler/CompEcon), which was written to accompany their [Computational Economics and Finance](https://mitpress.mit.edu/books/applied-computational-economics-and-finance) and is coded in Matlab.

[The source for this project is available in Github](https://github.com/randall-romero/CompEcon-python).

A major difference in this implementation is that much of the code is object-oriented, providing classes to represent:

* Interpolation bases: Chebyshev, Spline, and Linear
* Dynamic programming models: with discrete and/or continuous state and action variables
* Nonlinear problems
* Optimization problems

Some other differences are:

* The solution of dynamic models is returned as a [pandas dataframe](https://pandas.pydata.org/), as opposed to a collection of vectors and matrices.
* Some additional functionality is included, most notably for [Smolyak interpolation](http://nber.org/papers/w19326).
* Basis objects are callable, so they can be used to interpolation function by "calling" the basis.


The toolbox also replicates some of the demos and examples from [Miranda and Fackler's textbook](https://mitpress.mit.edu/books/applied-computational-economics-and-finance). The examples can be found in the .\textbook directory, while the demos are in the directories .\demos (for .py files) and .\notebooks (for Jupyter notebooks).

At this time the documentation is incomplete, so the best way to get going with this toolbox is by exploring the notebooks. 
