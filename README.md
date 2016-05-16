# CompEcon-python
A Python version of Miranda and Fackler's CompEcon toolbox

The aim of this toolbox is to provide a Python toolbox to replicate the funcionality of Miranda and Fackler's CompEcon toolbox, which is coded in Matlab.  A major difference in this implementation is that much of the code is object-oriented, providing classes to represent:

* Chebyshev bases
* Spline bases
* Linear bases
* Dynamic programming models
* Discrete dynamic programming models
* Nonlinear problems
* Optimization problems

Some other differences are:

* The solution of dynamic models is returned as a pandas dataframe, as opposed to a collection of vectors.
* Some additional functionality is included, most notably for Smolyak interpolation.
* Basis objects are callable, so they can be used to interpolation function by "calling" the basis.



The toolbox also replicates some of the demos and examples from Miranda and Fackler's textbook. The examples can be found in the .\textbook directory, while the demos are in the directories .\demos (for .py files) and .\notebooks (for Jupyter notebooks).

At this time the documentation is incomplete, so the best way to get going with this toolbox is by exploring the notebooks. 
