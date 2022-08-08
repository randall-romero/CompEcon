---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# A Python Implementation of CompEcon

**Randall Romero-Aguilar**


## About this website

I have been working from time to time on porting from Matlab to Python the [CompEcon toolbox](https://github.com/PaulFackler/CompEcon) that comes with the [textbook by Mario Miranda and Paul Fackler](https://mitpress.mit.edu/books/applied-computational-economics-and-finance)


## Why a Python version

I used the original CompEcon code extensively in my research during my PhD years. Because I realized the huge importance for economists to learn the methods from computational economics, I knew back then that eventually I would like to teach this material to new students, but was afraid that most of them would not be able to afford a Matlab license, necessary to run the CompEcon toolbox.

With that in mind, and as exercise to more thoroughly learn both numerical methods and Python, I decided to write a Python version of this material. Although the work is not complete just yet, I have shared the Python code for some time in [GitHub](https://github.com/randall-romero/CompEcon). This website contains a collection of Jupyter notebooks that replicate, with minor deviations in some cases, most of the demos in the original CompEcon toolbox.

The notebooks on this website, however, do not explain any of the numerical methods employed to solve the problems. For that purpose, I highly recommend interested readers to study Miranda and Fackler's textbook.

By the way: I did end up teaching a course based on this book, in Spanish. You can see the lectures on [this playlist from my YouTube channel](https://www.youtube.com/playlist?list=PLs3Si3L7f5OoOv18HIS9sBzw94KsSuYsF).

## Some differences between the Python and Matlab implementations

A major difference in this Python implementation is that much of the code is object-oriented, providing classes to represent:

* Interpolation bases: Chebyshev, Spline, and Linear
* Dynamic programming models: with discrete and/or continuous state and action variables
* Nonlinear problems
* Optimization problems

Some other differences are:

* The solution of dynamic models is returned as a pandas dataframe, as opposed to a collection of vectors and matrices.
* Some additional functionality is included, most notably for Smolyak interpolation.
* Basis objects are callable, so they can be used to interpolate functions by “calling” the basis.


## Getting the toolbox

To start using the toolbox, you have two options: one is to run it in your own computer, the other is to run it in Google Colab . In both cases, you need to install the toolbox first, by running the system command:

    pip install compecon --upgrade

This needs to be done only once if you run it in your computer, or in every session if you do it in Google Colab (in which case you are installing CompEcon in a virtual machine, not in your own computer).




```{warning}
Although I have put my best effort to ensure the quality of this work, some bugs might still be found in the code. Of course, that would be my mistake, not the mistake of the authors of the original CompEcon toolbox (Matlab). If you do find a bug in this code, please let me know by an email to randall.romero@ucr.ac.cr.
```
