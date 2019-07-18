""" The CompEcon package setup.
Based on setuptools

Randall Romero-Aguilar, 2015-2018
"""


from setuptools import setup
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here,'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='compecon',
    version='2019.07',
    description='A Computational Economics Toolbox',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='http://randall-romero.com/code/compecon',
    author='Randall Romero-Aguilar',
    author_email='randall.romero@outlook.com',
    classifiers=['Development Status :: 4 - Beta',
                 'Intended Audience :: Computational Economists',
                 'Topic :: Numerical Methods for Economics',
                 'License :: OSI Approved :: MIT License',
                 'Programming Language :: Python :: 3.6'],
    keywords='computations economics numerical methods',
    packages=['compecon', 'compecon.demos','textbook','notebooks'],
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'seaborn',
        'numba',
        'nose'],
    project_urls={
        'Bug Reports': 'https://github.com/randall-romero/CompEcon-python/issues',
        'Source': 'https://github.com/randall-romero/CompEcon-python/'
    }
)