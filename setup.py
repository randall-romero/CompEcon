""" The CompEcon package setup.
Based on setuptools

Randall Romero-Aguilar, 2015-2022
"""


from setuptools import setup

with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='compecon',
    version='2022.09.09',
    description='A Computational Economics Toolbox',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='http://randall-romero.com/code/compecon',
    author='Randall Romero-Aguilar',
    author_email='randall.romero@outlook.com',
    keywords='computations economics numerical methods',
    packages=['compecon', 'compecon.demos','textbook','notebooks'],
    python_requires='>=3.7',
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'seaborn',
        'numba',
        'nose'],
    project_urls={
        'Bug Reports': 'https://github.com/randall-romero/CompEcon/issues',
        'Source': 'https://github.com/randall-romero/CompEcon/'
    }
)