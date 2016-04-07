try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
   
config = {
    'description': 'A Python version of CompEcon',
    'author': 'Randall Romero',
    'url': 'URL to get it at.',
    'download_url': 'Where do download it.',
    'author_email': 'randall.romero@outlook.com',
    'version': '0.1',
    'install_requires': ['nose'],
    'packages': ['compecon'],
    'scripts': [],
    'name': 'compecon'
    }
    
# setup(requires=['nose', 'numpy', 'scipy', 'matplotlib', 'matplotlib', 'seaborn', 'numba'], **config)  # fixme Uncomment when numba is available for python 3.5
setup(requires=['nose', 'numpy', 'scipy', 'pandas', 'matplotlib', 'matplotlib', 'seaborn','ggplot'], **config)