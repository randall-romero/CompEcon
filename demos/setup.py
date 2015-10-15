__author__ = 'Randall'

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('dark')
np.set_printoptions(4, linewidth=120)

from matplotlib import rcParams
rcParams['lines.linewidth'] = 2.5
rcParams['figure.subplot.hspace'] = 0.25
rcParams['legend.frameon'] = False
rcParams['savefig.directory'] = '/figures/'
rcParams['savefig.format'] = 'pdf'

from compecon import tic, toc