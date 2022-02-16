from icecream import ic
from matplotlib import cm, colors as mplcolors, pyplot as plt, ticker
from matplotlib.colors import ListedColormap
import multiprocessing as mp
import numpy as np
import os
from scipy.stats import (
    binned_statistic as binstat, binned_statistic_dd as binstat_dd)
import sys
from time import time

from plottery.plotutils import colorscale, savefig, update_rcParams
from plottery.statsplots import contour_levels
update_rcParams()

from ..hbt_tools import save_plot
#from hbtpy.hbtplots import RelationPlotter
#from hbtpy.hbtplots.core import ColumnLabel
from .plot_auxiliaries import (
    binning, definitions, get_axlabel, get_bins, get_label_bincol, logcenters,
    massbins, plot_line)
from .plot_definitions import (
    ccolor, scolor, massnames, units, xbins, binlabel, events, axlabel, ylims)


def run(sim, subs, ncores=1):
    return
