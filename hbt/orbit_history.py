from icecream import ic
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
import multiprocessing as mp
import numpy as np
import os
import warnings

from plottery.plotutils import colorscale, savefig, update_rcParams
from plottery.statsplots import contour_levels
update_rcParams()

from ..hbt_tools import load_subhalos, parse_args, save_plot
from .plot_auxiliaries import (
    binning, definitions, format_filename, get_axlabel, get_bins,
    get_label_bincol, logcenters, massbins, plot_line)
from .plot_definitions import (
    ccolor, scolor, massnames, units, xbins, binlabel, events, axlabel, ylims)


warnings.simplefilter('ignore', RuntimeWarning)


def main():
    args = parse_args()
    subs = load_subhalos(args, -1, verbose=True)
    return


if __name__ == '__main__':
    main()
