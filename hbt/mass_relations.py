from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
import sys
from time import time
if sys.version_info[0] == 2:
    range = xrange

from plottools.plotutils import savefig, update_rcParams
update_rcParams()

from HBTReader import HBTReader
from core import hbt_tools
from core.simulation import Simulation
from core.subhalo import Subhalos, Track


def main(debug=True):

    args = hbt_tools.parse_args()
    sim = Simulation(args.simulation)

    to = time()
    reader = HBTReader(sim.path)
    print('Loaded reader in {0:.1f} seconds'.format(time()-to))

    # this applies to apostle
    mstarbins = mbins(sim, 'stars')
    subs = reader.LoadSubhalos(-1)
    sat = Subhalos(subs[subs['Rank'] > 0], sim)
    cen = Subhalos(subs[subs['Rank'] == 0], sim)
    print(sat.subhalos.dtype)

    plot_relation(sim, cen, sat, mstarbins, debug=debug)

    subs = Subhalos(subs, sim)
    to = time()
    dist3d = sat.distance2host(cen.subhalos['TrackId'][0])
    print('Measured distances in {0:.2f} minutes'.format((time()-to)/60))

    return


def plot_relation(sim, cen, sat, bins, xname='stars', yname='total',
                  ccolor='C0', scolor='C1', debug=False):
    sgood = ~sat.orphan
    # centrals should all be "good" but you never know
    cgood = ~cen.orphan
    x = (bins[:-1]+bins[1:]) / 2
    msat = average(sat, bins, mask=sgood, debug=debug)
    mcen = average(cen, bins, mask=cgood, debug=debug)
    fig, ax = plt.subplots(figsize=(8,6))
    # plot all objects
    # downsample if too many objects
    if sim.family == 'apostle':
        ax.plot(cen.mass(xname)[cgood], cen.mass(yname)[cgood], 'o',
                color=ccolor, ms=4, label='_none_')
        ax.plot(sat.mass(xname)[sgood], sat.mass(yname)[sgood], 's',
                color=scolor, ms=4, label='_none_')
    ax.plot(x, mcen, '-', color=ccolor, lw=3, label='Centrals')
    ax.plot(x, msat, '-', color=scolor, lw=3, label='Satellites')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(axlabel(sim, xname))
    ax.set_ylabel(axlabel(sim, yname))
    ax.annotate(
        sim.formatted_name, xy=(0.05,0.92), xycoords='axes fraction',
        ha='left', va='top', fontsize=16)
    ax.legend(loc='lower right')
    out = os.path.join(
        sim.plot_path, '{0}.pdf'.format(output(sim, xname, yname)))
    savefig(out, fig=fig)
    return


def average(sample, xbins, xname='stars', yname='total', mask=None,
            debug=False):
    if mask is None:
        mask = np.ones(sample.mass('total').size, dtype=bool)
    if debug:
        print('n =', np.histogram(sample.mass(xname), xbins)[0])
    yavg = np.histogram(
        sample.mass(xname)[mask], xbins,
        weights=sample.mass(yname)[mask])[0]
    yavg = yavg / np.histogram(sample.mass(xname)[mask], xbins)[0]
    return yavg

##
## Auxiliary functions
##


def axlabel(sim, name):
    return r'{0} (M$_\odot$)'.format(sim.masslabel(mtype=name))


def mbins(sim, mtype='stars'):
    bins = {'apostle': {'stars': np.logspace(3, 11, 21)},
            'eagle': {'stars': np.logspace(7, 13, 31)}}
    return bins[sim.family][mtype]


def output(sim, xname, yname):
    return '{0}_{1}'.format(sim.masslabel(mtype=xname, latex=False),
                            sim.masslabel(mtype=yname, latex=False))



main()




