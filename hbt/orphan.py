from matplotlib import pyplot as plt
import numpy as np
import os

from HBTReader import HBTReader
from hbtpy import hbt_tools
from hbtpy.helpers.plot_auxiliaries import get_axlabel
from hbtpy.simulation import Simulation
from hbtpy.subhalo import Subhalos

from plottery.plotutils import savefig, update_rcParams
update_rcParams()


def main():
    args = hbt_tools.parse_args()
    sim = Simulation(args.simulation)
    reader = HBTReader(sim.path)

    isnap = -1
    subs = Subhalos(
        reader.LoadSubhalos(isnap), sim, isnap, logMmin=0,
        logM200Mean_min=13, logMstar_min=7)

    Nmin = (1, 20, 100)
    mstarbins = np.logspace(7, 10, 31)
    # the 0.5 makes it clear where each N falls
    nbins = np.append(np.array([-1, 0, 1, 2, 5]), np.arange(10, 101, 10)) + 0.5
    fig, axes = plt.subplots(
        1, 2, figsize=(12,5), constrained_layout=True)
    ax = axes[0]
    h = np.histogram(subs['Nbound'], nbins)[0]
    ax.bar(np.arange(h.size), h)
    xlabels = ['N = 0', 'N = 1', 'N = 2'] \
        + [f'{nbins[i-1]-0.5:.0f} < N \leq {nbins[i]-0.5:.0f}'
           for i in range(4, nbins.size)]
    xlabels = [f'${label}$' for label in xlabels]
    ax.set(xticks = np.arange(h.size), yscale='log', ylim=(0.8,320))
    ax.set_xticklabels(xlabels, rotation=45)
    ax = axes[1]
    ax.hist(subs['Mstar'], mstarbins, histtype='step', lw=2,
            label='All subhaloes')
    for i, Nmin_i in enumerate(Nmin):
        mask = (subs['Ndm'] <= Nmin_i)
        ax.hist(subs['Mstar'][mask], mstarbins, histtype='step', lw=3,
                color=f'C{i+1}', zorder=10-i,
                label=f'$N_\mathrm{{DM}}\leq{Nmin_i}$')
    for i, Nmin_i in enumerate(Nmin):
        mask = (subs['Nbound'] <= Nmin_i)
        ax.hist(subs['Mstar'][mask], mstarbins, histtype='step', lw=3,
                zorder=10-i, ls='--', color=f'C{i+1}')
    #ax.text()
    ax.set(xscale='log', yscale='log', xlabel='Mstar', ylabel='N')
    ax.legend(fontsize=14)
    output = os.path.join(sim.plot_path, 'orphan.png')
    savefig(output, fig=fig, tight=False)


main()
