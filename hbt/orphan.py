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

    Nmin = (0, 20, 100)
    mstarbins = np.logspace(7, 10, 31)
    fig, ax = plt.subplots(figsize=(6,5), constrained_layout=True)
    ax.hist(subs['Mstar'], mstarbins, histtype='step', lw=2,
            label='All subhaloes')
    for i, Nmin_i in enumerate(Nmin):
        mask = (subs['Ndm'] <= Nmin_i)
        ax.hist(subs['Mstar'][mask], mstarbins, histtype='step', lw=3,
                zorder=10-i, label=f'$N_\mathrm{{DM}}\leq{Nmin_i}$')
    ax.set(xscale='log', yscale='log', xlabel='Mstar', ylabel='N')
    ax.legend(fontsize=14)
    output = os.path.join(sim.plot_path, 'orphan.png')
    savefig(output, fig=fig, tight=False)


main()
