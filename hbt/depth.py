from icecream import ic
from matplotlib import pyplot as plt
import numpy as np
import os
import seaborn as sns

from HBTReader import HBTReader
from hbtpy import hbt_tools
from hbtpy.helpers.plot_auxiliaries import get_axlabel
from hbtpy.simulation import Simulation
from hbtpy.subhalo import Subhalos

from plottery.plotutils import savefig, update_rcParams
update_rcParams()

sns.set_color_palette('flare')


def main():
    args = hbt_tools.parse_args()
    sim = Simulation(args.simulation)
    reader = HBTReader(sim.path)

    isnap = -1
    kwargs = dict(logMmin=0, logM200Mean_min=13, logMstar_min=7)
    subs = Subhalos(reader.LoadSubhalos(isnap), sim, isnap, **kwargs)
    subs.catalog['Depth'] = np.min(
        [subs.catalog['Depth'], 3*np.ones(subs.catalog['Depth'].size)], axis=0)

    cols = ['history:first_infall:time', 'history:cent:time',
            'Mstar/history:first_infall:Mstar',
            'Mbound/history:first_infall:Mbound', 'Mbound/Mstar', 'Depth']
    pair_grid = sns.pairplot(
        subs[cols][subs.satellite_mask], hue='Depth', kind='kde', corner=True,
        )
    help(pair_grid._legend)
    #pair_grid._legend.remove()
    pair_grid._legend.set(label=('1', '2', '$\geq3$'))
    fig = pair_grid.figure
    #pair_grid.map_diag(sns.kdeplot)
    #pair_grid.map_upper(sns.kdeplot, levels=[0.25,0.75])
    pair_grid.map_lower(sns.kdeplot, levels=[0.5,0.9], legend=False)
    for ax in fig.axes:
        xlabel = ax.get_xlabel()
        ylabel = ax.get_ylabel()
        if 'Mbound' in xlabel or 'Mstar' in xlabel:
            ax.set_xscale('log')
        # do not make diagonals log scale in y
        if xlabel != ylabel and ('Mbound' in ylabel or 'Mstar' in ylabel):
            ax.set_yscale('log')
        # custom limits
        if xlabel == cols[2]: ax.set_xlim((0.1, 10))
        if ylabel == cols[2]: ax.set_ylim((0.1, 10))
        if xlabel == cols[3]: ax.set_xlim((0.01, 10))
        if ylabel == cols[3]: ax.set_ylim((0.01, 10))
        if xlabel == cols[4]: ax.set_xlim((1, 500))
        if ylabel == cols[4]: ax.set_ylim((1, 500))
        xlabel = get_axlabel(xlabel).replace('$$', '$')
        ylabel = get_axlabel(ylabel).replace('$$', '$')
        ax.set(xlabel=xlabel, ylabel=ylabel)
        for spine in ('bottom', 'right', 'top', 'left'):
            ax.spines[spine].set_color('k')
            ax.spines[spine].set_linewidth(2)
    fig.legend(labels=('Depth $\geq3$', 'Depth = 2', 'Depth = 1'), loc='upper center')
    savefig(os.path.join(sim.plot_path, 'pairplots', 'depth.png'), fig=fig)

    return


main()