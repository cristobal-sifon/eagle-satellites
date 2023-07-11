import cmasher as cmr
from icecream import ic
from matplotlib import cm, colors, pyplot as plt, ticker
import multiprocessing as mp
import numpy as np
import os
from scipy.stats import (binned_statistic_2d as binstat2d, gaussian_kde)
import seaborn as sns
from tqdm import tqdm

from HBTReader import HBTReader
from hbtpy import hbt_tools
from hbtpy.helpers.plot_auxiliaries import (
    format_filename, get_axlabel)
from hbtpy.helpers.plot_definitions import xbins
from hbtpy.simulation import Simulation
from hbtpy.subhalo import Subhalos
from hbtpy.track import Track

from plottery.plotutils import savefig, update_rcParams
update_rcParams()

try:
    sns.set_color_palette('flare')
except AttributeError:
    pass


def main():
    args = hbt_tools.parse_args()
    sim = Simulation(args.simulation)
    reader = HBTReader(sim.path)

    isnap = -1
    kwargs = dict(logMmin=0, logM200Mean_min=13, logMstar_min=9)
    subs = Subhalos(reader.LoadSubhalos(isnap), sim, isnap, **kwargs)
    subs.catalog['Depth'] = np.min(
        [subs.catalog['Depth'], 3*np.ones(subs.catalog['Depth'].size)], axis=0)
    #subs.catalog['Depth'][subs['Depth'] > 3] = 3

    cols = ['TrackId', 'Depth', 'history:first_infall:time',
            'history:cent:time', 'Mstar/history:first_infall:Mstar',
            'Mbound/history:first_infall:Mbound', 'Mbound/Mstar']

    depth_history(args, reader, sim)

    xcol = 'history:first_infall:time'
    ycol = 'history:cent:time-history:first_infall:time'
    plot_fractions(subs, cols, xcol, ycol)
    ycol = ycol.split('-')[0]
    plot_fractions(subs, cols, xcol, ycol)
    return

    #pairplot_scipy(subs, cols)
    pairplot_sns(subs, cols)

    return

def plot_fractions(subs, cols, xcol, ycol):
    plot_fractions_1d(subs, cols, f'{ycol}-{xcol}')
    plot_fractions_2d(subs, cols, xcol, ycol)
    return


def plot_fractions_1d(subs, cols, xcol):
    tbins = np.arange(0, 13.6)
    return


def plot_fractions_2d(subs, cols, xcol, ycol):
    tbins = np.arange(0, 13.6, 0.5)
    dbins = np.arange(-13.5, 13.6, 0.5)
    bins = [dbins if '-' in col else tbins for col in (ycol,xcol)]
    s = subs.satellite_mask
    h = np.histogram2d(subs[ycol][s], subs[xcol][s], bins)[0]
    h1 = binstat2d(
            subs[ycol], subs[xcol], (subs['Depth'] == 1).astype(int),
            statistic='sum', bins=bins).statistic
    h2 = binstat2d(
            subs[ycol], subs[xcol], (subs['Depth'] >= 2).astype(int),
            statistic='sum', bins=bins).statistic
    h3 = binstat2d(
            subs[ycol], subs[xcol], (subs['Depth'] >= 3).astype(int),
            statistic='sum', bins=bins).statistic
    imkwargs = dict(origin='lower', cmap='magma', vmin=0, vmax=1)
    fig = plt.figure(figsize=(15, 6), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, height_ratios=[10,1])
    axes = [fig.add_subplot(grid[0,i]) for i in range(3)]
    caxes = [fig.add_subplot(grid[1,0]), fig.add_subplot(grid[1,1:])]
    im = axes[0].imshow(h, origin='lower', cmap='cmr.arctic_r')
    axes[0].set_title('Depth = 1', fontsize=15)
    plt.colorbar(im, cax=caxes[0], orientation='horizontal', label='N')
    im = axes[1].imshow(h2/h, **imkwargs)
    axes[1].set_title('Depth $\geq$ 2', fontsize=15)
    axes[2].imshow(h3/h, **imkwargs)
    axes[2].set_title('Depth $\geq$ 3', fontsize=15)
    axes[0].set_ylabel(get_axlabel(ycol))
    for ax in axes:
        ax.set_xlabel(get_axlabel(xcol))
    plt.colorbar(im, cax=caxes[1], orientation='horizontal', label='Fraction')
    #fig.tight_layout()
    path = os.path.join(subs.sim.plot_path, 'depth')
    os.makedirs(path, exist_ok=True)
    output = format_filename(f'depth_fraction__{xcol}__{ycol}.pdf')
    savefig(os.path.join(path, output), fig=fig, tight=False)
    return


def _track_depth_history(trackid, sim):
    track = Track(trackid, sim)
    return trackid, track['Depth']


def depth_history(args, reader, sim, z=0, nsub=12, seed=1):
    # if I choose z>0 then maybe I should be able to adjust
    # logM200Mean_min
    isnap = sim.snapshot_index_from_redshift(z)
    ic(isnap)
    subs = Subhalos(
        reader.LoadSubhalos(isnap), sim, isnap, logMmin=0, logMstar_min=9,
        logM200Mean_min=13)
    subs.sort('Mbound')
    rng = np.random.default_rng(seed=seed)
    idx = rng.choice(subs['TrackId'][subs.satellite_mask], size=nsub)
    #idx = idx[np.argsort(subs['Mbound'].iloc[idx].to_numpy())]
    ic(idx)
    # testing
    track = Track(idx[0], sim)
    ic(track.z)
    nrows, ncols = 3, 4
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.5*ncols,3.5*nrows), constrained_layout=True)
    axes = np.reshape(axes, -1)
    c = cmr.take_cmap_colors('cmr.rainforest', nsub, cmap_range=(0.2, 0.8))
    for i, (ax, trackid, color) in tqdm(enumerate(zip(axes, idx, c)),
                                        total=nsub):
        track = Track(trackid, sim)
        if i == 0:
            ic(np.sort(track.colnames))
        depth_history = track['Depth']
            # depth_history = np.append(
            #     np.zeros(track.birth_snapshot[0]), depth_history)
        life = np.s_[:-track.birth_snapshot[0]] if track.birth_snapshot[0] > 0 \
            else np.ones(depth_history.size, dtype=bool)
        ic(trackid, depth_history.shape, track.birth_snapshot)
        ax.plot(sim.t_lookback[life], depth_history, '-', color='k')
        # ax.annotate(f'Track {trackid}\n',
        #             (0.95,0.95), xycoords='axes fraction',
        #             ha='right', va='top', fontsize=14)
        ax.set_title(f'Track {trackid}', fontsize=14)
        ax.axhline(0.5, ls='--', color='0.5')
        yrng = ax.get_ylim()[1] - ax.get_ylim()[0]
        # ax.annotate('Central', xy=(13,0.5-0.03*yrng), ha='right', va='top',
        #             fontsize=12)
        # ax.annotate('Satellite', xy=(13,0.5+0.03*yrng), ha='right',
        #             va='bottom', fontsize=12)
        ax.set(xlim=(-0.5, 14), ylim=(-0.15, 5.15), yticks=range(5))
        ax.yaxis.set_minor_locator(ticker.NullLocator())
        if i >= nsub - ncols:
            ax.set(xlabel='$t_\mathrm{lookback}$ (Gya)')
        #if i % ncols == 0:
        # mass history
        rax = ax.twinx()
        rax.plot(sim.t_lookback[life], np.log10(track['Mstar']), 'C3-', lw=2)
        rax.plot(sim.t_lookback[life], np.log10(track['Mbound']), 'C0-', lw=2)
        if i % ncols == ncols - 1 and i % nrows == 1:
            ax.set(ylabel='Depth')
            rax.set_ylabel('log $\{m_\mathrm{sub},m_{\u2605}\}/$M$_\odot$')
    output = os.path.join(sim.plot_path, 'depth', 'depth_history.pdf')
    savefig(output, fig=fig, tight=False)
    return


def plot_times(subs, cols):
    xcol = 'history:cent:time'
    ycol = 'history:first_infall:time'
    fig, ax = plt.subplots()
    sns.kdeplot(
        subs.catalog[subs['Depth'] > 0], x=xcol, y=ycol, hue='Depth', ax=ax,
        bw_method=0.2, legend=False,
        palette=cmr.get_sub_cmap('magma', 0.3, 0.8), levels=[0.1,0.5,0.8])
    plt.legend(loc='upper left', fontsize=14)
    path = os.path.join(
        subs.sim.plot_path, 'depth')
    os.makedirs(path, exist_ok=True)
    output = format_filename(f'depth__{xcol}__{ycol}.pdf')
    savefig(os.path.join(path, output), fig=fig)
    return


def pairplot_scipy(subs, cols):
    nc = len(cols) - 1
    samples = [subs['Depth'] == 1, subs['Depth'] == 2, subs['Depth'] >= 3]
    ic([s.sum() for s in samples])
    fig, axes = plt.subplots(
        nc, nc, figsize=(2*nc,2*nc), constrained_layout=True)
    for i, xcol in enumerate(cols[1:]):
        for j, ycol in enumerate(cols[1:]):
            ax = axes[i,j]
            if j == i:
                plot_diagonal(ax, subs, samples, xcol)
            # lower triangle
            elif j < i:
                #plot_offdiag(ax, subs, xcol, ycol)
                ...
    return


def plot_diagonal(ax, subs, samples, col):
    bins = xbins[col]
    x = (bins[1:]+bins[:-1]) / 2
    for i, s in enumerate(samples):
        color = f'C{i}'
        kde = gaussian_kde(subs[col][samples])
        h = kde(x)
        ax.fill_between(x, h, color=color)
    return


def pairplot_sns(subs, cols):
    palette = cmr.get_sub_cmap('cmr.bubblegum', 0.2, 0.8)
    palette = sns.color_palette('flare', 3)
    pair_grid = sns.pairplot(
        subs[cols][subs.satellite_mask], hue='Depth', kind='kde', corner=True,
        plot_kws=dict(levels=[0.1, 0.5]),
        grid_kws=dict(despine=False),
        palette=palette)
    #pair_grid._legend.remove()
    pair_grid._legend.set(label=('1', '2', '$\geq3$'))
    fig = pair_grid.figure
    #pair_grid.map_diag(sns.kdeplot)
    #pair_grid.map_upper(sns.kdeplot, levels=[0.25,0.75])
    pair_grid.map_lower(sns.kdeplot, levels=[0.5,0.9], legend=False)
    for ax in fig.axes:
        
        xlabel = ax.get_xlabel()
        ylabel = ax.get_ylabel()
        if 'Mbound' in xlabel:# or 'Mstar' in xlabel:
            ax.set_xscale('log')
        # do not make diagonals log scale in y
        if xlabel != ylabel and ('Mbound' in ylabel):# or 'Mstar' in ylabel):
            ax.set_yscale('log')
        # custom limits
        # tcent
        # if xlabel == cols[2]: ax.set_xlim((0, 10))
        # if ylabel == cols[2]: ax.set_ylim((0, 10))
        # Mstar/Mstar_infall
        if xlabel == cols[3]: ax.set_xlim((0, 5))
        if ylabel == cols[3]: ax.set_ylim((0, 5))
        # Msub/Msub_infall
        if xlabel == cols[4]: ax.set_xlim((0.003, 2))
        if ylabel == cols[4]: ax.set_ylim((0.003, 2))
        # Msub/Mstar
        if xlabel == cols[5]: ax.set_xlim((1, 200))
        if ylabel == cols[5]: ax.set_ylim((1, 200))
        xlabel = get_axlabel(xlabel).replace('$$', '$')
        ylabel = get_axlabel(ylabel).replace('$$', '$')
        ax.set(xlabel=xlabel, ylabel=ylabel)
        for spine in ('bottom', 'right', 'top', 'left'):
            ax.spines[spine].set_color('k')
            ax.spines[spine].set_linewidth(2)
    fig.legend(labels=('Depth $\geq3$', 'Depth = 2', 'Depth = 1'),
               loc='upper center')
    savefig(os.path.join(subs.sim.plot_path, 'pairplots', 'depth.png'), fig=fig)

    return


main()