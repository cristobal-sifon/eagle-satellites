from icecream import ic
from matplotlib import pyplot as plt, ticker
import multiprocessing as mp
import numpy as np
import os
from time import time

from plottery.plotutils import colorscale, savefig, update_rcParams
update_rcParams()

from HBTReader import HBTReader
from hbtpy import hbt_tools
from hbtpy.hbt_tools import save_plot
from hbtpy.simulation import Simulation
from hbtpy.subhalo import HostHalos, Subhalos
from hbtpy.track import Track, HaloTracks


def main(debug=True):

    args = hbt_tools.parse_args()
    sim = Simulation(args.simulation)

    to = time()
    reader = HBTReader(sim.path)
    print('Loaded reader in {0:.1f} seconds'.format(time()-to))

    isnap = -1
    # min masses are the same as in mass_relations.py
    subs = Subhalos(
        reader.LoadSubhalos(isnap), sim, isnap, logMmin=8,
        logM200Mean_min=13)
    print('Loaded subhalos!')

    shmr_history(subs)
    return


def shmr_history(subhalos):
    subs = subhalos.catalog
    sats = subhalos.satellites
    cens = subhalos.centrals
    most_massive = np.argmax(cens['M200Mean'])
    ic(most_massive)
    ic(np.log10(cens['M200Mean'].iloc[most_massive]))
    ic(cens['TrackId'].iloc[most_massive])
    ic(cens['HostHaloId'].iloc[most_massive])
    hmsr = subhalos.hsmr(sample='satellites')
    halo_shmr_evolution(
        subs, subhalos.sim, cens['HostHaloId'].iloc[most_massive], n=50,
        shmr=subhalos.hsmr(sample='satellites'), ncores=13)
    # halo_shmr_evolution(
    #     subs, subhalos.sim, cens['HostHaloId'].iloc[most_massive], n=30,
    #     shmr=subhalos.hsmr(sample='satellites'))
    # halo_shmr_evolution(
    #     subs, subhalos.sim, cens['HostHaloId'].iloc[most_massive], seed=66)

    return


def halo_shmr_evolution(subs, sim, host_halo_id, seed=None, n=30,
                        shmr=None, ncores=10):
    """
    For now, shmr should be the output of either of `Subhalos.hsmr` or
    `Subhalos.shmr`
    """
    ic()
    assert (n > 0) or (seed is not None and seed >= 0), \
        'either `n` or `seed` must be specified and greater than zero'
    ic(host_halo_id)
    np.random.seed(seed)
    halo = subs[subs['HostHaloId'] == host_halo_id]
    ic(halo.shape)
    c = halo['TrackId'].loc[halo['Rank'] == 0].iloc[0]
    ic(c)
    cent = Track(c, sim)
    #ic(cent)
    sats = halo.loc[halo['Rank'] > 0]
    # vmin, vmax so the color scale is consistent
    vmin = sim.cosmology.age(sim.redshifts[0]).value
    vmax = sim.cosmology.age(sim.redshifts[-1]).value
    ic(vmin, vmax)
    # doing it randomly does not pick too many interesting objects
    if seed is not None and seed >= 0:
        jsat = np.append(np.argmax(sats['Mbound']),
                         np.random.randint(0, sats.shape[0], n))
        outputs = [f'shmr_track/shmr_track_{c}_seed{seed}_{when}'
                   for when in ('infall','present')]
    # let's do the 15 most massive (in stars)
    else:
        jsat = np.argsort(sats['MboundType4'])[-n:].values
        outputs = [f'shmr_track/shmr_track_{c}_n{n}_{when}'
                   for when in ('infall','present')]
    ic(jsat)
    ti = time()
    load_kwargs = dict(vmin=vmin, vmax=vmax)
    if ncores > 1:
        with mp.Pool(ncores) as pool:
            out = [pool.apply_async(
                       load_shmr_track, args=(sats['TrackId'].iloc[j],sim),
                       kwds=load_kwargs)
                   for j in jsat]
            pool.close()
            pool.join()
        out = [i.get() for i in out]
        xy_infall = [i[0] for i in out]
        xy_present = [i[1] for i in out]
    else:
        xy_infall, xy_present = [], []
        for i, j in enumerate(jsat):
            if i % 10 == 0:
                print(f'i={i:2d}, jsat={j:5d}...')
            trackid = sats['TrackId'].iloc[j]
            xy = load_shmr_track(trackid, sim, **load_kwargs)
            xy_infall.append(xy[0])
            xy_present.append(xy[1])
    print(f'Loaded tracks in {time()-ti:.2f} s')
    # plot
    fig_infall, ax_infall = plt.subplots(figsize=(7,6))
    fig_present, ax_present = plt.subplots(figsize=(7,6))
    # ax.set(xscale='log', yscale='log')
    # plot overall SHMR for reference
    if shmr is not None:
        ax_present.plot(*shmr, color='k', lw=3, zorder=-100)
        pass
    colors, cmap = colorscale(n=cent.z.size, vmin=vmin, vmax=vmax)
    # not plotting the central subhalo
    xy_infall_c, xy_present_c = load_shmr_track(c, sim, **load_kwargs)
    # note that the third elemenf of x_*_c is `colors`, which is
    # the first kwarg of `plot_shmr_track`
    plot_shmr_track(ax_infall, *xy_infall_c, is_central=True)
    plot_shmr_track(ax_present, *xy_present_c, is_central=True)
    # plot_shmr_track(
    #     ax_infall, ax_present, c, sim, vmin=vmin, vmax=vmax, is_central=True)
    # plot subhalos
    for xy_i, xy_p in zip(xy_infall, xy_present):
        plot_shmr_track(ax_infall, xy_i[0], xy_i[1], colors=xy_i[2])
        plot_shmr_track(ax_present, xy_p[0], xy_p[1], colors=xy_p[2])
    annot_kwargs = dict(
        xy=(0.95,0.05), xycoords='axes fraction', ha='right', va='bottom',
        fontsize=18)
    ax_infall.annotate(
        r'$t_\mathrm{birth}\,\rightarrow\,t_\mathrm{infall}$', **annot_kwargs)
    ax_present.annotate(
        r'$t_\mathrm{infall}\,\rightarrow\,$today', **annot_kwargs)
    for ax in (ax_infall, ax_present):
        plt.colorbar(cmap, ax=ax, label='Universe age (Gyr)')
        ax.set_xlabel(r'$\log\,{0}$'.format(sim.masslabel(mtype='stars')))
        ax.set_ylabel(r'$\log\,{0}$'.format(sim.masslabel(mtype='total')))
        ax.set(xlim=(9,13), ylim=(10,14.7))
    for fig, output in zip((fig_infall,fig_present), outputs):
        save_plot(fig, output, sim)
    return


def load_shmr_track(trackid, sim, mkey='Mbound', vmin=0, vmax=14):
    track = Track(trackid, sim)
    colors, _ = colorscale(n=track.z.size, vmin=vmin, vmax=vmax)
    if mkey == 'Mbound':
        mass = track.Mbound
    elif mkey == 'M200Mean':
        mass = track.track['M200Mean']
    x = np.log10(track.MboundType[:,4])
    y = np.log10(mass)
    i = track.infall() + 1
    return [x[:i], y[:i], colors[:i]], [x[i:], y[i:], colors[i:]]


def plot_shmr_track(ax, x, y, colors='k', size=6,
                    scatter_kwargs={}, is_central=False):
    ax.scatter(x, y, s=size, c=colors, zorder=-10, **scatter_kwargs)
    ax.plot(x, y, '-', color='k', lw=0.5, zorder=0)
    ls = '*' if is_central else 'o'
    ax.plot(
        x[-1], y[-1], ls, mec='k', mew=1, ms=size, mfc=colors[-1], zorder=10)
    return


main()
