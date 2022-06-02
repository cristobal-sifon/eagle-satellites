from icecream import ic
from matplotlib import pyplot as plt, ticker
from matplotlib.collections import LineCollection
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
import numpy as np
import os
import sys
from time import time
import warnings

from plottery.plotutils import colorscale, savefig, update_rcParams
update_rcParams()

from HBTReader import HBTReader
from hbtpy import hbt_tools
from hbtpy.hbt_tools import save_plot
from hbtpy.simulation import Simulation
from hbtpy.subhalo import HostHalos, Subhalos
from hbtpy.track import Track, HaloTracks


warnings.simplefilter('ignore', RuntimeWarning)

def main(debug=True):

    args = hbt_tools.parse_args(
        args=[('--halo', {'default': None, 'type': int}),
              ('--minmass', {'default': 5e13, 'type': float}),
              ('--nhalos', {'default': 0, 'type': int}),
              ('--nsub', {'default': 50, 'type': int}),
              ('--seed', {'default': 31, 'type': int}),
              ('--test', {'action': 'store_true'}),
              ])
    np.random.seed(args.seed)
    sim = Simulation(args.simulation)

    to = time()
    reader = HBTReader(sim.path)
    print('Loaded reader in {0:.1f} seconds'.format(time()-to))

    isnap = -1
    # min masses are the same as in mass_relations.py
    subs = Subhalos(
        reader.LoadSubhalos(isnap), sim, isnap, logMmin=8,
        logM200Mean_min=9)
    print('Loaded subhalos!')

    shmr_history(args, subs)
    return


def shmr_history(args, subs):
    sats = subs.satellites
    cens = subs.centrals
    jsort = np.argsort(cens['M200Mean']).values
    starts = ('birth', 'sat', 'first_infall', 'last_infall')
    ends = ('sat', 'first_infall', 'last_infall', 'today')
    if args.test:
        j = jsort[-5]
        halo_shmr_evolution(
            args, subs, subs.sim, cens['HostHaloId'].iloc[j], n=10, seed=66,
            hostmass=cens['M200Mean'].iloc[j])
        return
    if args.halo is not None:
        j = (cens['TrackId'].iloc[jsort] == args.halo).values
        jarr = jsort[j]
    else:
        j = (cens['M200Mean'].iloc[jsort] > args.minmass)
        jarr = jsort[j]
        if args.nhalos > 0:
            jarr = np.random.choice(jarr, args.nhalos, replace=False)
    for j in jarr:
        print()
        logm = np.log10(cens['M200Mean'].iloc[j])
        trackid = cens['TrackId'].iloc[j]
        print(f'HostHalo TrackID {trackid} (logM={logm:.2f})')
        for i, start in enumerate(starts):
            _ = [halo_shmr_evolution(
                    args, subs, subs.sim, cens['HostHaloId'].iloc[j],
                    n=args.nsub, ncores=args.ncores, start=start, end=end,
                    hostmass=cens['M200Mean'].iloc[j])
                 for end in ends[i:]]
    return


def halo_shmr_evolution(args, subs, sim, host_halo_id, seed=None, n=30,
                        cmap='inferno_r', vmin=0, vmax=14,
                        ncores=10, start='birth', end='today', hostmass=None,
                        show_shsmr=True, show_chsmr=True, show_central=True,
                        draw_constant_ratios=True):
    """
    For now, shmr should be the output of either of `Subhalos.hsmr` or
    `Subhalos.shmr`
    """
    _checkpoints = ['birth', 'cent', 'sat', 'first_infall', 'last_infall',
                    'today']
    ic()
    assert (n > 0) or (seed is not None and seed >= 0), \
        'either `n` or `seed` must be specified and greater than zero'
    ic(host_halo_id)
    np.random.seed(seed)
    hhid = subs['HostHaloId']
    hhid_mask = (subs['HostHaloId'] == host_halo_id)
    ic(hhid_mask.sum())
    halo = subs[hhid_mask]
    ic(halo.shape)
    c = halo['TrackId'].loc[halo['Rank'] == 0].iloc[0]
    ic(c)
    #haloid =
    cent = Track(c, sim)
    #ic(cent)
    sats = halo.loc[halo['Rank'] > 0]
    # vmin, vmax so the color scale is consistent
    if vmin is None:
        vmin = sim.cosmology.age(sim.redshifts[0]).value
    if vmax is None:
        vmax = sim.cosmology.age(sim.redshifts[-1]).value
    ic(vmin, vmax)
    output = define_output(args, halo, start, end)
    ic(output)
    jsat = select_satellites(args, sats['MboundType4'].values)
    nsat = (sats['MboundType4'] > 0)
    print(f'{c}: Selected {jsat.size}/{nsat.sum()} satellites')
    ic(jsat)
    ti = time()
    load_kwargs = dict(vmin=vmin, vmax=vmax, start=start, end=end, cmap=cmap)
    if ncores > 1:
        with Pool(ncores) as pool:
            out = [pool.apply_async(
                       load_shmr_track, args=(sats['TrackId'].iloc[j],sim),
                       kwds=load_kwargs)
                   for j in jsat]
            pool.close()
            pool.join()
        out = [i.get() for i in out]
        xy = [i for i in out if i is not None and i[0].size > 0]
    else:
        xy_infall, xy_present = [], []
        for i, j in enumerate(jsat):
            if i % 10 == 0:
                print(f'i={i:2d}, jsat={j:5d}...')
            trackid = sats['TrackId'].iloc[j]
            xy = load_shmr_track(trackid, sim, **load_kwargs)
            if xy is None:
                continue
            xy_infall.append(xy[0])
            xy_present.append(xy[1])
    print(f'{c}: Loaded tracks in {time()-ti:.2f} s')
    # plot
    fig, ax = plt.subplots(figsize=(7,6))
    # plot overall SHMR for reference
    if show_chsmr:
        kw = dict(sample='centrals', ls='-.', color='k', lw=1,
                  hostmass='M200Mean', zorder=10,
                  bins=np.logspace(8, 12, 9), min_hostmass=1e9,)
        subs.hsmr(ax=ax, **kw)
        #subs.hsmr(ax=ax_present, **kw)
    if show_shsmr:
        kw = dict(sample='satellites', ls='-', color='k', lw=1,
                  hostmass='M200Mean', zorder=10,
                  bins=np.logspace(8, 12, 9), min_hostmass=1e13)
        subs.hsmr(ax=ax, **kw)
    colors, cmap = colorscale(n=cent.z.size, vmin=vmin, vmax=vmax, cmap=cmap)
    # not plotting the central subhalo
    if show_central:
        load_kwargs['start'] = 'birth'
        load_kwargs['end'] = 'today'
        xy_c = load_shmr_track(c, sim, **load_kwargs)
        if xy_c is not None:
            plot_shmr_track(ax, *xy_c, is_central=True, size=10)
    # plot subhalos
    _ = [plot_shmr_track(ax, x, y, colors=c) for x, y, c in xy]
    if draw_constant_ratios:
        rc = '0.5'
        x = np.logspace(8, 12, 10)
        for r in np.logspace(0, 3, 4):
            line = ax.plot(x, r*x, rc, lw=1, dashes=(6,4))
            if r < 0:
                msg = fr'$m_\mathrm{{sub}}/'+'m_\u2605'+f'={r:.1f}$'
            else:
                msg = fr'$m_\mathrm{{sub}}/'+'m_\u2605'+f'={r:.0f}$'
                if r in (1, 1000):
                    ax.text(
                        5e11/r**0.5, 0.70*r**0.5*5e11, msg,
                        va='top', ha='right',
                        color=rc, rotation=40, fontsize=12)
    annot_kwargs = dict(
        xy=(0.05,0.95), xycoords='axes fraction', ha='left', va='top',
        fontsize=18)
    #chkp = checkpoint.replace('_', ' ')
    start, end = [i.replace('_', ' ') for i in (start, end)]
    if hostmass is not None:
        logm = np.log10(hostmass)
        mlabel = rf'$\log\,M_\mathrm{{200m}}/\mathrm{{M}}_\odot = {logm:.2f}$'
    #ax_infall.annotate(
        ##r'$t_\mathrm{birth}\,\rightarrow\,t_\mathrm{infall}$', **annot_kwargs)
        #f'{mlabel}' + '\n' + fr'birth $\rightarrow$ {chkp}', **annot_kwargs)
    #ax_present.annotate(
        ##r'$t_\mathrm{infall}\,\rightarrow\,$today', **annot_kwargs)
        #f'{mlabel}' + '\n' + fr'{chkp} $\rightarrow$ today', **annot_kwargs)
    if hostmass is not None:
        ax.annotate(
            #f'{mlabel}\n' + fr'{start} $\rightarrow$ {end}',
                #f'Halo ID {},
            '\n'.join([f'{mlabel}', fr'{start} $\rightarrow$ {end}']),
            **annot_kwargs)
    ylim = (3e9,2e14) if show_central else (3e9,1e14)
    #for ax in (ax_infall, ax_present):
    plt.colorbar(cmap, ax=ax, label='Universe age (Gyr)')
    ax.set_xlabel('$m_{\u2605}$'+r' $(\mathrm{M}_\odot)$')
    ax.set_ylabel(r'$m_\mathrm{sub}\,(\mathrm{M}_\odot)$')
    ax.set(xlim=(1e8,1e12), ylim=ylim, xscale='log', yscale='log')
    # for fig, output in zip((fig_infall,fig_present), outputs):
    #     save_plot(fig, output, sim)
    save_plot(fig, output, sim)
    return


def load_shmr_track(trackid, sim, mkey='Mbound', vmin=0, vmax=14,
                    start='birth', end='today', cmap='viridis'):
    """
    start, end must any valid checkpoint:
        ('birth', 'cent', 'sat', 'first_infall', 'last_infall', 'today'))
    """
    _checkpoints = [None, 'birth', 'cent', 'sat', 'first_infall',
                    'last_infall', 'today']
    assert start in _checkpoints, f'start must be one of {_checkpoints}'
    assert end in _checkpoints, f'end must be one of {_checkpoints}'
    # note that in the current implementation it is possible to have
    # t_cent > t_sat for a given subhalo
    assert not (start == 'cent' and end == 'sat'), \
        f'ambiguous start and end: {start} --> {end}'
    assert _checkpoints.index(start) < _checkpoints.index(end) \
            or end is None, \
        f'start ({start}) must be an earlier checkpoint than end ({end})'
    track = Track(trackid, sim)
    colors, _ = colorscale(n=track.z.size, vmin=vmin, vmax=vmax, cmap=cmap)
    if mkey == 'Mbound':
        mass = track.Mbound
    elif mkey == 'M200Mean':
        mass = track.track['M200Mean']
    x = track.MboundType[:,4]
    y = mass
    if start is None or start == 'birth':
        istart = 0
    else:
        istart = track.history(cols='isnap', when=start, raise_error=False)
        if istart is None:
            istart = -1
    if end is None or end == 'today':
        iend = None
    else:
        iend = track.history(cols='isnap', when=end, raise_error=False)
        if iend == istart:
            iend += 1
    ic(trackid, start, end, istart, iend)
    #if istart is None or iend is None:
        #return None
    j = np.s_[istart:iend]
    return x[j], y[j], colors[j]


def plot_shmr_track(ax, x, y, colors='k', size=6,
                    scatter_kwargs={}, is_central=False):
    #ax.scatter(x, y, s=size, c=colors, zorder=-10, **scatter_kwargs)
    #ax.plot(x, y, '-', color='k', lw=0.5, zorder=0)
    ic(colors.shape)
    segments = np.transpose([
        np.array([x[:-1], x[1:]]), np.array([y[:-1], y[1:]])], axes=(2,1,0))
    ic(segments.shape)
    lines = LineCollection(segments, colors=colors[1:])
    #ic(lines)
    ax.add_collection(lines)
    ls = '*' if is_central else 'o'
    # white border for clarity
    ax.plot(
        x[-1], y[-1], ls, mec='w', mew=1, ms=size+1, zorder=99)
    ax.plot(
        x[-1], y[-1], ls, mec='k', mew=1, ms=size,
        mfc='w' if is_central else colors[-1], zorder=100)
    return

def define_output(args, halo, start, end):
    mass_bins = (13, 13.3, 13.7, 14, 14.3, 15)
    c = halo['TrackId'].loc[halo['Rank'] == 0].iloc[0]
    logm = np.log10(halo['M200Mean'].iloc[0])
    logmlabel = f"{logm:.2f}".replace('.', 'p')
    jbin = np.digitize(logm, mass_bins)
    mbins_label = [f'{i:.1f}'.replace('.', 'p')
                   for i in mass_bins[jbin-1:jbin+1]]
    output = os.path.join(
        'shmr_history', f'logm_{mbins_label[0]}_{mbins_label[1]}',
        f'shmr_track{c}__{start}__{end}')
    return output


def select_satellites(args, Mstar, nbins=10, logmstar_min=8.5):
    # selecting satellites this way allows for good coverage in mass,
    # randomly
    ic()
    jmass = np.argsort(Mstar)
    logm = np.log10(Mstar[jmass])
    good = np.isfinite(logm)
    ic(good.sum())
    if good.sum() < args.nsub:
        return jmass[good]
    n = args.nsub // nbins
    ic(n)
    # the n most massive
    j = jmass[-n:]
    ic(j, logm[-n:])
    # the 1.1 is to ensure that the last element can be included
    mbins = np.linspace(logmstar_min, 1.1*logm[good].max(), nbins)
    ic(mbins)
    for i in range(len(mbins)):
        ji = (logm > mbins[i-1]) & (logm <= mbins[i])
        ic(i, ji.sum())
        if ji.sum() == 0:
            continue
        ji = np.random.choice(jmass[ji], n, replace=False) if ji.sum() > n \
            else jmass[ji]
        ic(mbins[i-1:i+1], np.log10(Mstar)[ji])
        ic(ji)
        j = np.append(j, ji)
    ic(j)
    ic(np.sort(np.log10(Mstar[j])))
    ic(j.size)
    return j


main()
