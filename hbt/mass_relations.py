from astropy import units as u
from astropy.io import ascii
from astropy.table import Table
from icecream import ic
import matplotlib as mpl
from matplotlib import colors as mplcolors, pyplot as plt, ticker
import multiprocessing as mp
import numpy as np
import os
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import (
    binned_statistic as binstat, binned_statistic_dd as binstat_dd)
import sys
from time import time
from tqdm import tqdm
if sys.version_info[0] == 2:
    range = xrange

import lnr
import plottery
from plottery.plotutils import colorscale, savefig, update_rcParams
update_rcParams()

from HBTReader import HBTReader
from hbtpy import hbt_tools
from hbtpy.hbt_tools import save_plot, timer
from hbtpy.simulation import Simulation
from hbtpy.subhalo import HostHalos, Subhalos
from hbtpy.track import Track


ccolor = 'C9'
scolor = 'C1'
Msun = r'\mathrm{{M}}_\odot'


def main():

    args = hbt_tools.parse_args()
    sim = Simulation(args.simulation)

    to = time()
    reader = HBTReader(sim.path)
    print('Loaded reader in {0:.1f} seconds'.format(time()-to))

    isnap = -1
    subs = reader.LoadSubhalos(isnap)
    print('Loaded subhalos!')

    wrap_plot(reader, sim, subs, isnap)
    return

    return


def wrap_plot(reader, sim, subs, isnap, hostmass='logM200Mean',
              logM200Mean_min=13, logMmin=8, debug=True,
              do_plot_relations=True, do_plot_massfunctions=False):

    subs = Subhalos(
        subs, sim, isnap, exclude_non_FoF=True, logMmin=logMmin,
        logM200Mean_min=logM200Mean_min)
    ic(np.sort(subs.colnames))
    ic(np.unique(subs['HostHaloId']).size)
    ic(np.log10(subs['M200Mean'].min()))
    ic(np.log10(subs['M200Mean'].max()))
    print('{0} objects'.format(subs[subs.colnames[0]].size))
    print()

    plot_path = f'{hostmass}_{logM200Mean_min:.1f}-logM_{logMmin}'
    plot_path = plot_path.replace('.', 'p')

    # fraction of dark satellites - less than 1% througout
    dark_fraction(sim, subs, plot_path=plot_path)
    dark_fraction(sim, subs, mhost_norm=False, plot_path=plot_path)

    # dark matter mass fraction as a function of time-since-infall
    #plot_dmfrac(reader, sim, subs, 'tinfall')

    # plot phase-space
    #plot_rv(sim, subs)
    #return
    #test_velocities(subs)

    massnames = ['Mbound', 'Mstar', 'Mdm', 'Mgas', 'Mass', 'M200', 'MVir']
    units = {'mass': r'$\mathrm{M}_\odot$', 'time': 'Gyr',
             'distance': '$h^{-1}$Mpc'}
    xbins = {
        'ComovingMostBoundDistance': np.logspace(-2, 0.5, 9),
        #'logComovingMostBoundDistance': np.logspace(-2, 0.5, 9),
        'M200Mean': np.logspace(13, 14.7, 9),
        'mu': np.logspace(-5, 0, 9),
        'Mstar': np.logspace(7.7, 12, 11),
        }
    binlabel = {
        #'ComovingMostBoundDistance': '$R_\mathrm{com}$ ($h^{-1}$Mpc)',
        'ComovingMostBoundDistance': '$R$',
        'M200Mean': r'$M_\mathrm{200m}$',
        'mu': r'$\mu$',
        'Mbound': r'$m_\mathrm{sub}$',
        'Mstar': r'$m_{\!\star}$',
        'Mdm': r'$m_\mathrm{DM}$',
        'Mgas': r'$m_\mathrm{gas}$',
        'LastMaxMass': '$m_\mathrm{sub,max}$'
    }
    xx = 'ComovingMostBoundDistance'
    for i in '012':
        xbins[f'{xx}{i}'] = xbins[xx]
        binlabel[f'{xx}{i}'] = f'{binlabel[xx]}_{i}'
    for event in ('last_infall', 'first_infall', 'cent', 'sat'):
        h = f'history:{event}'
        event_label = event.split('_')[0]
        if 'infall' in event:
            infall_label = fr'\mathrm{{i({event_label})}}'
        else:
            infall_label = f'\mathrm{{{event_label}}}'
        binlabel[f'{h}:Mbound'] = rf'$m_\mathrm{{sub}}^{infall_label}$'
        binlabel[f'{h}:Mstar'] = rf'$m_\star^{infall_label}$'
        binlabel[f'{h}:Mdm'] = fr'$m_\mathrm{{DM}}^{infall_label}$'
        binlabel[f'{h}:Mgas'] = rf'$m_\mathrm{{gas}}^{infall_label}$'
        binlabel[f'{h}:time'] = rf'$t_\mathrm{{lookback}}^{infall_label}$'
    axlabel = {}
    for key, label in binlabel.items():
        ismass = np.any([mn in key for mn in massnames])
        if ismass or 'time' in key:
            label = label.replace('$', '')
            unit_key = 'mass' if ismass else (
                'time' if 'time' in key else 'distance')
            un = units[unit_key].replace('$', '')
            axlabel[key] = rf'${label}$ $({un})$'
        else:
            axlabel[key] = label

    def get_axlabel(col, statistic):
        if col in axlabel:
            label = axlabel[col]
        else:
            label = col
        if statistic == 'mean':
            return label
        label = label.replace('$', '')
        label = label.split()
        if len(label) == 2:
            unit = label[1]
        label = label[0]
        if statistic == 'std':
            lab = fr'$\sigma({label})$'
            if len(label) == 2:
                lab = fr'{lab} ${unit}$'
        elif statistic == 'std/mean':
            lab = fr'$\sigma({label})/\langle {label} \rangle$'
        return lab

    def get_bins(n=8):
        if bincol in xbins:
            bins = xbins[bincol]
        elif '/' in bincol:
            bins = np.logspace(-3, 0.3, n+1)
        else: bins = n+1
        return bins

    def get_label_bincol():
        label = f'{bincol}'
        for key, val in binlabel.items():
            if key[:7] == 'history':
                label = label.replace(key, val)
        # now for non-history quantities
        for key, val in binlabel.items():
            label = label.replace(key, val)
        ic(label)
        if '/' not in bincol:
            if np.any([i in bincol for i in massnames]):
                unit = units['mass']
            elif bincol[-4:] == 'time':
                unit = units['time']
            elif 'Distance' in bincol:
                unit = units['distance']
            label = rf'{label} ({unit})'
        return label

    if do_plot_relations:
        # x-axis: cluster-centric distance
        #wrap_relation_distance(sim, subs
        xscale = 'log'
        #for event in ('last_infall', 'first_infall', 'cent', 'sat'):
        for event in []:
            for bincol in ('Mstar', f'{h}:Mstar', f'{h}:Mbound', f'{h}:time'):
                bins = get_bins()
                logbins = ('time' not in bincol)
                xb = np.logspace(-2, 0.5, 9) if xscale == 'log' \
                    else np.linspace(0, 3, 9)
                for statistic in ('std', 'mean'):
                    label = get_label_bincol()
                    showsat = (statistic == 'mean')
                    for ycol in ('Mbound/Mstar', 'Mstar/Mbound',
                                 f'Mstar/{h}:Mbound', f'Mbound/{h}:Mbound',
                                 f'Mstar/{h}:Mstar', f'Mbound/{h}:Mstar'):
                        for xyz in ('', 0, 1, 2):
                            xcol = f'ComovingMostBoundDistance{xyz}'
                            plot_relation(
                                sim, subs, xcol=xcol, ycol=ycol, xbins=xb,
                                selection='Mstar', selection_min=1e8,
                                statistic=statistic, xscale=xscale, bincol=bincol,
                                binlabel=label, bins=bins, logbins=logbins,
                                xlabel=axlabel[xcol],
                                ylabel=get_axlabel(ycol, statistic),
                                show_satellites=True, show_centrals=False)

        relation_kwargs = dict(
            satellites_label='Present-day satellites',
            centrals_label='Present-day centrals',
            min_hostmass=logM200Mean_min, xlim=(3e7,1e12))

        # historical HSMR binned by present-day quantities
        #for event in ('last_infall', 'first_infall', 'cent', 'sat'):
        for event in []:
            h = f'history:{event}'
            bincols = ('Mbound', 'Mstar', 'Mstar/Mbound', f'{h}:time',
                       'ComovingMostBoundDistance','ComovingMostBoundDistance0',
                       'LastMaxMass', 'Mbound/LastMaxMass', 'Mstar/LastMaxMass')
            for bincol in bincols:
                bins = get_bins()
                logbins = ('time' not in bincol)
                for statistic in ('mean', 'std'):
                    label = get_label_bincol()
                    plot_relation(
                        sim, subs, xcol=f'{h}:Mstar', ycol=f'{h}:Mbound',
                        statistic=statistic,
                        bincol=bincol, binlabel=label, bins=bins, logbins=logbins,
                        xlabel=axlabel[f'{h}:Mstar'],
                        ylabel=get_axlabel(f'{h}:Mbound', statistic),
                        show_satellites=True, show_centrals=False, **relation_kwargs)

        # present-day HSMR binned by historical quantities
        relation_kwargs['satellites_label'] = 'All satellites'
        relation_kwargs['centrals_label'] = 'All centrals'
        #relation_kwargs['xbins'] = np.logspace(8, 12, 11)
        for event in ('first_infall', 'last_infall', 'cent', 'sat'):
            h = f'history:{event}'
            bincols = (f'{h}:Mbound', f'{h}:Mstar', f'{h}:Mdm',
                       f'{h}:Mstar/{h}:Mbound', f'{h}:Mdm/{h}:Mbound',
                       f'{h}:Mstar/{h}:Mdm', f'Mbound/{h}:Mbound',
                       f'Mstar/{h}:Mbound', f'{h}:time',
                       'ComovingMostBoundDistance','ComovingMostBoundDistance0',
                       'LastMaxMass', 'Mbound/LastMaxMass', 'Mstar/LastMaxMass')
            bincols = (f'{h}:time',)
            for bincol in bincols:
                if bincol in xbins:
                    bins = xbins[bincol]
                elif '/' in bincol:
                    bins = np.logspace(-3, 0.3, 8)
                else: bins = 10
                logbins = ('time' not in bincol)
                for statistic in ('std/mean',):#'mean', 'std'):
                    label = get_label_bincol()
                    yscale = 'linear' if statistic == 'std/mean' else 'log'
                    ylim = (0, 1) if statistic == 'std/mean' else (5e8,2e14)
                    plot_relation(
                        sim, subs, bincol=bincol, binlabel=label, bins=bins,
                        statistic=statistic, logbins=logbins, yscale=yscale,
                        xlabel=axlabel['Mstar'],
                        ylabel=get_axlabel('Mbound', statistic), ylim=ylim,
                        show_satellites=True, show_centrals=False,
                        **relation_kwargs)

    ## mass functions
    if do_plot_massfunctions:
        for norm in (True, False):
            plot_massfunction2(
                sim, subs, 'M200Mean', xbins['M200Mean'],
                r'$M_\mathrm{{200m}}$ ($h^{{-1}}{0}$)'.format(Msun),
                norm=norm)
            bincol = 'ComovingMostBoundDistance'
            for log in (True, False):
                pref = 'log' * log
                rbins = xbins[f'{pref}{bincol}']
                plot_massfunction2(
                    sim, subs, bincol, bins=rbins, binlabel=binlabel[bincol],
                    norm=norm, bin_in_log10=log)

    return

    plot_occupation(sim, subs)
    mass_sigma_host(sim, subs)
    return

    if norm:
        print('xyz =', xyz.shape, subhalos['R200MeanComoving'].shape)
        xyz = xyz / np.array(subhalos['R200MeanComoving'])
        vhostkey = 'Physical{0}HostVelocityDispersion'.format(vref)
        v = v / np.array(subhalos[vhostkey])
    #for ax, projection in zip(projections):

    return

def get_relation_kwargs(xcol, ycol, bincol=None, statistic='mean'):
    yscale = 'linear' if statistic == 'std/mean' else 'log'
    if statistic == 'std/mean':
        ylim = (0, 1)
    elif ycol in massnames:
        ylim = (5e8, 5e14)
    return dict(ylim=ylim, yscale=yscale)


def dark_fraction(sim, subs, mhost_norm=True, plot_path=None):
    if mhost_norm:
        mbins = np.logspace(-4, -0.3, 21)
    else:
        mbins = massbins(sim, mtype='total')
    mx = (mbins[:-1]+mbins[1:]) / 2
    m = definitions(subs, as_dict=True)
    dark = np.array(m['dark'], dtype=np.uint8)
    x = m['mtot']/m['mhost'] if mhost_norm else m['mtot']
    Ndark = np.histogram(x, bins=mbins, weights=dark)[0]
    Ngal = np.histogram(x, bins=mbins, weights=1-dark)[0]
    ndark = Ndark / np.histogram(x, bins=mbins)[0]
    fig, ax = plt.subplots(figsize=(7,6))
    ax.plot(np.log10(mx), ndark)
    if mhost_norm:
        xlabel = r'$\log\mu\equiv\log m_\mathrm{sub}/M_\mathrm{host}$'
    else:
        xlabel = r'log $m_\mathrm{sub}/$M$_\odot$'
    ax.set(xlabel=xlabel, ylabel='Fraction of dark satellites')
    output = 'dark_fraction'
    if mhost_norm:
        output += '_mhost'
    if plot_path:
        output = os.path.join(plot_path, output)
    save_plot(fig, output, sim)
    return


def mass_sigma_host(sim, subs):
    # mass bins (defined this way so that central values are
    # equally spaced in log space)
    mbins = np.linspace(10, 15, 20)
    m = 10**((mbins[:-1]+mbins[1:]) / 2)
    mbins = 10**mbins
    fig, axes = plt.subplots(figsize=(15,5), ncols=2)
    #fig = plt.figure(figsize=(12,5))
    #axes = [plt.subplot2grid((1,15), (0,0), colspan=7),
            #plt.subplot2grid((1,15), (0,7), colspan=8)]
    h = (subs.catalog['Rank'] == 0) & (subs.catalog['Nsat'] > 20)
    print('Using {0} halos'.format(h.sum()))
    ax = axes[0]
    sigma3d = subs.sigma()[h]
    mtot = subs.mass('total')[h]
    nsat = subs.catalog['Nsat'][h]
    sigma_mean = binstat(mtot, sigma3d, 'mean', mbins)[0]
    sigma_median = binstat(mtot, sigma3d, 'median', mbins)[0]
    c = ax.scatter(
        mtot, sigma3d, marker=',', s=1, c=np.log10(nsat), cmap='viridis',
        label='_none_')
    plt.colorbar(c, ax=ax, label=r'log $N_\mathrm{sat}$')
    ax.plot(m, sigma_mean, 'C0-', label='Mean')
    ax.plot(m, sigma_median, 'C1--', label='Median')
    ax.set_ylabel('${0}$ (km/s)'.format(subs.slabel()))
    ax.legend()
    ax = axes[1]
    c = ax.scatter(
        mtot, subs.sigma(0)[h], marker=',', s=2, c=np.log10(nsat),
        cmap='viridis')
    plt.colorbar(c, ax=ax, label=r'log $N_\mathrm{sat}$')
    ax.plot(m, sigma_mean/3**0.5, 'C0-')
    ax.plot(m, sigma_median/3**0.5, 'C1--')
    # by hand for now, while isnap=-1 in eagle (shouldn't be too
    # different in other sims)
    ax.plot(m, munari13(m, sim.cosmology, z=0), 'C2-', lw=3,
            label='Munari+13')
    ax.set_ylabel('${0}$ (km/s)'.format(subs.slabel(0)))
    for ax in axes:
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('${0}$ (M$_\odot$)'.format(
            sim.masslabel(mtype='total')))
    output = 'mass_sigma'
    out = os.path.join(sim.plot_path, '{0}.pdf'.format(output))
    savefig(out, fig=fig)
    return


#@timer
def plot_dmfrac(reader, sim, subs, xparam, bins=None, hostmass='M200Mean',
                ncores=1):
    """Plot the dark matter fraction as a function of a few things

    """
    #cen, sat, mtot, mstar, mhost, dark, Nsat, Ndark, Ngsat = \
        #definitions(subs, hostmass=hostmass)
    mdm = subs.mass('dm')
    mtot = subs.mass('total')
    dmfrac = mdm / mtot
    # plot
    #fig, ax = plt.subplots(figsize=(7,6))

    return infall


def plot_massfunction2(
        sim, subs, binparam, bins=None, binlabel='', bin_in_log10=True,
        hostmass='M200Mean', overwrite=True, norm=False, show_centrals=False,
        mbins=np.logspace(8,15,21), mubins=np.logspace(-5, 0, 21),
        show_all=False, show_slope=False):
    """Plot subhalo mass function, both as a function of msub and mu

    TO-DO:
        * I still need to store the data in tables
        * Allow calculation of ratios of columns e.g., by passing two
          names in ``binparam``, perhaps with another kwarg whose
          value is the operation (e.g., 'div', 'sum', 'mean', etc)
    """
    print('Plotting mass function with {0}'.format(binparam), end=' ')
    if bin_in_log10:
        print('(log10)')
    else:
        print()
    # we will make one plot each
    names = ('msub', 'mu')
    suff = 'log{0}'.format(binparam) if bin_in_log10 else binparam
    if norm:
        suff += '_norm'
    outputs = ['n{0}_{1}'.format(name, suff) for name in names]
    cen, sat, mtot, mstar, mhost, dark, Nsat, Ndark, Ngsat = \
        definitions(subs, hostmass=hostmass)
    gcen = cen & ~dark
    gsat = sat & ~dark
    fig1, ax1 = plt.subplots(figsize=(8,6))
    fig2, ax2 = plt.subplots(figsize=(8,6))
    figs = [fig1, fig2]
    axes = [ax1, ax2]
    for ax in axes:
        ax.set_yscale('log')
    ax1.set_xlabel(r'log $m_\mathrm{{sub}}/{0}$'.format(Msun))
    # ylabel = r'$N(m_\mathrm{sub})$'
    # if norm:
    #     ylabel = ylabel[:-1] + r'/m_\mathrm{sub}$'
    ylabel = r'$n(m_\mathrm{sub})$' if norm else r'$N(\mathrm{sub})$'
    ax1.set_ylabel(ylabel)
    ax2.set_xlabel(
        r'log $\mu$ $\equiv$ log $(m_\mathrm{sub}/M_\mathrm{200m})$')
    nlabel = r'$n(\mu)$' if norm else r'$N(\mu)$'
    ax2.set_ylabel(nlabel)
    # central bin values for plots (msub and mu; it is assumed that
    # they are binned in log space)
    logm = np.log10(logcenters(mbins))
    logmu = np.log10(logcenters(mubins))
    binvals = subs.catalog[binparam]
    hfig, hax = plt.subplots()
    hquants = [binvals, subs.centrals[binparam], subs.satellites[binparam]]
    if bin_in_log10:
        hquants = [np.log10(i) for i in hquants]
    #hbins = np.linspace(0, 100, 201)
    hbins = 100
    if not bin_in_log10:
        hax.hist(hquants[0], hbins, log=True, label='all')
        hax.hist(hquants[1], hbins, log=True, histtype='step',
                 lw=3, label='centrals')
    hax.hist(hquants[2], hbins, log=True, histtype='step',
             lw=3, label='satellites')
    hax.legend()
    hax.set_xlabel(binlabel)
    save_plot(hfig, 'hist/{0}{1}'.format(binparam, '_log'*bin_in_log10), sim)
    print('binvals =', binvals[sat].min(), binvals[sat].max(),
        np.percentile(binvals[sat], [1,25,50,75,99]))
    print('bins =', bins)
    h = np.histogram(binvals[sat], bins)[0]
    print('hist =', h, h.sum(), sat.sum())
    print('R>60:')
    cols = ['TrackId','HostHaloId','Rank','Nbound',binparam]
    if 'Distance' in binparam or 'Velocity' in binparam:
        dcols = [binparam+i for i in '012']
        cols = cols + dcols
    #
    xbins = np.log10(logcenters(bins)) if bin_in_log10 \
        else (bins[:-1]+bins[1:]) / 2
    # normalization (not used right now)
    nij = 1
    fn = lambda x, a, b, x0=10: a + b*(x-x0)
    fitmasks = ((logm >= 9) & (logm <= 12),
                (logmu >= -4) & (logmu <= -1))
    for fig, ax, x, b, xb, mask, name, output, legloc \
            in zip(figs, axes, (mtot,mtot/mhost), (mbins,mubins),
                   (logm,logmu), fitmasks, names,
                   outputs, ('upper right','upper right')):
        logxo = np.log10(logcenters(b))
        if show_centrals:
            # centrals
            nc = np.histogram(x[cen], b)[0]
            ngc = np.histogram(x[gcen], b)[0]
            n0 = nc.sum() if norm else 1
            ax.plot(logxo[nc > 0], nc[nc > 0]/n0, '--', color=ccolor, lw=4,
                    label='Central subhaloes')
            ax.plot(logxo[ngc > 0], ngc[ngc > 0]/n0, '-', color=ccolor, lw=4,
                    label='Central galaxies')
        # satellites
        ns = np.histogram(x[sat], b)[0]
        ngs = np.histogram(x[gsat], b)[0]
        nfit, ncov = curve_fit(fn, xb[mask], np.log10(ns[mask]), p0=(-2,-1))
        ic(nfit)
        ic(np.diag(ncov)**0.5)
        if show_all:
            n0 = ns.sum() if norm else 1
            ax.plot(logxo[ns > 0], ns[ns > 0]/n0, '--', color=scolor, lw=4,
                    zorder=100, label='All satellite subhaloes')
            ax.plot(logxo[ngs > 0], ngs[ngs > 0]/n0, '-', color=scolor, lw=4,
                    zorder=100, label='All satellite galaxies')
        else:
            ax.plot([], [], '--', color='k', lw=3,
                    label='Satellite subhaloes')
            ax.plot([], [], '-', color='k', lw=3,
                    label='Satellite galaxies')
        ns, xe, ye = np.histogram2d(binvals[sat], x[sat], (bins,b))
        ngs = np.histogram2d(binvals[gsat], x[gsat], (xe,ye))[0]
        if bin_in_log10:
            colors, cmap = colorscale(array=10**xbins, log=True)
        else:
            colors, cmap = colorscale(array=xbins, log=True)
        for i in range(ns.shape[0]):
            j = (ns[i] > 0)
            if j.sum() == 0:
                continue
            n0 = ns[i].sum() if norm else 1
            ax.plot(logxo[j], ns[i][j]/n0, '--', color=colors[i], lw=4,
                    zorder=20-i)
            ax.plot(logxo[j], ngs[i][j]/n0, '-', color=colors[i], lw=4,
                    zorder=20-i)
        cbar = plt.colorbar(cmap, ax=ax)
        cbar.set_label(binlabel)
        if bin_in_log10:
            cbar.ax.set_yscale('log')
            if binparam == 'ComovingMostBoundDistance':
                cbar.ax.yaxis.set_major_formatter(
                    ticker.FormatStrFormatter('%.1f'))
        ax.legend(loc=legloc, ncol=1, fontsize=14)
        # to give the legend some space
        # ylim = ax.get_ylim()
        # ax.set_ylim(ylim[0], 10*ylim[1])
        save_plot(fig, output, sim)
    return


def plot_massfunction(sim, subs, hostmass='M200Mean'):
    cen, sat, mtot, mstar, mhost, dark, Nsat, Ndark, Ngsat = \
        definitions(subs, hostmass=hostmass)
    gcen = cen & ~dark
    gsat = sat & ~dark
    # testing
    fig1, ax1 = plt.subplots(figsize=(9,6))
    fig2, ax2 = plt.subplots(figsize=(9,6))
    figs = [fig1, fig2]
    axes = [ax1, ax2]
    for ax in axes:
        ax.set_yscale('log')
    ax1.set_xlabel(r'log $m_\mathrm{{sub}}/{0}$'.format(Msun))
    ax1.set_ylabel(r'$N(m_\mathrm{sub})$')
    ax2.set_xlabel(
        r'log $\mu$ $\equiv$ log $(m_\mathrm{sub}/M_\mathrm{200m})$')
    ax2.set_ylabel(r'$N(\mu)$')
    mbins = np.logspace(8, 15, (15-8)//0.2)
    logm = np.log10(logcenters(mbins))
    mubins = np.logspace(-5, 0, 5//0.2)
    mhbins = np.logspace(10, 14.5, 9)
    logmh = np.log10(logcenters(mhbins))
    # normalization?
    nij = 1
    for ax, x, bins, name \
            in zip(axes, (mtot,mtot/mhost), (mbins,mubins), ('msub','mu')):
        xo = logcenters(bins)
        #logxo = np.log10(xo)
        logxo = logm
        # normalization of y axis. Just keeping the name from below
        #if name == 'mu':
            #nij = 1 / xo
        # all subhalos
        # centrals
        c = np.histogram(x[cen], bins)[0]
        gc = np.histogram(x[gcen], bins)[0]
        ax.plot(logxo[c > 0], (nij*c)[c > 0], '--', color=ccolor, lw=4,
                label='Central subhaloes')
        ax.plot(logxo[gc > 0], (nij*gc)[gc > 0], '-', color=ccolor, lw=4,
                label='Central galaxies')
        # satellites
        s = np.histogram(x[sat], bins)[0]
        gs = np.histogram(x[gsat], bins)[0]
        ax.plot(logxo[s > 0], (nij*s)[s > 0], '--', color=scolor, lw=4,
                label='Satellite subhaloes')
        ax.plot(logxo[gs > 0], (nij*gs)[gs > 0], '-', color=scolor, lw=4,
                label='Satellite galaxies')
        xcolname = 'log{0}'.format(name)
        nm = Table({xcolname: logxo, 'csub': c, 'cgal': gc,
                    'ssub': s, 'sgal': gs})
        nm[xcolname].format = '%.2f'
        ascii.write(nm, os.path.join(sim.data_path, 'n{0}.txt'.format(name)),
                    format='fixed_width', overwrite=True)
        # split by host mass
        s = np.histogram2d(mhost[sat], x[sat], (mhbins,bins))[0]
        gs = np.histogram2d(mhost[gsat], x[gsat], (mhbins,bins))[0]
        for sample, sname in zip((s, gs), ('subhaloes', 'galaxies')):
            ns = Table({'{0:.2f}'.format(mhi): si
                        for mhi, si in zip(logmh, sample)})
            ns[xcolname] = nm[xcolname]
            output = os.path.join(
                sim.data_path, 'n{0}_{1}.txt'.format(name, sname))
            ascii.write(ns, output, format='fixed_width', overwrite=True)
        colors, cmap = colorscale(array=logmh)
        for i in range(s.shape[0]):
            j = (s[i] > 0)
            if j.sum() == 0:
                continue
            #if name == 'mu':
                #nij = 1 / xo[j]
            ax.plot(logxo[j], nij*s[i][j], '--', color=colors[i], lw=2,
                    zorder=20-i)
            ax.plot(logxo[j], nij*gs[i][j], '-', color=colors[i], lw=2,
                    zorder=20-i)
        cbar = plt.colorbar(cmap, ax=ax)
        cbar.set_label(r'log $M_\mathrm{{host}}/{0}$'.format(Msun))
        ax.legend(loc='upper right', ncol=1, fontsize=16)
    for fig, name in zip(figs, ('msub','mu')):
        save_plot(fig, 'n{0}'.format(name), sim)
    return


def plot_rv(sim, subs, hostmass='M200Mean', weights=None):
    cen, sat, mtot, mstar, mhost, dark, Nsat, Ndark, Ngsat = \
        definitions(subs, hostmass=hostmass)
    # use subs attributes for distance and velocity
    testID = subs.catalog['HostHaloId'].iloc[10000]
    j = (subs.catalog['HostHaloId'] == testID)
    print('testID =', testID, j.sum())
    print('calculating r and v')
    r = (subs.catalog['ComovingMostBoundDistance'] \
         / subs.catalog['R200MeanComoving'])[sat]
    v = (subs.catalog['PhysicalMostBoundPeculiarVelocity'] \
         / subs.catalog['PhysicalMostBoundHostVelocityDispersion'])[sat]
    rbins = np.arange(0, 3.01, 0.05)
    vbins = np.arange(-5, 5.01, 0.1)
    print('Calculating histograms')
    hist2d, xe, ye = np.histogram2d(r, v, (rbins,vbins))
    #hist2d[hist2d == 0] = np.nan
    #hist2d.T
    print('hist2d =', hist2d.shape)
    #return
    extent = (xe[0], xe[-1], ye[0], ye[-1])
    print('Making plot')
    fig, ax = plt.subplots(figsize=(9,6))
    im = ax.imshow(hist2d.T, origin='lower', extent=extent, aspect='auto',
                   norm=mpl.colors.LogNorm())
    plt.colorbar(im, ax=ax, label=r'$N_\mathrm{sat}$')
    ax.set_xlabel(r'$')
    ax.set_ylabel(r'$v_\mathrm{pec}/\sigma_{v,\mathrm{host}}$')
    output = os.path.join(sim.plot_path, 'phase-space', 'rv_3d.pdf')
    print('Saving plot')
    savefig(output, fig=fig)
    print('Finished!')
    return


def plot_relation(sim, subs, xcol='Mstar', ycol='Mbound',
                  statistic='mean', selection=None,
                  selection_min=None, selection_max=None, xlim=None,
                  ylim=None, xbins=10, xscale='log', yscale='log',
                  hostmass='M200Mean', min_hostmass=13, show_hist=True,
                  bindata=None, bincol=None, bins=8, logbins=False,
                  binlabel='', mask=None, xlabel=None, ylabel=None,
                  show_satellites=True, show_centrals=True,
                  satellites_label='All satellites',
                  centrals_label='Centrals'):
    """Plot the SHMR and HSMR

    ``bincol`` and ``bins`` allow the relations to be binned in a
    third quantity

    """
    cen, sat, mtot, mstar, mhost, dark, Nsat, Ndark, Ngsat = \
        definitions(subs, hostmass=hostmass, min_hostmass=min_hostmass)
    xdata = get_data(subs, xcol)
    ydata = get_data(subs, ycol)
    if bincol is not None:
        bindata = get_data(subs, bincol)
        if not np.iterable(bins):
            j = np.isfinite(bindata)
            if logbins:
                j = j & (bindata > 0)
                bins = np.logspace(
                    np.log10(bindata[j].min()), np.log10(bindata[j].max()),
                    bins+1)
            else:
                bins = np.linspace(bindata[j].min(), bindata[j].max(), bins+1)
        ic(bins)
        ic(np.histogram(bindata, bins)[0])
        colors, cmap = colorscale(array=bins, log=logbins)
        if not binlabel:
            binlabel = f'{statistic}({bincol})'
    mask = np.isfinite(xdata) & np.isfinite(ydata) & np.isfinite(bindata)
    if selection is not None:
        seldata = get_data(subs, selection)
        if selection_min is not None:
            mask = mask & (seldata >= selection_min)
        if selection_max is not None:
            mask = mask & (seldata <= selection_max)
    ic(mask.sum())
    xdata = xdata[mask]
    ydata = ydata[mask]
    bindata = bindata[mask]
    sat = sat[mask]
    dark = dark[mask]
    cen = cen[mask]
    gsat = sat & ~dark
    ic(xbins)
    if isinstance(xbins, int):
        if xscale == 'log':
            gsat = gsat & (xdata > 0)
            xbins = np.linspace(
                np.log10(xdata[gsat].min()), np.log10(xdata[gsat].max()),
                xbins+1)
        else:
            xbins = np.linspace(xdata[gsat].min(), xdata[gsat].max(), xbins+1)
        if xscale == 'log':
            xbins = 10**xbins
    if xscale == 'log':
        xb = np.log10(xbins)
        xcenters = 10**((xb[:-1]+xb[1:])/2)
    else:
        xcenters = (xbins[:-1]+xbins[1:]) / 2
    ic(xcol, xbins)
    ic(xdata.min(), xdata.max())
    ic(np.histogram(xdata, xbins)[0])
    ic(xcenters)
    ic(xlim, ylim)
    logx = np.log10(xcenters)

    lw = 4
    # as a function of third variable
    fig, ax = plt.subplots(figsize=(8,6))
    # fix bins here
    if '/' in statistic:
        stat = statistic.split('/')
        relation = binstat_dd(
                [bindata[gsat], xdata[gsat]], ydata[gsat], stat[0],
                [bins,xbins]).statistic \
            / binstat_dd(
                [bindata[gsat], xdata[gsat]], ydata[gsat], stat[1],
                [bins,xbins]).statistic
    else:
        relation = binstat_dd(
            [bindata[gsat], xdata[gsat]], ydata[gsat], statistic,
            [bins,xbins]).statistic
    # this will make the plot look nicer
    if 'std' in statistic:
        relation[relation == 0] = np.nan
    ic(relation)
    ic(relation.shape)
    for i in range(bins.size-1):
        ax.plot(xcenters, relation[i], '-', color=colors[i],
                lw=4, zorder=2*bins.size-i)
    cbar = plt.colorbar(cmap, ax=ax)
    cbar.set_label(binlabel)
    if logbins:
        cbar.ax.set_yscale('log')
        if bins[0] >= 0.001 and bins[-1] <= 1000:
            cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%s'))
    # compare to overall and last touches
    if show_satellites and '/' not in statistic:
        j = mask & gsat
        satrel = binstat(mstar[j], mtot[j], statistic, xbins)[0]
        plot_line(
            ax, xcenters, satrel, marker='+', lw=3, color=scolor,
            label=satellites_label, zorder=100)
    if show_centrals and '/' not in statistic:
        j = mask & cen
        cenrel = binstat(mstar[j], mtot[j], statistic, xbins)[0]
        plot_line(
            ax, xcenters, cenrel, marker='+', lw=3, color=ccolor,
            label=centrals_label, zorder=100)
    if show_satellites or show_centrals:
        ax.legend(fontsize=18)
    if xlabel is None:
        xlabel = r'$\log\,{0}$'.format(sim.masslabel(mtype='stars'))
    if ylabel is None:
        ylabel = r'$\log\,{0}$'.format(sim.masslabel(mtype='total'))
    ax.set(xlabel=xlabel, ylabel=ylabel, xscale=xscale, yscale=yscale,
           xlim=xlim, ylim=ylim)
    if 'Distance' in xcol and xscale == 'log':
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%s'))
    # format filename
    bincol = bincol.replace('/', '-over-').replace(':', '-')
    outname = f'{statistic}__{xcol}_{ycol}'.replace('/', '-over-')
    output = f'relations/{outname}/{outname}__bin__{bincol}'.replace(':', '-')
    save_plot(fig, output, sim)
    return relation


def plot_occupation(sim, subs, hostmass='M200Mean'):
    cen, sat, mtot, mstar, mhost, dark, Nsat, Ndark, Ngsat = definitions(subs)
    # binning and plot
    mbins = np.logspace(8, 15, 41)
    m = logcenters(mbins)
    msbins = np.logspace(5.5, 12.5, 26)
    ms = logcenters(msbins)
    mlim = np.log10(m[::m.size-1])
    mslim = np.log10(ms[::ms.size-1])
    nh = np.histogram(mhost[cen], mbins)[0]
    nc = np.histogram2d(
        mstar[cen & ~dark], mhost[cen & ~dark], (msbins,mbins))[0]
    nsat = np.histogram2d(
        mstar[sat & ~dark], mhost[sat & ~dark], (msbins,mbins))[0]
    nsub = np.histogram2d(
        mstar[sat & ~dark], mtot[sat & ~dark], (msbins,mbins))[0]

    fig, ax = plt.subplots(figsize=(6,5))
    #ax.plot(m, nh_mtot, 'k-', label='Halos')
    #ax.plot(m, np.sum(nc, axis=0), 'C0:', label='Total Centrals')
    #ax.plot(m, np.sum(nsat, axis=0), 'C1:', label='Total Satellites')
    #ax.hist(mhost[cen & ~dark], mbins, histtype='step', alpha=0.5, color='C3')
    nc = nc / nh
    nsat = nsat / nh
    ax.plot(m, np.sum(nc, axis=0), 'C0-', label='Centrals')
    ax.plot(m, np.sum(nsat, axis=0), 'C1-', label='Satellites')
    # just a narrow bin
    mmin = 10
    mmax = 11
    logmsbins = np.log10(msbins)
    jmin = np.argmin((logmsbins-mmin)**2)
    jmax = np.argmin((logmsbins-mmax)**2)
    j = np.arange(jmin, jmax+1, 1, dtype=int)
    ax.plot(m, np.sum(nc[j], axis=0), 'C0--')
    ax.plot(m, np.sum(nsat[j], axis=0), 'C1--')
    ax.plot([], [], 'k--',
            label=r'$\log {2}$ in [{0:.2f},{1:.2f})'.format(
                logmsbins[j][0],logmsbins[j][-1],
                sim.masslabel(mtype='stars')))
    ax.legend(fontsize=10)
    ax.set_xscale('log')
    ax.set_yscale('log')
    mhlabel = r'M_\mathrm{{{0}}}'.format(hostmass[1:5].lower())
    ax.set_xlabel(r'${0}$'.format(mhlabel))
    ax.set_ylabel(r'$\langle N\rangle ({0})$'.format(mhlabel))
    save_plot(fig, 'occupation', sim)
    return


def plot_vpec():
    fig, axes = plt.subplots(figsize=(12,5), ncols=2)
    bins = [np.linspace(0, 3000, 100), np.logspace(-1, 3.5, 100)]
    for ax, b in zip(axes, bins):
        ax.hist(subs.velocity(), b, histtype='step', label='3d', bottom=1)
        for i, x in enumerate('xyz'):
            ax.hist(subs.velocity(i), b, histtype='step', label=x, bottom=1)
        ax.legend(loc='upper right')
        ax.set_xlabel('Peculiar velocity (km/s)')
        ax.set_yscale('log')
    axes[1].set_xscale('log')
    axes[0].set_title('Linear binning')
    axes[1].set_title('Log-scale binning')
    output = 'vpeculiar'
    save_plot(fig, output, sim)



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


def munari13(m, cosmo, z=0):
    hz = (cosmo.H(z) / (100*u.km/u.s/u.Mpc)).value
    return 1177 * (hz*m/1e15)**0.364


def test_velocities(subs):
    _, clusters = np.unique(subs.catalog['HostHaloId'], return_index=True)
    print('clusters =', clusters.min(), clusters.max(), clusters.shape)
    #print('Nbound[clusters] =', subs.catalog['Nbound'][clusters])
    fig, axes = plt.subplots(figsize=(12,5), ncols=2)
    ax = axes[0]
    #for proj in ('xyz', 'x', 'y', 'z'):
        #vi = np.array(subs.catalog['v_cl_{0}'.format(proj)])
    for proj in ('', '1', '2', '3'):
        vi = np.array(
            subs.catalog['Physical{1}HostMeanVelocity{0}'.format(
                proj, vref)])
        print('vi =', vi.min(), vi.max(),
              np.percentile(vi, [1,5,25,50,75,95,99]))
        nanvi = ~np.isfinite(vi)
        print('nan: {0}/{1} ({2:.1f}%)'.format(
            nanvi.sum(), nanvi.size, 100*nanvi.sum()/nanvi.size))
        nanhosts = np.array(subs.catalog['HostHaloId'])[nanvi]
        goodhosts = np.array(subs.catalog['HostHaloId'])[~nanvi]
        vavgcols = ['PhysicalAverageVelocity{0}'.format(i) for i in range(3)]
        vmbcols = ['PhysicalMostBoundVelocity{0}'.format(i) for i in range(3)]
        ncols = ['NboundType{0}'.format(i) for i in range(6)]
        mcols = ['MboundType{0}'.format(i) for i in range(6)]
        print()
        print('in each nanhalo:')
        for hostid in nanhosts[:10]:
            inhalo = np.where(subs.catalog['HostHaloId'] == hostid)[0]
            print('ID:', hostid, '| Nsub:', inhalo.size,
                  '| Nbound > 0:', (subs.catalog['Nbound'][inhalo] > 0).sum(),
                  '| NboundType > 0:',
                  [(subs.catalog[c][inhalo] > 0).sum() for c in ncols],
                  '| v_avg < inf:',
                  [np.isfinite(subs.catalog[c][inhalo]).sum() for c in vavgcols],
                  '| v_bound < inf:',
                  [np.isfinite(subs.catalog[c][inhalo]).sum() for c in vmbcols])
        print('in each good halo:')
        for hostid in goodhosts[:10]:
            inhalo = np.where(subs.catalog['HostHaloId'] == hostid)[0]
            print('ID:', hostid, 'Nsub:', inhalo.size,
                  '| Nbound > 0:', (subs.catalog['Nbound'][inhalo] > 0).sum(),
                  '| NboundType > 0:',
                  [(subs.catalog[c][inhalo] > 0).sum() for c in ncols],
                  '| v_avg < inf:',
                  [np.isfinite(subs.catalog[c][inhalo]).sum() for c in vavgcols],
                  '| v_bound < inf:',
                  [np.isfinite(subs.catalog[c][inhalo]).sum() for c in vmbcols])
        print()
        print('high-v:')
        highv = subs.catalog[vi > 1e8]
        #print(highv[['HostHaloId','Rank']])
        #print(highv['HostHaloId'].size, np.unique(highv['HostHaloId']))
        cols = ['HostHaloId','PhysicalHostMeanVelocity',
                'PhysicalHostVelocityDispersion']
        mult = highv.groupby('HostHaloId')[cols].count()
        #print('mult =', mult)
        ax.hist(vi[clusters], 50,
                histtype='step', label=proj)
    ax.set_xlabel('Cluster velocity')
    ax.set_ylabel(r'$N_\mathrm{cl}$')
    ax = axes[1]
    for i, vi in enumerate(v, 1):
        ax.hist(vi, 100, histtype='step', label=str(i))
    ax.set_xlabel('Subhalo velocity')
    ax.set_ylabel(r'$N_\mathrm{sub}$')
    for ax in axes:
        ax.legend(loc='upper right', fontsize=15)
    out = os.path.join(sim.plot_path, 'hist_velocities.pdf')
    savefig(out, fig=fig)
    return


def view_velocities(subs, sim, vref='MostBound'):
    # note that I just want to look at hosts here
    cat = np.array(subs.catalog['Rank'] == 0)
    vcol = 'Physical{0}HostMeanVelocity'.format(vref)
    scol = 'Physical{0}HostVelocityDispersion'.format(vref)
    vinf = ~np.isfinite(subs.catalog[vcol])[cat]
    sinf = ~np.isfinite(subs.catalog[scol])[cat]
    logm = np.log10(subs.mass('total'))[cat]
    logmstar = np.log10(subs.mass('stars'))[cat]
    logmbins = np.arange(6, 16.01, 0.2)
    # histogram of velocities
    fig, axes = plt.subplots(figsize=(15,6), ncols=2)
    ax = axes[0]
    for ax, m in zip(axes, (logm, logmstar)):
        ax.hist(m, logmbins, histtype='stepfilled', label='all',
                bottom=1, lw=0, alpha=0.5)
        ax.hist(m[vinf], logmbins, histtype='step', bottom=1, lw=2,
                label=r'$\langle v \rangle = \infty$')
        ax.hist(m[sinf], logmbins, histtype='step', bottom=1,
                lw=2, label=r'$\sigma_\mathrm{v} = \infty$')
    axes[0].set_xlabel(r'log $m_\mathrm{sub}$')
    axes[1].set_xlabel(r'log $m_\star$')
    for ax in axes:
        ax.set_yscale('log')
        ylim = ax.get_ylim()
        ax.set_ylim(1, ylim[1])
        ax.set_ylabel('1+N')
        ax.legend()
    out = os.path.join(sim.plot_path, 'vinf.pdf')
    savefig(out, fig=fig)
    return


##
## Auxiliary functions
##


def binning(bins=None, sim=None, mtype='total', n=None, xmin=0, xmax=1,
            log=True):
    if bins is None:
        if sim is None:
            f = np.logspace if log else np.linspace
            bins = f(xmin, xmax, n+1)
        else:
            bins = massbins(sim, mtype=mtype)
            if n is not None:
                bins = np.log10(bins[::bins.size-1])
                bins = np.logspace(bins[0], bins[1], n+1)
    if log:
        x = logcenters(bins)
    else:
        x = (bins[:-1]+bins[1:]) / 2
    xlim = np.log10(x[::x.size-1])
    return bins, x, xlim
    """
    mbins = massbins(sim, mtype='total')
    m = logcenters(mbins)
    msbins = massbins(sim, mtype='stars')
    ms = logcenters(msbins)
    mlim = np.log10(m[::m.size-1])
    mslim = np.log10(ms[::ms.size-1])
    return mbins, m, mlim, msbins, ms, mslim
    """


def definitions(subs, hostmass='M200Mean', min_hostmass=13, as_dict=False):
    """Subhalo quantities

    Parameters
    ----------
    subs : ``Subhalos`` object
        subhalos
    hostmass : str, optional
        host mass definition
    as_dict : bool, optional
        whether to return a tuple (if ``False``) or a dictionary

    Returns either a dictionary or tuple with the following content:
     * cen : centrals mask
     * sat : satellites mask
     * mtot : total subhalo mass
     * mstar : stellar mass
     * mhost : Host mass as defined in the ``hostmass`` argument
     * dark : dark subhalos mask
     * Nsat : number of satellite subhalos associated with the same host
     * Ndark : number of dark satellite subhalos
     * Ngsat : number of satellites galaxies
    """
    if min_hostmass is None:
        min_hostmass = 0
    if min_hostmass < 20:
        min_hostmass = 10**min_hostmass
    mtot = subs.mass('total')
    mstar = subs.mass('stars')
    mhost = subs[hostmass]
    cen = (subs['Rank'] == 0)
    sat = (subs['Rank'] > 0) & (mhost > min_hostmass)
    dark = (subs['IsDark'] == 1)
    Nsat = subs['Nsat']
    Ndark = subs['Ndark']
    Ngsat = Nsat - Ndark
    if as_dict:
        return dict(
            cen=cen, sat=sat, mtot=mtot, mstar=mstar, mhost=mhost, dark=dark,
            Nsat=Nsat, Ndark=Ndark, Ngsat=Ngsat)
    return cen, sat, mtot, mstar, mhost, dark, Nsat, Ndark, Ngsat


def get_data(subs, col):
    if col is None:
        return
    if '/' in col:
        col1, col2 = col.split('/')
        data = subs[col1] / subs[col2]
    else:
        data = subs[col]
    return data


def logcenters(bins):
    logbins = np.log10(bins)
    return 10**((logbins[:-1]+logbins[1:])/2)


def massbins(sim, mtype='stars'):
    if mtype in ('dm', 'bound'):
        mtype = 'total'
    bins = {'apostle': {'stars': np.logspace(3, 11, 21)},
            'eagle': {'stars': np.logspace(6, 12.5, 26),
                      'total': np.logspace(8, 15, 31)}}
    return bins[sim.family][mtype]


def plot_line(ax, *args, marker='+', **kwargs):
    """For now, forcing thick dots; lines not supported"""
    if 'ls' in kwargs:
        kwargs.pop('ls')
    kwargs_bg = kwargs.copy()
    #if dashes is not None:
        #kwargs['dashes'] = dashes
        # the offsets probably depend on the size of the dashes...
        #kwargs_bg['dashes'] = (dashes[0]-2.2,dashes[1]-3.8)
    if 'color' in kwargs:
        kwargs_bg.pop('color')
    if 'label' in kwargs:
        kwargs_bg.pop('label')
    #kwargs_bg['ls'] = ls
    #if 'dashes' in kwargs or ls in ('-', '-.', ':', '--'):
    #    if 'lw' not in kwargs:
    #        kwargs['lw'] = 4
    #    kwargs_bg['lw'] = kwargs['lw'] + 2
    if 'ms' not in kwargs:
        kwargs['ms'] = 8
    if 'mew' not in kwargs:
        kwargs['mew'] = 3
    kwargs_bg['ms'] = kwargs['ms'] + 2
    kwargs_bg['mew'] = kwargs['mew'] + 2
    ax.plot(*args, marker, **kwargs_bg, color='w', label='_none_')
    ax.plot(*args, marker, **kwargs)
    return


def output(sim, xname, yname):
    return '{0}_{1}'.format(sim.masslabel(mtype=xname, latex=False),
                            sim.masslabel(mtype=yname, latex=False))


def plotpath(f):
    """Not in use right now"""
    @functools.wraps
    def wrapper(*args, **kwargs):
        if 'plot_path' not in kwargs:
            pass
            #args = inspect.getfullargspec(
            #kwargs['plot_path'] =

main()
