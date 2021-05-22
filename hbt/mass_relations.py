from astropy import units as u
from astropy.io import ascii
from astropy.table import Table
from icecream import ic
import matplotlib as mpl
from matplotlib import pyplot as plt, ticker
import multiprocessing as mp
import numpy as np
import os
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import (
    binned_statistic as binnedstat, binned_statistic_dd as binnedstat_dd)
import sys
from time import time
from tqdm import tqdm
if sys.version_info[0] == 2:
    range = xrange

import lnr
from plottery.plotutils import colorscale, savefig, update_rcParams
update_rcParams()

from HBTReader import HBTReader
from hbtpy import hbt_tools
from hbtpy.hbt_tools import timer
from hbtpy.simulation import Simulation
from hbtpy.subhalo import HostHalos, Subhalos, Track


ccolor = 'C9'
scolor = 'k'
Msun = r'\mathrm{{M}}_\odot'


def main(debug=True):

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


def wrap_plot(reader, sim, subs, isnap, debug=True):

    subs = Subhalos(subs, sim, isnap, exclude_non_FoF=True)
    subhalos = subs.catalog
    ic(np.sort(subs.colnames))
    print('{0} objects'.format(subs.catalog[subs.colnames[0]].size))
    print()

    # dark matter mass fraction as a function of time-since-infall
    #plot_dmfrac(reader, sim, subs, 'tinfall')

    # plot phase-space
    #plot_rv(sim, subs)
    #return
    #test_velocities(subs)

    xbins = dict(
        'ComovingMostBoundDistance': np.append(
            np.arange(0, 0.51, 0.1), np.arange(1, 3.1, 0.5)),
        )

    cshmr, sshmr, chsmr, shsmr = plot_shmr(sim, subs)
    #plot_shmr_fmass(sim, subs, shsmr=shsmr, chsmr=chsmr)
    plot_shmr_fmass(
        sim, subs, bincol='ComovingMostBoundDistance',
        bins=xbins['ComovingMostBoundDistance'])
    return

    for norm in (True, False):
        plot_massfunction2(
            sim, subs, 'M200Mean', np.logspace(11, 14.7, 9),
            r'log $M_\mathrm{{200m}}$ ($h^{{-1}}{0}$)'.format(Msun),
            norm=norm)
        for log in (True, False):
            rbins = (np.logspace(-1, 0.5, 9) if log else np.linspace(0, 3, 9))
            pref = 'log ' * log
            plot_massfunction2(
                sim, subs, 'ComovingMostBoundDistance', rbins,
                r'{0}$R_\mathrm{{com}}$/($h^{{-1}}$Mpc)'.format(pref),
                norm=norm, bin_in_log10=log)
        #plot_massfunction2(
            #sim, subs, 'PhysicalMostBoundDistance', rbins,
            #r'{0}$R_\mathrm{{phys}}$/($h^{{-1}}$Mpc)'.format(pref),
            #bin_in_log10=log)
    #plot_massfunction2(
        #sim, subs, 'M200Mean', np.logspace(10,14.5,9),
        #r'log $M_\mathrm{{host}}/{0}$'.format(Msun))
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
    sigma_mean = binnedstat(mtot, sigma3d, 'mean', mbins)[0]
    sigma_median = binnedstat(mtot, sigma3d, 'median', mbins)[0]
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
        hostmass='M200Mean', overwrite=True, norm=False,
        mbins=np.logspace(8,15,21), mubins=np.logspace(-5, 0, 21)):
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
    fig1, ax1 = plt.subplots(figsize=(9,6))
    fig2, ax2 = plt.subplots(figsize=(9,6))
    figs = [fig1, fig2]
    axes = [ax1, ax2]
    for ax in axes:
        ax.set_yscale('log')
    ax1.set_xlabel(r'log $m_\mathrm{{sub}}/{0}$'.format(Msun))
    ylabel = r'$N(m_\mathrm{sub})$'
    if norm:
        ylabel = ylabel[:-1] + r'/m_\mathrm{sub}$'
    ax1.set_ylabel(ylabel)
    ax2.set_xlabel(
        r'log $\mu$ $\equiv$ log $(m_\mathrm{sub}/M_\mathrm{200m})$')
    ax2.set_ylabel(r'$N(\mu)$')
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
    print(subs.catalog[cols][subs.catalog['HostHaloId'] == 2377])
    #
    xbins = np.log10(logcenters(bins)) if bin_in_log10 \
        else (bins[:-1]+bins[1:]) / 2
    print('xbins =', xbins)
    # normalization (not used right now)
    nij = 1
    for fig, ax, x, b, name, output, legloc \
            in zip(figs, axes, (mtot,mtot/mhost), (mbins,mubins), names,
                   outputs, ('upper right','upper left')):
        logxo = np.log10(logcenters(b))
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
        n0 = ns.sum() if norm else 1
        ax.plot(logxo[ns > 0], ns[ns > 0]/n0, '--', color=scolor, lw=4,
                label='Satellite subhaloes')
        ax.plot(logxo[ngs > 0], ngs[ngs > 0]/n0, '-', color=scolor, lw=4,
                label='Satellite galaxies')
        ns, xe, ye = np.histogram2d(binvals[sat], x[sat], (bins,b))
        ngs = np.histogram2d(binvals[gsat], x[gsat], (xe,ye))[0]
        colors, cmap = colorscale(array=xbins)
        for i in range(ns.shape[0]):
            j = (ns[i] > 0)
            if j.sum() == 0:
                continue
            n0 = ns[i].sum() if norm else 1
            ax.plot(logxo[j], ns[i][j]/n0, '--', color=colors[i], lw=2,
                    zorder=20-i)
            ax.plot(logxo[j], ngs[i][j]/n0, '-', color=colors[i], lw=2,
                    zorder=20-i)
        cbar = plt.colorbar(cmap, ax=ax)
        cbar.set_label(binlabel)
        ax.legend(loc=legloc, ncol=1, fontsize=14)
        # to give the legend some space
        ylim = ax.get_ylim()
        ax.set_ylim(ylim[0], 10*ylim[1])
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


def plot_shmr_fmass(sim, subs, shsmr=None, chsmr=None, hostmass='M200Mean',
                    show_hist=True, bincol=None, bins=8, mask=None):
    """Plot the SHMR and HSMR

    ``bincol`` and ``bins`` allow the relations to be binned in a
    third quantity

    """
    cen, sat, mtot, mstar, mhost, dark, Nsat, Ndark, Ngsat = \
        definitions(subs, hostmass=hostmass)
    gsat = sat & ~dark
    mbins, m, mlim = binning(sim, mtype='total')
    msbins, ms, mslim = binning(sim, mtype='stars')
    logms = np.log10(ms)
    mubins, xmu, mulim = binning(xmin=-5, xmax=0, n=10, log=True)
    mhbins, mh, mhlim = binning(xmin=10, xmax=14.5, n=9, log=True)
    ic(mhbins.shape)
    ic(mh.shape)
    ic(mhlim.shape)
    if bincol is not None:
        if bincol == 'mu':
            xbin = mtot / mhost
        else:
            xbin = subs.catalog[bincol]
        ic(xbin.min())
        ic(xbin.max())
        if not np.iterable(bins):
            j = np.isfinite(xbin)
            bins = np.linspace(xbin[j].min(), xbin[j].max(), bins+1)
        ic(bins)
        ic(np.histogram(xbin, bins)[0])
        colors, cmap = colorscale(array=bins)
    # completeness limit
    msmin = 7.5
    jms = (ms >= 10**msmin)

    extent = [np.append(mlim, mslim), np.append(mslim, mlim)]
    lw = 4
    # as a function of third variable
    fig, ax = plt.subplots(figsize=(8,6))
    # fix bins here
    hsmr_binned = binnedstat_dd(
        [xbin[gsat], mstar[gsat]], mtot[gsat], 'mean', [bins,msbins])
    hsmr_binned = hsmr_binned.statistic
    ic(hsmr_binned.shape)
    for i in range(bins.size-1):
        ax.plot(logms[jms], np.log10(hsmr_binned[i,jms]), '-', color=colors[i],
                lw=4, zorder=2*bins.size-i)
    cbar = plt.colorbar(cmap, ax=ax)
    cbar.set_label(bincol)
    # compare to overall and last touches
    if shsmr is not None:
        plot_line(
            ax, logms[jms], np.log10(shsmr[jms]), '-', lw=3, color=scolor,
            label='Overall', zorder=100)
    if chsmr is not None:
        plot_line(
            ax, logms[jms], np.log10(chsmr[jms]), '-', lw=3, color=ccolor,
            label='Centrals', zorder=100)
    if shsmr is not None or chsmr is not None:
        ax.legend(fontsize=18)
    ax.set_xlabel(r'$\log\,{0}$'.format(sim.masslabel(mtype='stars')))
    ax.set_ylabel(r'$\log\,{0}$'.format(sim.masslabel(mtype='total')))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    #ax.set_xlim(mslim)
    #ax.set_ylim(mlim)
    save_plot(fig, f'shmr_{bincol}', sim)
    #save_plot(fig, 'shmr_mu', sim)
    return


def plot_shmr(sim, subs, hostmass='M200Mean'):
    cen, sat, mtot, mstar, mhost, dark, Nsat, Ndark, Ngsat = \
        definitions(subs, hostmass=hostmass)
    #mbins, m, mlim, msbins, ms, mslim = binning(sim)
    mbins, m, mlim = binning(sim, mtype='total')
    msbins, ms, mslim = binning(sim, mtype='stars')
    nh = np.histogram(mhost[cen], mbins)[0]
    nc = np.histogram2d(
        mstar[cen & ~dark], mhost[cen & ~dark], (msbins,mbins))[0]
    nsat = np.histogram2d(
        mstar[sat & ~dark], mhost[sat & ~dark], (msbins,mbins))[0]
    nsub = np.histogram2d(
        mstar[sat & ~dark], mtot[sat & ~dark], (msbins,mbins))[0]

    extent = [np.append(mlim, mslim), np.append(mslim, mlim)]
    lw = 4
    fig, axes = plt.subplots(figsize=(16,12), ncols=2, nrows=2)
    row = axes[0]
    for ax, n, label, mtype \
            in zip(row, (nc,nsub), 'cs', ('total','total')):
        s = ax.imshow(
            np.log10(n), extent=extent[0], origin='lower', aspect='auto')
        plt.colorbar(
            s, ax=ax, label=r'$\log\,N_\mathrm{{{0}}}$'.format(label))
        ax.set_xlabel(r'$\log\,{0}$'.format(sim.masslabel(mtype=mtype)))
        ax.set_ylabel(r'$\log\,{0}$'.format(sim.masslabel(mtype='stars')))
    # mean central stellar to halo mass relation
    mstar_cmean = np.histogram(
            mhost[cen & ~dark], mbins, weights=mstar[cen & ~dark])[0] \
        / np.histogram(mhost[cen & ~dark], mbins)[0]
    # and for satellites
    mstar_smean = np.histogram(
        mtot[sat & ~dark], mbins, weights=mstar[sat & ~dark])[0] \
        / np.histogram(mtot[sat & ~dark], mbins)[0]
    # minimum masses for complete samples
    mmin = 10
    jm = (m >= 10**mmin)
    msmin = 7.5
    jms = (ms >= 10**msmin)
    for ax in row:
        plot_line(ax, np.log10(m[jm]), np.log10(mstar_cmean[jm]), '-',
                  color=ccolor, lw=lw, zorder=10, label='Central SHMR')
        plot_line(ax, np.log10(m[jm]), np.log10(mstar_smean[jm]), '-',
                  color=scolor, lw=lw, zorder=10, label='Satellite SHMR')
    for ax in row:
        ax.legend(loc='upper left')
        ax.set_xlim(*mlim)
        ax.set_ylim(*mslim)
    row = axes[1]
    for ax, n, label, mtype \
            in zip(row, (nc,nsub), 'cs', ('total','total')):
        s = ax.imshow(
            np.log10(n.T), extent=extent[1], origin='lower', aspect='auto')
        plt.colorbar(
            s, ax=ax, label=r'$\log\,N_\mathrm{{{0}}}$'.format(label))
        ax.set_ylabel(r'$\log\,{0}$'.format(sim.masslabel(mtype=mtype)))
        ax.set_xlabel(r'$\log\,{0}$'.format(sim.masslabel(mtype='stars')))
    mtot_cmean = np.histogram(
            mstar[cen & ~dark], msbins, weights=mhost[cen & ~dark])[0] \
        / np.histogram(mstar[cen & ~dark], msbins)[0]
    mtot_smean = np.histogram(
            mstar[sat & ~dark], msbins, weights=mtot[sat & ~dark])[0] \
        / np.histogram(mstar[sat & ~dark], msbins)[0]
    for ax in row:
        plot_line(ax, np.log10(ms[jms]), np.log10(mtot_cmean[jms]), '-',
                  color=ccolor, lw=lw, zorder=10, label='Central HSMR')
        plot_line(ax, np.log10(ms[jms]), np.log10(mtot_smean[jms]), '-',
                  color=scolor, lw=lw, zorder=10, label='Satellite HSMR')
    # van Uitert+16 (KiDS)
    v16 = ascii.read('literature/vanuitert16.txt', format='commented_header')
    v16['logmstar'] = sim.mass_to_sim_h(
        v16['logmstar'], 0.7, 'stars', log=True)
    v16['logmhalo'] = sim.mass_to_sim_h(
        v16['logmhalo'], 0.7, 'total', log=True)
    #row[0].plot(v16['logmstar'], v16['logmhalo'], 'w-', lw=5, zorder=19,
                #label='_none_')
    #row[0].plot(v16['logmstar'], v16['logmhalo'], 'k-', lw=3, zorder=20,
                #label='van Uitert+16')
    plot_line(row[0], v16['logmstar'], v16['logmhalo'], 'C1-', lw=lw,
              zorder=20, label='van Uitert+16')
    s18 = np.array(
        [[9.51,10.01,10.36,10.67,11.01], [10.64,11.41,11.71,11.84,12.15],
         [0.53,0.21,0.17,0.15,0.20], [0.39,0.17,0.15,0.15,0.17]])
    # correcto to EAGLE's h
    s18[0] = sim.mass_to_sim_h(s18[0], 0.7, 'stars', log=True)
    s18[1] = sim.mass_to_sim_h(s18[1], 0.7, 'total', log=True)
    row[1].errorbar(
        s18[0], s18[1], s18[2:], fmt='wo', ms=18, elinewidth=6, zorder=19)
    row[1].errorbar(
        s18[0], s18[1], s18[2:], fmt='ko', ms=14, elinewidth=2, zorder=20,
        label="Sifon+18")
    for ax in row:
        ax.legend(loc='upper left')
        ax.set_xlim(*mslim)
        ax.set_ylim(*mlim)
    save_plot(fig, 'shmr_censat', sim)
    return mstar_cmean, mstar_smean, mtot_cmean, mtot_smean


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
    axes[0].set_xlabel(r'log $m_\mathrm{total}$')
    axes[1].set_xlabel(r'log $m_\mathrm{star}$')
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


def binning(sim=None, mtype='total', n=None, xmin=0, xmax=1, log=True):
    if sim is None:
        f = np.logspace if log else np.linspace
        bins = f(xmin, xmax, n+1)
    else:
        bins = massbins(sim, mtype=mtype)
        if n is not None:
            bins = np.log10(bins[::bins.size-1])
            bins = np.logspace(bins[0], bins[1], n+1)
    x = logcenters(bins)
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


def definitions(subs, hostmass='M200Mean'):
    cen = (subs.catalog['Rank'] == 0)
    sat = ~cen
    mtot = subs.mass('total')
    mstar = subs.mass('stars')
    mhost = subs.catalog[hostmass]
    dark = (subs.catalog['IsDark'] == 1)
    Nsat = subs.catalog['Nsat']
    Ndark = subs.catalog['Ndark']
    Ngsat = Nsat - Ndark
    return cen, sat, mtot, mstar, mhost, dark, Nsat, Ndark, Ngsat


def logcenters(bins):
    logbins = np.log10(bins)
    return 10**((logbins[:-1]+logbins[1:])/2)


def massbins(sim, mtype='stars'):
    bins = {'apostle': {'stars': np.logspace(3, 11, 21)},
            'eagle': {'stars': np.logspace(5.5, 12.5, 26),
                      'total': np.logspace(8, 15, 31)}}
    return bins[sim.family][mtype]


def plot_line(ax, *args, **kwargs):
    if 'lw' not in kwargs:
        kwargs['lw'] = 4
    kwargs_bg = kwargs.copy()
    if 'color' in kwargs:
        kwargs_bg.pop('color')
    if 'label' in kwargs:
        kwargs_bg.pop('label')
    kwargs_bg['lw'] += 2
    ax.plot(*args, **kwargs_bg, color='w', label='_none_')
    ax.plot(*args, **kwargs)
    #ax.plot(np.log10(m[jm]), np.log10(mstar_cmean[jm]), 'w-', lw=lw+2,
            #zorder=10, label='_none_')
    #ax.plot(np.log10(m[jm]), np.log10(mstar_cmean[jm]), 'C9-', lw=lw,
            #zorder=10, label='Central SHMR')
    return


def output(sim, xname, yname):
    return '{0}_{1}'.format(sim.masslabel(mtype=xname, latex=False),
                            sim.masslabel(mtype=yname, latex=False))


def save_plot(fig, output, sim, **kwargs):
    if '/' in output:
        path = os.path.join(sim.plot_path, os.path.split(output)[0])
        if not os.path.isdir(path):
            os.makedirs(path)
    out = os.path.join(sim.plot_path, '{0}.pdf'.format(output))
    savefig(out, fig=fig, **kwargs)
    return



main()




