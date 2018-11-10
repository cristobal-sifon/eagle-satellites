from astropy import units as u
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
from scipy.stats import binned_statistic as binnedstat
import sys
from time import time
if sys.version_info[0] == 2:
    range = xrange

from plottools.plotutils import savefig, update_rcParams
update_rcParams()

from HBTReader import HBTReader
from core import hbt_tools
from core.simulation import Simulation
from core.subhalo import HostHalos, Subhalos, Track



def main(debug=True):

    args = hbt_tools.parse_args()
    sim = Simulation(args.simulation)

    to = time()
    reader = HBTReader(sim.path)
    print('Loaded reader in {0:.1f} seconds'.format(time()-to))

    isnap = -1
    subs = reader.LoadSubhalos(isnap)

    wrap_plot(sim, subs, isnap)

    return


def wrap_plot(sim, subs, isnap, debug=True):
    # this applies to apostle
    mstarbins = mbins(sim, 'stars')

    #halos = HostHalos(sim, isnap)
    #print(halos.dtype)
    #print(halos.colnames)

    #sat = Subhalos(subs[subs['Rank'] > 0], sim, isnap)
    #cen = Subhalos(subs[subs['Rank'] == 0], sim, isnap)
    #print('{2}: {0} centrals and {1} satellites'.format(
        #(subs['Rank'] == 0).sum(), (subs['Rank'] > 0).sum(),
        #subs['Rank'].size))

    #sat.load_hosts(verbose=True)

    subs = Subhalos(subs, sim, isnap)
    #subs.load_hosts(verbose=True)
    #subs.host_velocities()
    subhalos = subs.catalog
    print(np.sort(subs.colnames))
    #sys.exit()

    # plot phase-space
    #norm = True
    #fig, axes = plt.subplots(figsize=(14,10), ncols=2, nrows=2)
    #axes = np.reshape(axes, -1)
    #axes[0].set_title('3d')
    #axes[0].plot(subs.distance(), 

    fig, axes = plt.subplots(figsize=(12,5), ncols=2)
    bins = [np.linspace(0, 10000, 100), np.logspace(-1, 4, 100)]
    for ax, b in zip(axes, bins):
        ax.hist(subs.velocity(), b, histtype='step', label='3d', bottom=1)
        for i, x in enumerate('xyz'):
            ax.hist(subs.velocity(i), b, histtype='step', label=x, bottom=1)
        ax.legend(loc='upper right')
        ax.set_xlabel('Peculiar velocity (km/s)')
        ax.set_yscale('log')
    axes[0].set_title('Linear binning')
    axes[1].set_title('Log-scale binning')
    output = 'vpeculiar'
    save_plot(fig, output, sim)

    mass_sigma_host(sim, subs)

    #test_velocities(subs)

    plot_occupation(sim, subs)
    #plot_relation(sim, subs, cen, sat, mstarbins, debug=debug)
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


def plot_occupation(sim, subs, hostmass='M200Mean'):
    # quick test
    #plt.
    cen = (subs.catalog['Rank'] == 0)
    sat = ~cen
    mtot = subs.mass('total')
    mstar = subs.mass('stars')
    mhost = subs.catalog[hostmass]
    #mhost = subs.
    dark = (subs.catalog['IsDark'] == 1)
    Nsat = subs.catalog['Nsat']
    Ndark = subs.catalog['Ndark']
    Ngsat = Nsat - Ndark
    # testing
    bins = np.append([0], np.logspace(0, 6, 100))
    fig, ax = plt.subplots()
    ax.hist(subs.nbound('stars'), bins)
    ax.set_xscale('log')
    ax.set_yscale('log')
    save_plot(fig, 'Nbound_stars', sim)
    fig, ax = plt.subplots()
    logm = np.log10(subs.mass('total'))
    logmbins = np.arange(8, 15, 0.1)
    ax.hist(logm[cen & dark], logmbins, histtype='step', bottom=1,
            label='Dark centrals')
    ax.hist(logm[sat & dark], logmbins, histtype='step', bottom=1,
            label='Dark satellites')
    ax.hist(logm[cen], logmbins, histtype='step', bottom=1,
            label='All centrals')
    ax.hist(logm[sat], logmbins, histtype='step', bottom=1,
            label='All satellites')
    ax.legend(fontsize=12)
    ax.set_xlabel('log total mass')
    ax.set_ylabel('1 + Number of dark subhalos')
    ax.set_yscale('log')
    save_plot(fig, 'mass_Ndark', sim)
    print('{0} dark subhalos'.format(subs.catalog['IsDark'].sum()))
    # binning and plot
    mbins = np.logspace(8, 15, 41)
    m = logcenters(mbins)
    msbins = np.logspace(5, 13, 31)
    ms = logcenters(msbins)
    #nh = np.histogram2d(mstar[cen], mhost[cen], (msbins,mbins))[0]
    nh = np.histogram(mhost[cen], mbins)[0]
    nc = np.histogram2d(
        mstar[cen & ~dark], mhost[cen & ~dark], (msbins,mbins))[0]
    nsat = np.histogram2d(
        mstar[sat & ~dark], mhost[sat & ~dark], (msbins,mbins))[0]
    print('nh =', nh.shape)
    extent = np.log10(np.array([mbins[0],mbins[-1],msbins[0],msbins[-1]]))
    fig, axes = plt.subplots(figsize=(15,6), ncols=2)
    for ax, n, label in zip(axes, (nc, nsat), 'cs'):
        s = ax.imshow(
            np.log10(n), extent=extent, origin='lower', aspect='auto')
        plt.colorbar(
            s, ax=ax, label=r'$\log\,N_\mathrm{{{0}}}$'.format(label))
        ax.set_xlabel(r'$\log\,{0}$'.format(sim.masslabel(mtype='total')))
        ax.set_ylabel(r'$\log\,{0}$'.format(sim.masslabel(mtype='stars')))
    save_plot(fig, 'numbers_mstar_mtot', sim)
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
    print('ms[j] =', ms[j])
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


def plot_relation(sim, subs):
    cen = (subs.catalog['Rank'] == 0)
    sat = ~cen
    mstar = subs.mass('stars')
    mtot = subs.mass('total')
    nsat = subs.catalog['Nsat']
    msbins = np.linspace(6, 13, 25)
    ms = 10**((msbins[1:]+msbins[:-1]) / 2)
    msbins = 10**msbins
    #fig, axes = plt.subplots(figsize=(12,5), ncols=2)
    #ax = axes[0]
    #ax.scatter(
    return


def plot_relation_old(sim, subs, cen, sat, bins, rbins, xname='stars',
                  yname='total', ccolor='C0', scolor='C1', debug=False):
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
    # satellites at varying distances
    
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


def logcenters(bins):
    logbins = np.log10(bins)
    return 10**((logbins[:-1]+logbins[1:])/2)


def munari13(m, cosmo, z=0):
    hz = (cosmo.H(z) / (100*u.km/u.s/u.Mpc)).value
    return 1177 * (hz*m/1e15)**0.364 


def save_plot(fig, output, sim):
    out = os.path.join(sim.plot_path, '{0}.pdf'.format(output))
    savefig(out, fig=fig)
    return


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




