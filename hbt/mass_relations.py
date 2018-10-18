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
from core.subhalo import HostHalos, Subhalos, Track


def main(debug=True):

    args = hbt_tools.parse_args()
    sim = Simulation(args.simulation)

    to = time()
    reader = HBTReader(sim.path)
    print('Loaded reader in {0:.1f} seconds'.format(time()-to))

    isnap = -1
    subs = reader.LoadSubhalos(isnap)

    # testing inf velocities
    vavg = subs['PhysicalAverageVelocity']
    vbound = subs['PhysicalMostBoundVelocity']
    bad_avg = (np.sum(1-np.isfinite(vavg), axis=1) > 0)
    bad_bound = (np.sum(1-np.isfinite(vbound), axis=1) > 0)
    print('bad_avg: {0:.0f}/{1} ({2:.1f}%)'.format(
        bad_avg.sum(), bad_avg.size, 100*bad_avg.sum()/bad_avg.size))
    print('bad_bound: {0:.0f}/{1} ({2:.1f}%)'.format(
        bad_bound.sum(), bad_bound.size, 100*bad_bound.sum()/bad_bound.size))
    #return

    wrap_plot(sim, subs, isnap)

    return


def wrap_plot(sim, subs, isnap, debug=True):
    # this applies to apostle
    mstarbins = mbins(sim, 'stars')

    #halos = HostHalos(sim, isnap)
    #print(halos.dtype)
    #print(halos.colnames)

    sat = Subhalos(subs[subs['Rank'] > 0], sim, isnap)
    cen = Subhalos(subs[subs['Rank'] == 0], sim, isnap)
    print('{2}: {0} centrals and {1} satellites'.format(
        (subs['Rank'] == 0).sum(), (subs['Rank'] > 0).sum(),
        subs['Rank'].size))

    #sat.load_hosts(verbose=True)

    subs = Subhalos(subs, sim, isnap)
    subs.load_hosts(verbose=True)
    #subs.distance2host(projection='xyz', append_key='distance_xyz')
    subhalos = subs.catalog
    print(np.sort(subs.colnames))

    # plot phase-space
    norm = True
    fig, axes = plt.subplots(figsize=(14,10), ncols=2, nrows=2)
    axes = np.reshape(axes, -1)
    projections = ['xyz', 'xy', 'yz', 'xz']
    to = time()
    xyz = np.array(
         [subhalos['ComovingMostBoundPosition{0}'.format(i)]
          - subhalos['CenterComoving{0}'.format(i)]
         for i in range(3)])
    print('Calculated cluster-centric distances in {0:.2f} s'.format(time()-to))
    # note that for the velocities I want to use the direction I'm
    # *not* using for the positions
    # this is 3d for now
    to = time()
    vref = 'MostBound'
    #subs.host_velocities(ref='Average', mass_weighting='stars')
    #subs.host_velocities(ref='Average', mass_weighting=None)
    subs.host_velocities(ref=vref, mass_weighting='stars')
    print('Calculated host velocities in {0:.2f} s'.format(time()-to))
    print()
    #view_velocities(subs, sim, vref)
    print(np.sort(list(subs.catalog)))

    to = time()
    v = np.array(
        [subhalos['Physical{1}Velocity{0}'.format(i, vref)]
         for i in range(3)])
    print('Stored velocities in {0:.2f} s'.format(time()-to))

    #test_velocities(subs)

    if norm:
        print('xyz =', xyz.shape, subhalos['R200MeanComoving'].shape)
        print(xyz[:,0], subhalos['R200MeanComoving'][0])
        xyz = xyz / np.array(subhalos['R200MeanComoving'])
        print('xyz =', xyz.shape)
        #v = v / np.array(subhalos['
    #for ax, projection in zip(projections):

    plot_relation(sim, subs, cen, sat, mstarbins, debug=debug)

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


def plot_relation(sim, subs, cen, sat, bins, rbins, xname='stars',
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




