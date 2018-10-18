from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import h5py
from itertools import count
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.recfunctions import append_fields
import os
import pandas as pd
import six
import sys
from time import time

from HBTReader import HBTReader

from .simulation import BaseSimulation, Simulation


class BaseDataSet(object):

    def __init__(self, catalog, as_dataframe=True):
        self._catalog = catalog
        self._as_dataframe = as_dataframe
        # private attributes
        self.__range = None

    ### private properties ###

    @property
    def _range(self):
        if self.__range is None:
            self.__range = np.arange(self._catalog.size, dtype=int)
        return self.__range

    ### public properties ###

    @property
    def as_dataframe(self):
        return self._as_dataframe

    @as_dataframe.setter
    def as_dataframe(self, as_df):
        assert isinstance(as_df, bool), 'as_dataframe must be boolean'
        self._as_dataframe = as_df

    @property
    def colnames(self):
        if self.as_dataframe:
            return list(self.catalog)
        else:
            return self.catalog.dtype.names

    @property
    def catalog(self):
        if self.as_dataframe and not isinstance(self._catalog, pd.DataFrame):
            self._catalog = self.DataFrame(self._catalog)
        elif not self.as_dataframe \
                and isinstance(self._catalog, pd.DataFrame):
            self._catalog = self._catalog.to_records()
        return self._catalog

    @catalog.setter
    def catalog(self, cat):
        self._catalog = cat

    ### methods ###

    def DataFrame(self, recarray):
        """Return a ``pandas.DataFrame`` object"""
        df = {}
        for dtype in recarray.dtype.descr:
            name = dtype[0]
            if len(recarray[name].shape) == 1:
                df[name] = recarray[name]
            else:
                for i in range(recarray[name].shape[1]):
                    df['{0}{1}'.format(name, i)] = recarray[name][:,i]
        return pd.DataFrame(df)


class BaseSubhalo(BaseDataSet):
    """BaseSubhalo class"""

    def __init__(self, catalog, sim, as_dataframe=True):
        """
        Parameters
        ----------
        catalog : ``Track.track`` or ``Subhalos.data``
        sim : ``Simulation``
        """
        # setting to False here to load the Mbounds and Nbounds below
        # (easier to load from hdf5 than DataFrame)
        super(BaseSubhalo, self).__init__(catalog, as_dataframe=False)
        BaseSimulation.__init__(self, sim)
        self.Mbound = 1e10 * self.catalog['Mbound']
        self.MboundType = 1e10 * self.catalog['MboundType']
        self.Nbound = self.catalog['Nbound']
        self.NboundType = self.catalog['NboundType']
        self.as_dataframe = as_dataframe

    ### methods ###

    def mass(self, mtype=None, index=None):
        assert mtype is not None or index is not None, \
            'must provide either ``mtype`` or ``index``'
        if mtype is not None:
            if mtype.lower() in ('total', 'mbound'):
                return self.Mbound
            return self.MboundType[:,self.sim._masstype_index(mtype)]
        if index == -1:
            return self.Mbound
        return self.MboundType[:,index]


class HostHalos(BaseDataSet, BaseSimulation):
    """Class containing virial quantities of host halos

    """

    def __init__(self, sim, isnap, as_dataframe=True, force_isnap=False):
        """

        if `force_isnap` is True, a ValueError will be raised if
        virial quantities do not exist for snapshot `isnap`. Otherwise,
        the nearest snapshot will be used (raises UserWarning).

        """
        BaseSimulation.__init__(self, sim)
        # alias
        self.available_snapshots = self.sim.virial_snapshots
        self.force_isnap = force_isnap
        self.isnap = self._set_isnap(isnap)
        self._filename = None
        catalog = h5py.File(self.filename, 'r')['HostHalos']
        super(HostHalos, self).__init__(
            catalog, as_dataframe=as_dataframe)

    @property
    def filename(self):
        if self._filename is None:
            self._filename = os.path.join(
                self.sim.path, 'HaloSize',
                'HaloSize_{0}.hdf5'.format(self.isnap))
        return self._filename

    ### private methods

    def _set_isnap(self, isnap):
        if isnap < 0:
            isnap = self.sim.snapshots[isnap]
        if isnap not in self.available_snapshots:
            if self.force_isnap:
                msg = 'Halo sizes not present for snapshot {0}'.format(
                    isnap)
                raise ValueError(msg)
            j = np.argmin(abs(isnap-self.available_snapshots))
            isnap_used = self.available_snapshots[j]
            warn = 'Snapshot {0} does not have halo information. Using' \
            ' snapshot {1} instead.'.format(isnap, isnap_used)
            warnings.warn(warn)
            isnap = isnap_used
        return isnap


class HostHaloTrack(BaseSubhalo):
    """Class containing tracks of all subhalos in a given halo

    """


    def __init__(self, hosthaloid, subhalos, sim, isnap=-1):
        self._hosthaloid = hosthaloid
        self._subhalos = subhalos
        self._isnap = isnap
        super(HostHaloTrack, self).__init__(subhalos, sim)

    @property
    def children(self):
        return self._children

    @property
    def hosthaloid(self):
        return self._hosthaloid

    @hosthaloid.setter
    def hosthaloid(self, hostid):
        self._hosthaloid = hostid
        self._children = \
            self._range[self.catalog['HostHaloId'] == self._hosthaloid]


class Subhalos(BaseSubhalo):
    """Class to manage a sample of subhalos at a given snapshot

    """

    def __init__(self, catalog, sim, isnap, exclude_non_FoF=False,
                 as_dataframe=True):
        """
        Parameters
        ----------
        subhalos : output of ``reader.LoadSubhalos``
            subhalo sample, defined at a specific snapshot
        sim : ``Simulation`` object or ``str``
            simulation containing the track. If ``str``, should be
            simulation label (see ``Simulation.mapping``)
        isnap : ``int``
            snapshot index at which the subhalos were retrieved.
            Unfortunately this has to be given by hand for now.
        exclude_non_FoF : bool, optional
            whether to exclude all objects not part of any FoF halo
            (i.e., those with HostHaloId=-1). Some attributes or
            methods may not work if set to False.
        """
        super(Subhalos, self).__init__(
            catalog, sim, as_dataframe=as_dataframe)
        self.exclude_non_FoF = exclude_non_FoF
        self.non_FoF = (self.catalog['HostHaloId'] == -1)
        if self.exclude_non_FoF:
            print('Excluding {0} non-FoF subhalos'.format(self.non_FoF.sum()))
            self.catalog = self.catalog[~self.non_FoF]
            print('catalog size =', self.catalog['Rank'].size)
        self.isnap = isnap
        self._central_idx = None
        self._central_mask = None
        self._centrals = None
        self._orphan = None
        self._satellite_idx = None
        self._satellite_mask = None
        self._satellites = None
        self._has_host_properties = False
        self._has_velocities = False

    ### attributes ###

    @property
    def centrals(self):
        return self.catalog[self.central_mask]

    @property
    def central_idx(self):
        if self._central_idx is None:
            self._central_idx = self._range[self._central_mask]
        return self._central_idx

    @property
    def central_mask(self):
        if self._central_mask is None:
            self._central_mask = (self.catalog['Rank'] == 0)
        return self._central_mask

    @property
    def orphan(self):
        """boolean mask, True if object is orphan"""
        if self._orphan is None:
            self._orphan = (self.catalog['Nbound'] == 1)
        return self._orphan

    @property
    def satellites(self):
        return self.catalog[self.satellite_mask]

    @property
    def satellite_idx(self):
        if self._satellite_idx is None:
            self._satellite_idx =  self._range[self.satellite_mask]
        return self._satellite_idx

    @property
    def satellite_mask(self):
        if self._satellite_mask is None:
            self._satellite_mask = (self.catalog['Rank'] > 0)
        return self._satellite_mask

    ### methods ###

    def distance2host(self, projection='xyz', append_key='HostHaloDistance',
                      position_name='ComovingMostBoundPosition', read=True,
                      verbose=True):
        """Calculate the distance of all subhalos to the center of
        their host

        Parameters
        ----------
        hostid : int
            trackID of the host subhalo
        projection : str, optional
            projection along which to calculate distances. Must be a
            combination of two of {'xyz'}
        append : bool, optional
            whether to add the result as a new column to
            ``self.catalog``
        read : bool, optional -- NOT IMPLEMENTED
            whether to read the distances stored in a file, if it
            exists, rather than calculating them again. The file path
            is constructed automatically from the simulation details.

        Returns
        -------
        dist2host : float array
            distances to host in Mpc, in 3d or along the given
            projection
        """
        axes = ' xyz'
        self.as_dataframe = True
        columns = [
            '{0}{1}'.format(position_name, axes.index(i))
            for i in projection]
        # this is how the columns will be in aux
        auxcols = ['{0}_h'.format(c) for c in columns]
        columns.append('HostHaloId')
        # alias
        sub = self.catalog
        to = time()
        aux = sub.join(
            sub[sub['Rank'] == 0][columns].set_index('HostHaloId'),
            on='HostHaloId', rsuffix='_h')
        dist = np.sum(
            (aux[columns[:-1]].values - aux[auxcols].values)**2, axis=1)**0.5
        if verbose:
            print('Calculated distances in {0:.2f} s'.format(time()-to))
        if append_key:
            self.catalog[append_key] = dist
        else:
            return dist

    def host(self, trackid, return_value='index'):
        """Host halo of a given trackid

        Parameters
        ----------
        trackid : int
            track ID
        return_value : {'index', 'mask', 'trackid', 'table'}, optional
            whether to return the index, a boolean mask, or the full
            table containing properties of siblings only

        Returns
        -------
        The returned value will depend on ``return_value``:
            -if ``return_value='index'``:
                int corresponding to the index of the host in
                ``self.track``
            -if ``return_value='trackid'``:
                int corresponding to the host's track ID
            -if ``return_value='mask'``:
                boolean mask to be applied to ``self.track``
            -if ``return_value='table'``:
                np.struct_array corresponding to the full entry for the
                host in ``self.table``
        """
        assert trackid // 1 == trackid, 'trackid must be an int'
        _valid_return = ('index','mask','table','track','trackid')
        assert return_value in _valid_return, \
            'return_value must be one of {0}'.format(_valid_return)
        sib = self.siblings(trackid, return_value='mask')
        if sib is None:
            return None
        else:
            host_mask = sib & (self.catalog['Rank'] == 0)
        if return_value == 'mask':
            return host_mask
        if return_value == 'track':
            return self.reader.GetTrack(
                self.catalog['TrackId'][host_mask][0])
        if return_value == 'trackid':
            return self.catalog['TrackId'][host_mask][0]
        if return_value == 'index':
            return self._range[host_mask][0]
        return self.catalog[host_mask]

    def _mass_weighted_stat(self, values, mass, label):
        """Apply mass weighting

        Parameters
        ----------
        values : ``np.array``
            quantities to be mass-weighted
        mass : ``pandas.Series``
            mass with which to weight
        label : str
            name of the returned Series

        Returns
        -------
        weighted_stat : ``pd.Series``
            mass-weighted statistic
        """
        wstat = {label: np.sum(values*mass) / np.sum(mass)}
        return pd.Series(wstat, index=[label])

    def _mass_weighted_average(self, x, cols, mcol, label='vcl'):
        """Mass-weighted average

        Calculate the mass-weighted average of the quantity(ies)
        defined in column(s) `cols`. Meant to be used with
        ``pandas.groupby.apply``

        Parameters
        ----------
        x : ``pd.groupby
        cols : str or list of str
            column(s) to be averaged
        mcol : str
            mass column used for the weighting. See
            ``Simulation.masstypes`` and
            ``Simulation.masstype_pandas_columns``
        

        Returns
        -------
        avg : ``pandas.Series``
            average quantities
        """
        cols = [cols] if isinstance(cols, six.string_types) else cols
        if x[cols[0]].count() == 1:
            return pd.Series({label: 0}, index=[label])
        vmean = np.mean([x[i] for i in cols], axis=0)
        vxyz = np.sum([(x[i]-vi)**2 for i, vi in zip(cols, vmean)], axis=0)
        #print('vxyz =', vxyz, vxyz.shape, np.array(x[cols[0]]).shape)
        return self._mass_weighted_stat(vxyz, np.array(x[mcol]), label)**0.5

    def _mass_weighted_std(self, x, cols, mcol, label='sigma_cl'):
        cols = [cols] if isinstance(cols, six.string_types) else cols
        if x[cols[0]].count() == 1:
            return pd.Series({label: 0}, index=[label])
        # this is correct only if they are peculiar velocities
        sigma_xyz = np.sum([x[i]**2 for i in cols], axis=0)
        return self._mass_weighted_stat(
            sigma_xyz, np.array(x[mcol]), label)**0.5

    def host_velocities(self, ref='Average', mass_weighting='stars'):
        if not self._has_host_properties:
            self.load_hosts()
        # alias
        cx = self.catalog.copy()
        fig, ax = plt.subplots(figsize=(8,7))
        mbins = np.logspace(3, 11, 31)
        for i in range(6):
            m = 1e10 * cx['MboundType{0}'.format(i)]
            n = (m >= mbins[0]) & (m <= mbins[-1])
            ax.hist(m, mbins, histtype='step', bottom=1,
                    label='Type {0} ({1})'.format(i, n.sum()))
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend()
        fig.savefig(os.path.join(self.sim.plot_path, 'Mhist.pdf'))
        plt.close()
        axes = 'xyz'
        vcols = ['Physical{1}Velocity{0}'.format(i, ref) for i in range(3)]
        mcol = self.sim.masstype_pandas_column(mass_weighting) \
            if isinstance(mass_weighting, six.string_types) else None
        mweight = cx[mcol]
        keys = list(cx)
        # mean velocities
        grcols = ['HostHaloId', 'Nsat']
        skip = len(grcols)
        mvcols = []
        if mcol is not None:
            print('mass:\n', mweight.describe())
        print()
        for ax, vcol in zip(axes, vcols):
            mvcols.append('mv{0}'.format(ax))
            cx[mvcols[-1]] = cx[vcol] if mcol is None else cx[vcol] * mweight
        grcols = np.append(grcols, mvcols)
        print('grcols =', grcols)
        group = cx.groupby('HostHaloId')
        wmean = group[mvcols].sum()
        msum = group[mcol].sum()
        #msum[msum == 0] = 1
        print('velocity:\n', cx[vcols[0]].describe())
        print()
        print('wmean:\n', wmean[mvcols[0]].describe())
        print()
        #print('msum:\n', np.sort(np.unique(msum)))
        print('msum:\n', msum.describe())
        print()
        print('cx: ', np.sort(list(cx)))
        #print('cx =', cx)
        print()
        # these should be discarded at some other point
        #msum[msum == 0] = 1
        vcl = pd.DataFrame({})
        vcol = 'Physical{0}HostMeanVelocity'.format(ref)
        for i, mvcol in enumerate(mvcols):
            vcl[vcol+str(i)] = wmean[mvcol] / msum
        vcl[vcol] = np.sum(wmean[mvcols]**2, axis=1)**0.5
        print('vcl =', np.sort(list(vcl)))
        vnan = np.isnan(vcl['Physical{0}HostMeanVelocity'.format(ref)]) \
            | ~np.isfinite(vcl['Physical{0}HostMeanVelocity'.format(ref)])
        print('vcol:', vcol, 'size:', vcl[vcol].size, 'nan:', vnan.sum(),
              'finite:', (1-vnan).sum())
        # velocity dispersions
        to = time()
        for i, ax, vcol in zip(count(), axes, vcols):
            vdiff = cx[vcol] \
                - vcl['Physical{0}HostMeanVelocity{1}'.format(ref, i)]
            cx[mvcols[i]] = mweight * vdiff**2
        print('weighted differences in {0:.2f} seconds'.format(time()-to))
        # can I do the above without the loop?
        #cx[
        to = time()
        group = cx.groupby('HostHaloId')
        wstd = group[mvcols].sum()
        print('grouped in {0:.1f} seconds'.format(time()-to))
        scol = 'Physical{0}HostVelocityDispersion'.format(ref)
        scols = []
        to = time()
        for i, mvcol in enumerate(mvcols):
            scols.append(scol+str(i))
            vcl[scols[-1]] = (wstd[mvcol] / msum)**0.5
        vcl[scol] = np.sum(wstd, axis=1)**0.5
        print('dispersions in {0:.1f} seconds'.format(time()-to))
        #vcl[scol][msum == 0] = 0
        snan = np.isnan(vcl[scol]) | ~np.isfinite(vcl[scol])
        print('scol:', scol, 'size:', vcl[scol].size, 'nan:', snan.sum(),
              'finite:', (1-snan).sum())
        #sys.exit()
        self.catalog = self.catalog[keys].join(
            vcl, on='HostHaloId', rsuffix='')
        printcols = ['HostHaloId','Nsat',#'PhysicalMostBoundHostMeanVelocity1',
                     'PhysicalMostBoundHostMeanVelocity']
        isolated = (self.catalog['Nsat'] == 0)
        vinf = ~np.isfinite(self.catalog[vcol])
        sinf = ~np.isfinite(self.catalog[scol])
        #
        """
        jhost = np.sort(np.unique(self.catalog['HostHaloId'][vinf]))
        print('jhost =', jhost.size)
        nprinted = 0
        for j in jhost:
            jj = (self.catalog['HostHaloId'] == j)
            nsat = np.unique(self.catalog['Nsat'][jj])[0]
            if 0 < nsat < 20:
                print(j, nsat)
                nprinted += 1
            if nprinted == 20:
                break
        print(np.sort(np.unique(self.catalog['Nsat'][vinf])))
        print('vinf:', vinf.size, isolated.sum(), vinf.sum(),
              (isolated & vinf).sum(), np.array_equal(isolated, vinf))
        print('sinf:', sinf.size, sinf.sum(), (sinf == 0).sum(),
              (msum == 0).sum())
        massive = (self.catalog['Mbound'] > 1)
        print('massive:', massive.sum(), (massive & sinf).sum(),
              (massive & vinf).sum())
        print()
        from astropy.io import fits
        from astropy.table import Table
        cat = self.catalog[jj]
        tbl = Table.from_pandas(cat)
        tbl.write('test_vinf.fits', format='fits', overwrite=True)
        #sys.exit()
        """
        return


    def index(self, trackid):
        """Index of a given trackid in the subhalo catalog

        Parameters
        ----------
        trackid : int or array of int
            track ID(s)
        """
        assert trackid // 1 == trackid, \
            'trackid must be an int or an array of int'
        if hasattr(trackid, '__iter__'):
            return self._range[self.catalog['TrackId'] == trackid]
        return self._range[self.catalog['TrackId'] == trackid][0]

    def is_central(self, trackid):
        """Whether a given subhalo is a central subhalo

        Parameters
        ----------
        trackid : int or array of int
            track ID of the subhalo in question

        Returns
        -------
        is_central : bool or array of bool
            ``True`` if ``Rank=0``, else ``False``
        """
        assert trackid // 1 == trackid, \
            'trackid must be an int or an array of int'
        # in case they're floats of ints
        trackid = np.array(trackid, dtype=int)
        cent = (self.data['Rank'][self.data['TrackId'] == trackId] == 0)
        if hasattr(trackid, '__iter__'):
            return cent
        return cent[0]

    def load_hosts(self, force_isnap=False, verbose=False):
        """Load halo masses and sizes into the subhalo data

        See `HostHalos` for details
        """
        if self.isnap not in self.sim.virial_snapshots:
            hosts = HostHalos(self.sim, self.isnap, force_isnap=force_isnap)
            #print('hosts =', np.sort(hosts.colnames))
            to = time()
            grouped = self.catalog[['HostHaloId']].groupby('HostHaloId')
            nm = pd.DataFrame({'Nsat': grouped['HostHaloId'].count()-1})
            self.catalog = self.catalog.join(nm, on='HostHaloId', rsuffix='')
            self.catalog = self.catalog.join(
                hosts.catalog.set_index('HaloId'), on='HostHaloId',
                rsuffix='_h')
            if verbose:
                print('Joined hosts in {0:.2f} s'.format(time()-to))
            del hosts
        self._has_host_properties = True
        return

    def siblings(self, trackid, return_value='index'):
        """All subhalos hosted by the same halo at present

        Parameters
        ----------
        trackid : int
            track ID
        return_value : {'index', 'mask', 'table'}, optional
            whether to return the indices, a boolean mask, or the full
            table containing properties of siblings only

        Returns
        -------
        see ``return_value``
        """
        assert trackid // 1 == trackid, 'trackid must be an int'
        _valid_return = ('index','mask','table','trackid')
        assert return_value in _valid_return, \
            'return_value must be one of {0}'.format(_valid_return)
        try:
            idx = self.index(trackid)
        except IndexError:
            return None
        sibling_mask = (self.catalog['HostHaloId'] \
                        == self.catalog['HostHaloId'][idx])
        if sibling_mask.sum() == 0:
            return None
        if return_value == 'mask':
            return sibling_mask
        if return_value == 'track':
            return self.catalog['TrackId'][sibling_mask]
        if return_value == 'index':
            return self._range[sibling_mask]
        return self.catalog[sibling_mask]



class Track(BaseSubhalo):

    def __init__(self, track, sim, as_dataframe=True):
        """
        Parameters
        ----------
        track : output of ``reader.GetTrack``
            track
        sim : ``Simulation`` object or ``str``
            simulation containing the track. If ``str``, should be
            simulation label (see ``Simulation.mapping``)

        Examples
        --------
        To load a subhalo and its track from Aspotle/V1_LR, do
        >>> track = Subhalo(trackid, 'LR')
        or
        >>> track = Subhalo(trackid, Simulation('LR'))
        """
        super(Track, self).__init__(
            track, sim, as_dataframe=as_dataframe)
        self._track = track
        self._trackid = self.track['TrackId']
        self.hostid = self.track['HostHaloId']
        self.current_hostid = self.hostid[-1]
        self._infall_snapshot = None
        self._infall_snapshot_index = None
        self._last_central_snapshot = None
        self._last_central_snapshot_index = None
        self._zcentral = None
        self._zinfall = None

    ### attributes ###

    @property
    def icent(self):
        """alias for ``self.last_central_snapshot_index``"""
        return self.last_central_snapshot_index

    @property
    def last_central_snapshot(self):
        if self._last_central_snapshot is None:
            self._last_central_snapshot = \
                self.track['Snapshot'][self.track['Rank'] == 0][-1]
        return self._last_central_snapshot

    @property
    def last_central_snapshot_index(self):
        if self._last_central_snapshot_index is None:
            self._last_central_snapshot_index = \
                 self._range[self.track['Rank'] == 0][-1]
        return self._last_central_snapshot_index

    @property
    def scale(self):
        return self.track['ScaleFactor']

    @property
    def track(self):
        if self.as_dataframe and isinstance(self._track, np.ndarray):
            self._track = self.DataFrame(self._track)
        elif not self.as_dataframe \
                and isinstance(self._track, pd.DataFrame):
            self._track = self._track.to_records()
        return self._track

    @property
    def trackid(self):
        if hasattr(self._trackid, '__iter__'):
            self._trackid = self._trackid[0]
        return self._trackid

    @property
    def z(self):
        return 1/self.scale - 1

    @property
    def zcentral(self):
        if self._zcentral is None:
            if self.is_central():
                self._zcentral = self.z[-1]
            else:
                self._zcentral = self.z[self.icent]
        return self._zcentral

    ### methods ###

    def host(self, isnap=-1, return_value='trackid'):
        """Host halo (i.e., central subhalo) information at a given
        snapshot

        Parameters
        ----------
        isnap : int, optional
            snapshot number at which the host is to be identified

        Returns
        -------
        host : int
            track ID of the host halo

        Usage
        -----
        The host halo may be loaded as a Track with:
        >>> track = Track(trackid, sim)
        >>> host_track = track.host(isnap)
        >>> host = Track(host_track, sim)
        """
        _valid_return = ('index','mask','table','track','trackid')
        assert return_value in _valid_return, \
            'return_value must be one of {0}'.format(_valid_return)
        hostid = self.hostid[isnap]
        # Rank is necessary for the definition of the Subhalos object
        snap = Subhalos(
            self.reader.LoadSubhalos(
                isnap, ['TrackId','HostHaloId','Rank']),
            self.sim)
        #hostid = snap.host(self.trackid, return_value='trackid')
        return snap.host(self.trackid, return_value=return_value)
        return self.reader.GetTrack(hostid)

        sib = self.siblings(trackid, return_value='mask')
        host_mask = sib & (self.catalog['Rank'] == 0)
        if return_value == 'mask':
            return host_mask
        if return_value == 'track':
            return self.reader.GetTrack(
                self.catalog['TrackId'][host_mask][0])
        if return_value == 'trackid':
            return self.catalog['TrackId'][host_mask][0]
        if return_value == 'index':
            return self._range[host_mask][0]
        return self.catalog[host_mask]

    def is_central(self, isnap):
        """Whether the subhalo is a central at a given moment

        Parameters
        ----------
        isnap : int or array of int
            snapshot index or indices

        Returns
        -------
        is_central : bool or array of bool
            whether the subhalo is a central at the specified time(s)
        """
        return (self.data['Rank'][isnap] == 0)

    def lookback_time(self, z=None, scale=None, isnap=None):
        if z is not None:
            return self.sim.cosmology.lookback_time(z)
        if scale is not None:
            return self.sim.cosmology.lookback_time(1/scale - 1)
        if isnap is not None:
            z = 1/self.track['ScaleFactor'][isnap] - 1
            return self.sim.cosmology.lookback_time(z)
        return self.sim.cosmology.lookback_time(self.z)

    def mergers(self, output='index'):
        """Identify merging events

        A merging event is defined as one in which the host halo at a
        particular snapshot is both different and more massive than the
        host halo at the previous snapshot

        Parameters
        ----------

        """
        return

    def infall(self, return_value='index'):
        """Last redshift at which the subhalo was not in its present
        host

        Parameters
        ----------
        return_value : {'index', 'tlookback', 'redshift'}, optional
            output value. Options are:
        """
        valid_outputs = ('index', 'redshift', 'tlookback')
        assert return_value in valid_outputs, \
            'return_value must be one of {0}. Got {1} instead'.format(
                valid_outputs, return_value)
        hostid = self.host(isnap=-1, return_value='trackid')
        iinf = 0
        # Running backwards because in HBT+ all tracks exist today,
        # at least as 'orphan' galaxies with Nbound=1
        for isnap in self.sim.snapshots[::-1]:
            snapcat = Subhalos(self.reader.LoadSubhalos(isnap), self.sim)
            sib = snapcat.siblings(hostid, 'trackid')
            # this means we reached the beginning of the track
            if sib is None:
                iinf = 0
                break
            if self.trackid not in sib['TrackId']:
                iinf_backward = isnap - self.sim.snapshots.max() - 1
                iinf =  self._range[iinf_backward]
                break
        if return_value == 'tlookback':
            return self.lookback_time(isnap=iinf)
        if return_value == 'redshift':
            return self.z[iinf]
        return iinf
