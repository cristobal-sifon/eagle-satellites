from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from astropy.io import fits
from astropy.table import Table
import h5py
from itertools import combinations, count
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.recfunctions import append_fields
import os
import pandas as pd
import six
import sys
from time import time
import warnings

from HBTReader import HBTReader

from .simulation import BaseSimulation, Simulation


class BaseDataSet(object):

    def __init__(self, catalog, as_dataframe=True):
        self._catalog = catalog
        self._as_dataframe = as_dataframe
        self._none_value = 999

    ### private properties ###

    @property
    def _range(self):
        #return np.arange(self.catalog.size, dtype=int)
        return np.arange(self.catalog[self.colnames[0]].size, dtype=int)

    ### public properties ###

    @property
    def as_dataframe(self):
        return self._as_dataframe

    @as_dataframe.setter
    def as_dataframe(self, as_df):
        assert isinstance(as_df, bool), 'as_dataframe must be boolean'
        self._as_dataframe = as_df

    @property
    def catalog(self):
        if self.as_dataframe and not isinstance(self._catalog, pd.DataFrame):
            self._catalog = self.DataFrame(self._catalog)
        elif not self.as_dataframe \
                and isinstance(self._catalog, pd.DataFrame):
            self._catalog = self._catalog.to_records()
        return self._catalog

    #@catalog.setter
    #def catalog(self, cat):
        #self._catalog = cat

    @property
    def colnames(self):
        if self.as_dataframe:
            return list(self.catalog)
        else:
            return self.catalog.dtype.names

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

    def __init__(self, catalog, sim, as_dataframe=True, pvref='MostBound'):
        """
        Parameters
        ----------
        catalog : ``Track.track`` or ``Subhalos.data``
        sim : ``Simulation``
        """
        assert pvref in ('Average', 'MostBound')
        # setting to False here to load the Mbounds and Nbounds below
        # (easier to load from hdf5 than DataFrame)
        #super(BaseSubhalo, self).__init__(catalog, as_dataframe=False)
        super(BaseSubhalo, self).__init__(catalog, as_dataframe=as_dataframe)
        BaseSimulation.__init__(self, sim)
        #self.Mbound = 1e10 * self.catalog['Mbound']
        #self.MboundType = 1e10 * self.catalog['MboundType']
        #self.Nbound = self.catalog['Nbound']
        #self.NboundType = self.catalog['NboundType']
        self.as_dataframe = as_dataframe
        self.pvref = pvref

    ### hidden properties ###

    @property
    def _valid_axes(self):
        return ('', '0', '1', '2', '01', '02', '12')

    ### properties ###

    @property
    def Mbound(self):
        return 1e10 * self.catalog['Mbound']

    @property
    def MboundType(self):
        if self.as_dataframe:
            cols = ['MboundType{0}'.format(i) for i in range(6)]
            return 1e10 * np.array(self.catalog[cols])
        return 1e10 * self.catalog['MboundType']

    @property
    def mvir(self):
        return self.catalog['MVir']

    @property
    def Nbound(self):
        return self.catalog['Nbound']

    @property
    def NboundType(self):
        if self.as_dataframe:
            cols = ['NboundType{0}'.format(i) for i in range(6)]
            return np.array(self.catalog[cols])
        return self.catalog['NboundType']


    ### hidden methods ###

    def _get_ax1d_label(self, ax):
        try:
            ax = int(ax)
        except ValueError as err:
            msg = 'ax must be an int, 0 <= ax <= 2'
            raise ValueError(err)
        assert ax in [0, 1, 2]
        return 'xyz'[[0, 1, 2].index(ax)]

    ### methods ###

    def mass(self, mtype=None, index=None):
        assert mtype is not None or index is not None, \
            'must provide either ``mtype`` or ``index``'
        if mtype is not None:
            if mtype.lower() in ('total', 'mbound'):
                return self.Mbound
            return self.MboundType[:,self.sim._masstype_index(mtype)]
        if index == -1:
            return np.array(self.Mbound)
        return self.MboundType[:,index]

    def nbound(self, mtype=None, index=None):
        assert mtype is not None or index is not None, \
            'must provide either ``mtype`` or ``index``'
        if mtype is not None:
            if mtype.lower() in ('total', 'nbound'):
                return self.Nbound
            return self.NboundType[:,self.sim._masstype_index(mtype)]
        if index == -1:
            return np.array(self.Nbound)
        return self.NboundType[:,index]

    ## easy access to positions, distances and velocities ##

    def column(self, value, ax):
        # consider moving `ref` to a class attribute
        value = value.lower()
        acceptable = ('distance', 'position', 'velocity')
        assert value in acceptable, \
            '`value` must be one of {0}'.format(acceptable)
        if value == 'distance':
            return dcol(ax)
        if value == 'position':
            return pcol(ax)
        if value == 'velocity':
            return vcol(ax)

    def dcol(self, ax='', frame='Physical'):
        """Distance column name

        Parameters
        ----------
        ax : {'0', '1', '2', '01', '02', '12'}, optional
            distance axis or plane. If not given, will return
            3-dimenional distance column

        Returns
        -------
        dcol : str
            distance column name
        """
        assert ax in ('', '0', '1', '2', '01', '02', '12')
        return '{2}{0}Distance{1}'.format(self.pvref, ax, frame.capitalize())

    def dcols(self, ndim=1, frame='Physical'):
        """Distance column names

        Parameters
        ----------
        ndim : int {1,2, 3}, optional
            number of dimensions over which distances are desired
        """
        assert ndim in (1, 2, 3)
        if ndim == 3:
            return self.dcol(frame=frame)
        return [self.dcol(''.join(plane), frame=frame)
                for plane in combinations('012', ndim)]

    def dlabel(self, ax=''):
        ax = str(ax)
        assert ax in self._valid_axes, \
            'ax must be one of {0}'.format(self._valid_axes)
        if ax != 0 and not ax:
            return 'r_\mathrm{3D}'
        return 'r_{{{0}}}'.format(
            ''.join([self._get_ax1d_label(i) for i in ax]))

    def distance(self, ax=''):
        return self.catalog[self.dcol(ax=ax)]

    def pcol(self, ax, frame='Comoving'):
        assert int(ax) in [0,1,2]
        return '{2}{0}Position{1}'.format(self.pvref, ax, frame)

    def pcols(self, frame='Comoving'):
        """All position column names"""
        return [self.pcol(i, frame) for i in range(3)]

    def plabel(self, ax):
        return self._get_ax1d_label(ax)

    def position(self, ax):
        return self.catalog[self.pcol(ax)]

    def scol(self, ax=''):
        assert not ax or int(ax) in [0,1,2]
        return 'Physical{0}HostVelocityDispersion{1}'.format(self.pvref, ax)

    def scols1d(self):
        return [self.scol(i) for i in range(3)]

    def slabel(self, ax=''):
        assert not ax or int(ax) in [0,1,2]
        if ax != 0 and not ax:
            return r'\sigma_\mathrm{3D}'
        return r'\sigma_{0}'.format(self._get_ax1d_label(ax))

    def sigma(self, ax=''):
        """Velocity dispersion"""
        return self.catalog[self.scol(ax)]

    def vcol(self, ax='', peculiar=True):
        assert not ax or int(ax) in [0,1,2]
        if peculiar:
            return 'Physical{0}PeculiarVelocity{1}'.format(self.pvref, ax)
        return 'Physical{0}Velocity{1}'.format(self.pvref, ax)

    def vcols1d(self):
        return [self.vcol(i) for i in range(3)]

    def vlabel(self, ax):
        assert not ax or int(ax) in [0,1,2]
        if ax != 0 and not ax:
            return r'v_\mathrm{3D}'
        return r'v_{0}'.format(self._get_ax1d_label(ax))

    def velocity(self, ax=''):
        return self.catalog[self.vcol(ax=ax)]


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

    def __init__(self, catalog, sim, isnap,
                 logMmin=9, logM200Mean_min=12, as_dataframe=True,
                 exclude_non_FoF=True,
                 load_hosts=True, load_distances=True, load_velocities=True):
        """
        Parameters
        ----------
        catalog : output of ``reader.LoadSubhalos``
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
        assert isinstance(as_dataframe, bool)
        assert isinstance(load_hosts, bool)
        assert isinstance(load_distances, bool)
        assert isinstance(load_velocities, bool)
        super(Subhalos, self).__init__(
              catalog, sim, as_dataframe=as_dataframe)
        #print('{0} objects in the full catalog'.format(
            #catalog['TrackId'].size))
        self.exclude_non_FoF = exclude_non_FoF
        self.non_FoF = (self.catalog['HostHaloId'] == -1)# \
                        #| np.isnan(self.catalog['HostHaloId']))
        if self.exclude_non_FoF:
            #print('Excluding {0} non-FoF subhalos'.format(self.non_FoF.sum()))
            self._catalog = self.catalog[~self.non_FoF]
        if 'Mbound' in self.colnames:
            self.logMmin = logMmin
            self._catalog = self.catalog[self.mass('total') > 10**self.logMmin]
        else:
            self.logMmin = 0
            #warnings.warn('No Mbound column. Not applying Mbound cut')
        if 'M200Mean' in self.colnames:
            self.logM200Mean_min = logM200Mean_min
            mask = (self.catalog['M200Mean'] > 10**self.logM200Mean_min)
            self._catalog = self.catalog[mask]
        if 'Nbound' in self.colnames:
            if self.as_dataframe:
                self.catalog['IsDark'] = (self.nbound('stars') == 0)
            else:
                self._catalog = append_fields(
                    self.catalog, 'IsDark', (self.nbound('stars') == 0))
        self.isnap = isnap
        #self.redshift = self.redshift[self.isnap]
        self._central_idx = None
        self._central_mask = None
        self._centrals = None
        self._orphan = None
        self._satellite_idx = None
        self._satellite_mask = None
        self._satellites = None
        self._has_host_properties = False
        self._has_distances = False
        self._has_velocities = []
        self.load_hosts = load_hosts
        if self.load_hosts:
            self.host_properties()
        else:
            load_distances = False
            load_velocities = False
        self.load_distances = load_distances
        self.load_velocities = load_velocities
        if self.load_distances:
            #self.distance2host('Physical')
            self.distance2host('Comoving')
        if self.load_velocities:
            self.host_velocities()

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

    ### hidden methods ###

    ### methods ###

    def distance2host(self, frame='Physical', verbose=True):
        """Calculate the distance of all subhalos to the center of
        their host

        Parameters
        ----------

        Returns
        -------
        dist2host : float array
            distances to host in Mpc, in 3d or along the given
            projection
        """
        #if self._has_distances:
            #print('Distances already calculated')
            #return
        print('Calculating distances...')
        input_fmt = self.as_dataframe
        self.as_dataframe = True
        # alias
        sub = self.catalog
        columns = list(np.append(self.pcols(), ['HostHaloId','Rank']))
        to = time()
        hosts = sub[columns].join(
            sub[columns][sub['Rank'] == 0].set_index('HostHaloId'),
            on='HostHaloId', rsuffix='_h')
        print('hosts:', np.sort(hosts.columns))
        j = (hosts['HostHaloId'] > 100) & (hosts['HostHaloId'] <= 102)
        #print('j =', j.sum())
        """
        tbl = Table.from_pandas(hosts[j])
        cols = [fits.Column(name=key, array=tbl[key]) for key in tbl.colnames]
        hdu = fits.BinTableHDU.from_columns(cols)
        hdu.writeto('test_hosthalo.fits')
        """
        print('Joined hosts in {0:.2f} min'.format((time()-to)/60))
        # 1d
        ti = time()
        print('1d:')
        print(self.pcols())
        for dcol, pcol in zip(self.dcols(1, frame), self.pcols(frame)):
            print(dcol, pcol)
            self.catalog[dcol] = ((hosts[pcol] - hosts[pcol+'_h'])**2)**0.5
            print('percentiles:', np.percentile(self.satellites[dcol],
                  [0,1,25,50,99,100]))
        #print(hosts[['HostHaloId','HostHaloId_h','Rank','Rank_h']][j])
        #print(hosts[['ComovingMostBoundPosition0'[j])
        print('1d distances in {0:.2f} s'.format(time()-ti))
        # 2d
        ti = time()
        for dcol in self.dcols(2, frame):
            dcols = [self.dcol(dcol[-i], frame) for i in (2,1)]
            self.catalog[dcol] = np.sum(
                self.catalog[dcols]**2, axis=1)**0.5
            print(dcol)
            print('percentiles:', np.percentile(self.satellites[dcol],
                  [0,1,25,50,99,100]))
        print('2d distances in {0:.2f} s'.format(time()-ti))
        # 3d
        ti = time()
        self.catalog[self.dcol(frame=frame)] = np.sum(
            self.catalog[self.dcols(frame=frame)]**2, axis=1)**0.5
        print('3d distances in {0:.2f} s'.format(time()-ti))
        print('percentiles:',
              np.percentile(self.satellites[self.dcol(frame=frame)],
              [0,1,25,50,99,100]))
        if verbose:
            print('Calculated distances in {0:.2f} s'.format(time()-to))
        self.as_dataframe = input_fmt
        self._has_distances = True
        return

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
                np.array(self.catalog['TrackId'][host_mask])[0])
        if return_value == 'trackid':
            return np.array(self.catalog['TrackId'][host_mask])[0]
        if return_value == 'index':
            return self._range[host_mask][0]
        return np.array(self.catalog[host_mask])

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
        return self._mass_weighted_stat(vxyz, np.array(x[mcol]), label)**0.5

    def _mass_weighted_std(self, x, cols, mcol, label='sigma_cl'):
        cols = [cols] if isinstance(cols, six.string_types) else cols
        if x[cols[0]].count() == 1:
            return pd.Series({label: 0}, index=[label])
        # this is correct only if they are peculiar velocities
        sigma_xyz = np.sum([x[i]**2 for i in cols], axis=0)
        return self._mass_weighted_stat(
            sigma_xyz, np.array(x[mcol]), label)**0.5

    def host_velocities(self, mass_weighting=None):
        # note that this screws things up if self.pvref changes
        if mass_weighting in self._has_velocities:
            print('velocities already loaded')
            return
        if not self._has_host_properties:
            self.host_properties()
        print('Calculating velocities...')
        to = time()
        adf = self.as_dataframe
        self.as_dataframe = True
        # alias
        cx = self.catalog
        axes = 'xyz'
        vcols = ['Physical{1}Velocity{0}'.format(i, self.pvref)
                 for i in range(3)]
        if mass_weighting is None:
            mweight = 1
        else:
            mcol = self.sim.masstype_pandas_column(mass_weighting)
            mweight = cx[mcol]
        keys = list(cx)
        new_keys = []
        # mean velocities
        grcols = ['HostHaloId', 'Nsat']
        skip = len(grcols)
        mvcols = []
        for i, vcol in enumerate(vcols):
            mvcols.append('mv{0}'.format(i))
            cx[mvcols[-1]] = cx[vcol] * mweight
        grcols = np.append(grcols, mvcols)
        group = cx[grcols].groupby('HostHaloId')
        if mass_weighting is None:
            wmean = group[mvcols].mean()
        else:
            wmean = group[mvcols].sum()
        if mass_weighting is None:
            #msum = group['Nsat']
            msum = 1
        else:
            msum = group[mcol].sum()
        ## host mean velocities
        hosts = pd.DataFrame({'HostHaloId': np.array(group.size().index)})
        vhcol = 'Physical{0}HostMeanVelocity'.format(self.pvref)
        vhcols = [vhcol+str(i) for i in range(3)]
        for i, mvcol in enumerate(mvcols):
            hosts[vhcol+str(i)] = wmean[mvcol] / msum
        hosts[vhcol] = np.sum(wmean[mvcols]**2, axis=1)**0.5
        ## velocity dispersions
        print('velocity dispersions...')
        ti = time()
        hostkeys = np.append(['HostHaloId', vhcol], vhcols)
        cx = cx.join(hosts[hostkeys].set_index('HostHaloId'), on='HostHaloId',
                     rsuffix='_h')
        for i, vcol in enumerate(vcols):
            vdiff = cx[vcol] \
                - cx['Physical{0}HostMeanVelocity{1}'.format(self.pvref, i)]
            cx[mvcols[i]] = mweight * vdiff**2
        # group again because I want the new mvcols grouped as well
        group = cx.groupby('HostHaloId')
        wvar = group[mvcols].sum()
        scol = 'Physical{0}HostVelocityDispersion'.format(self.pvref)
        scols = []
        for i, mvcol in enumerate(mvcols):
            scols.append(scol+str(i))
            hosts[scols[-1]] = (wvar[mvcol] / msum)**0.5
        hosts[scol] = np.sum(wvar/msum, axis=1)**0.5
        new_keys = np.append(['HostHaloId', scol], scols)
        cx = cx.join(hosts[new_keys].set_index('HostHaloId'), on='HostHaloId')
        # -1 or not?
        for col in np.append(scol, scols):
            cx[col] = cx[col] / (cx['Nsat']-1)**0.5
        print('dispersions in {0:.1f} seconds'.format(time()-ti))
        # peculiar velocities
        ti = time()
        vpcol = 'Physical{0}PeculiarVelocity'.format(self.pvref)
        vpcols = [vpcol+str(i) for i in range(3)]
        for col in (vhcol, vhcols, scol, scols):
            new_keys = np.append(new_keys, col)
        for i in range(3):
            cx[vpcol+str(i)] = \
                cx['Physical{0}Velocity{1}'.format(self.pvref, i)] \
                - cx['Physical{0}HostMeanVelocity{1}'.format(self.pvref, i)]
        # 3d
        cx['Physical{0}Velocity'.format(self.pvref)] = \
            np.sum(cx[self.vcols1d()]**2, axis=1)**0.5 \
            * np.sign(np.sum(cx[self.vcols1d()], axis=1))
        cx[vpcol] = cx['Physical{0}Velocity'.format(self.pvref)] \
            - cx['Physical{0}HostMeanVelocity'.format(self.pvref)]
        print('peculiar velocities in {0:.2f} min'.format((time()-ti)/60))
        cx.drop(columns=mvcols)
        self._catalog = cx
        self._has_velocities.append(mass_weighting)
        self.as_dataframe = adf
        print('Calculated velocities in {0:.2f} min'.format((time()-to)/60))
        print()
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
        """
        assert trackid // 1 == trackid, \
            'trackid must be an int or an array of int'
        # in case they're floats of ints
        trackid = np.array(trackid, dtype=int)
        cent = (self.data['Rank'][self.data['TrackId'] == trackId] == 0)
        if hasattr(trackid, '__iter__'):
            return cent
        return cent[0]

    def host_properties(self, force_isnap=False, verbose=False):
        """Load halo masses and sizes into the subhalo data

        See `HostHalos` for details

        Creates two new attributes:
            -`Nsat`: number of satellite subhalos
            -`Ndark`: number of dark subhalos. Note that this includes
                the central subhalo
        """
        if self._has_host_properties:
            print('Hosts already loaded')
            return
        print('Loading hosts...')
        to = time()
        adf = self.as_dataframe
        self.as_dataframe = True
        if self.isnap not in self.sim.virial_snapshots:
            hosts = HostHalos(self.sim, self.isnap, force_isnap=force_isnap)
            ti = time()
            # number of star particles, to identify dark subhalos
            cols = ['HostHaloId']
            if 'IsDark' in self.colnames:
                cols.append('IsDark')
            grouped = self.catalog[cols].groupby('HostHaloId')
            nmdict = {'Nsat': grouped.size()-1}
            if 'IsDark' in self.colnames:
                 nmdict['Ndark'] = grouped['IsDark'].sum()
            nm = pd.DataFrame(nmdict)
            self._catalog = self.catalog.join(nm, on='HostHaloId', rsuffix='_h')
            self._catalog = self.catalog.join(
                hosts.catalog, on='HostHaloId', rsuffix='_h')
            # update host masses
            for col in list(self.catalog):
                if 'M200' in col or col == 'MVir':
                    # otherwise I think this happens twice?
                    if self.catalog[col].max() < 1e10:
                        self.catalog[col] = 1e10 * self.catalog[col]
            if verbose:
                print('Joined hosts in {0:.2f} s'.format(time()-to))
            del hosts
        self._has_host_properties = True
        self.as_dataframe = adf
        print('Loaded in {0:.2f} s'.format(time()-to))
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
        _valid_return = ('index','mask','table','track','trackid')
        assert return_value in _valid_return, \
            'return_value must be one of {0}'.format(_valid_return)
        try:
            idx = self.index(trackid)
        except IndexError as e:
            print('IndexError:', e)
            return None
        hostids = np.array(self.catalog['HostHaloId'])
        sibling_mask = (hostids == hostids[idx])
        if sibling_mask.sum() == 0:
            return None
        if return_value == 'mask':
            return sibling_mask
        if return_value == 'track':
            return self.catalog[sibling_mask]
        if return_value == 'trackid':
            return np.array(self.catalog['TrackId'])[sibling_mask]
        if return_value == 'index':
            return self._range[sibling_mask]



class _Track(BaseSubhalo):

    def __init__(self, track, sim, as_dataframe=False):
        """
        Discontinued. Please use ``track.Track`` instead.

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
        #print(as_dataframe)
        #print(track.dtype)
        #print(track.dtype[0])
        #print(track.dtype[0][:4])
        #print()
        #if isinstance(track, np.ndarray) and as_dataframe:
            #track = pd.DataFrame.from_records(track, columns=track.dtype.names)
        self._track = track
        #print(track.dtype)
        #print(type(track))
        #print(type(self._track))
        #print(np.sort(self._track.columns))
        self._trackid = self._track['TrackId']
        self.hostid = np.array(self._track['HostHaloId'])
        self.current_hostid = self.hostid[-1]
        self._first_satellite_snapshot = self._none_value
        self._first_satellite_snapshot_index = self._none_value
        self._last_central_snapshot = self._none_value
        self._last_central_snapshot_index = self._none_value
        self._zcentral = None
        self._zinfall = None

    def __str__(self):
        return f'Track ID {self.trackid} in {self.sim.name}'

    ### attributes ###

    @property
    def first_satellite_snapshot(self):
        """
        If the track has never been a satellite, this will remain None
        """
        if self.first_satellite_snapshot_index == self._none_value:
            return self._none_value
        if self._first_satellite_snapshot == self._none_value:
            self._first_satellite_snapshot = \
                self.track['Snapshot'][self.first_satellite_snapshot_index]
        return self._first_satellite_snapshot

    @property
    def first_satellite_snapshot_index(self):
        """
        If the track has never been a satellite, this will remain None
        """
        if self._first_satellite_snapshot_index == self._none_value:
            sat = (self.track['Rank'] > 0)
            if sat.sum() > 0:
                self._first_satellite_snapshot_index = self._range[sat][0]
        return self._first_satellite_snapshot_index

    @property
    def icent(self):
        """alias for ``self.last_central_snapshot_index``"""
        return self.last_central_snapshot_index

    @property
    def last_central_snapshot(self):
        """
        If the track has never been a central, this will remain None
        """
        if self.last_central_snapshot_index == self._none_value:
            return self._none_value
        if self._last_central_snapshot == self._none_value:
            self._last_central_snapshot = \
                self.track['Snapshot'][self.last_central_snapshot_index]
        return self._last_central_snapshot

    @property
    def last_central_snapshot_index(self):
        """
        If the track has never been a central, this will remain None
        """
        if self._last_central_snapshot_index == self._none_value:
            cent = (self.track['Rank'] == 0)
            if cent.sum() > 0:
                self._last_central_snapshot_index = self._range[cent][-1]
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
                isnap, ['TrackId','HostHaloId','Rank','Mbound']),
            self.sim, isnap, load_hosts=False)
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

    def lookback_time(self, z=None, scale=None, isnap=None,
                     include_units=False):
        """
        if z is not None:
            t = self.sim.cosmology.lookback_time(z)
        elif scale is not None:
            t = self.sim.cosmology.lookback_time(1/scale - 1)
        elif isnap is not None:
            z = 1/self.track['ScaleFactor'][isnap] - 1
            return self.sim.cosmology.lookback_time(z)
        return self.sim.cosmology.lookback_time(self.z)
        """
        if z is None:
            if scale is not None:
                z = 1/scale - 1
            elif isnap is not None:
                z = 1/self.track['ScaleFactor'][isnap] - 1
            else:
                z = self.z
        t = self.sim.cosmology.lookback_time(z)
        if not include_units:
            t = t.value
        return t

    def mergers(self, output='index'):
        """Identify merging events

        A merging event is defined as one in which the host halo at a
        particular snapshot is both different and more massive than the
        host halo at the previous snapshot

        Parameters
        ----------

        """
        return

    def infall(self, return_value='index', min_snap_range_brute=3):
        """Last redshift at which the subhalo was not in its present
        host

        **Should be able to read the infall file if it exists**

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
        # first jump by halves until we've narrowed it down to
        # very few snapshots
        imin = self.sim.snapshots.min()
        imax = self.sim.snapshots.max()
        do_smart = True
        if do_smart:
            while imax - imin > min_snap_range_brute:
                isnap = (imin+imax) // 2
                subs = self.reader.LoadSubhalos(
                    isnap, ['TrackId','HostHaloId'])
                if len(subs) == 0:
                    imin = isnap
                    continue
                snapcat = Subhalos(
                    subs, self.sim, isnap, load_hosts=False, logMmin=0)
                sib = snapcat.siblings(hostid, 'trackid')
                if sib is None or self.trackid not in sib:
                    imin = isnap
                else:
                    imax = isnap
        # once we've reached the minimum range allowed above,
        # we just do backwards brute-force
        for isnap in range(imax, imin-1, -1):
            subs = self.reader.LoadSubhalos(
                isnap, ['TrackId','HostHaloId'])
            if len(subs) == 0:
                iinf_backward = 0
                break
            snapcat = Subhalos(
                subs, self.sim, isnap, load_hosts=False, logMmin=0)
            sib = snapcat.siblings(hostid, 'trackid')
            # this means we reached the beginning of the track
            if sib is None:
                iinf_backward = 0
                break
            elif self.trackid not in sib:
                iinf_backward = isnap - self.sim.snapshots.max()
                break
        else:
            iinf_backward = 0
        iinf =  self._range[iinf_backward]
        if return_value == 'tlookback':
            return self.lookback_time(isnap=iinf)
        if return_value == 'redshift':
            return self.z[iinf]
        return iinf
