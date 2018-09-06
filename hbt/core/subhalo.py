from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from numpy.lib.recfunctions import append_fields
import pandas as pd
import six

from HBTReader import HBTReader

from .simulation import Simulation


class BaseSubhalo(object):
    """BaseSubhalo class

    NOTES
        -Probably does not require ``sim`` argument. ``sim._masstype_index``
         can probably be moved out of ``core.Simulation`` as it does not
         depend on simulation
    """

    def __init__(self, catalog, sim, as_dataframe=True):
        """
        Parameters
        ----------
        catalog : ``Track.track`` or ``Subhalos.data``
        sim : ``Simulation``
        """
        self.catalog = catalog
        self.dtype = catalog.dtype
        self._as_dataframe = as_dataframe
        # initialize simulation and reader. This cannot be modified
        # within the same instance
        if isinstance(sim, six.string_types):
            self.sim = Simulation(sim)
        else:
            self.sim = sim
        self.reader = HBTReader(self.sim.path)
        # lazy initialize properties
        self._Mbound = None
        self._MboundType = None
        self._Nbound = None
        self._NboundType = None
        # private attributes
        self.__range = None

    ### private properties ###

    @property
    def _range(self):
        if self.__range is None:
            self.__range = np.arange(self.catalog.size, dtype=int)
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
    def Mbound(self):
        if self._Mbound is None:
            self._Mbound = 1e10 * self.catalog['Mbound']
        return self._Mbound

    @property
    def MboundType(self):
        if self._MboundType is None:
            self._MboundType = 1e10 * self.catalog['MboundType']
        return self._MboundType

    @property
    def Nbound(self):
        if self._Nbound is None:
            self._Nbound = self.catalog['Nbound']
        return self._Nbound

    @property
    def NboundType(self):
        if self._NboundType is None:
            self._NboundType = self.catalog['NboundType']
        return self._NboundType

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
                    df['{0}{1}'.format(name, i+1)] = recarray[name][:,i]
        return pd.DataFrame(df)

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
            self._range[self.subhalos['HostHaloId'] == self._hosthaloid]


class Subhalos(BaseSubhalo):
    """Class to manage a sample of subhalos at a given snapshot

    """

    def __init__(self, subhalos, sim, as_dataframe=True):
        """
        Parameters
        ----------
        subhalos : output of ``reader.LoadSubhalos``
            subhalo sample, defined at a specific snapshot
        sim : ``Simulation`` object or ``str``
            simulation containing the track. If ``str``, should be
            simulation label (see ``Simulation.mapping``)
        """
        super(Subhalos, self).__init__(
            subhalos, sim, as_dataframe=as_dataframe)
        self._subhalos = subhalos
        self._central_idx = None
        self._central_mask = None
        self._centrals = None
        self._orphan = None
        self._satellite_idx = None
        self._satellite_mask = None
        self._satellites = None

    ### attributes ###

    @property
    def centrals(self):
        return self.subhalos[self.central_mask]

    @property
    def central_idx(self):
        if self._central_idx is None:
            self._central_idx = self._range[self._central_mask]
        return self._central_idx

    @property
    def central_mask(self):
        if self._central_mask is None:
            self._central_mask = (self.subhalos['Rank'] == 0)
        return self._central_mask

    @property
    def orphan(self):
        """boolean mask, True if object is orphan"""
        if self._orphan is None:
            self._orphan = (self.subhalos['Nbound'] == 1)
        return self._orphan

    @property
    def satellites(self):
        return self.subhalos[self.satellite_mask]

    @property
    def satellite_idx(self):
        if self._satellite_idx is None:
            self._satellite_idx =  self._range[self.satellite_mask]
        return self._satellite_idx

    @property
    def satellite_mask(self):
        if self._satellite_mask is None:
            self._satellite_mask = (self.subhalos['Rank'] > 0)
        return self._satellite_mask

    @property
    def subhalos(self):
        if self.as_dataframe and isinstance(self._subhalos, np.ndarray):
            self._subhalos = self.DataFrame(self._subhalos)
        elif not self.as_dataframe \
                and isinstance(self._subhalos, pd.DataFrame):
            self._subhalos = self._subhalos.to_records()
        self._subhalos = self._subhalos[self._subhalos['HostHaloId'] > -1]
        return self._subhalos

    ### methods ###

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
        sibling_mask = (self.subhalos['HostHaloId'] \
                        == self.subhalos['HostHaloId'][idx])
        if sibling_mask.sum() == 0:
            return None
        if return_value == 'mask':
            return sibling_mask
        if return_value == 'track':
            return self.subhalos['TrackId'][sibling_mask]
        if return_value == 'index':
            return self._range[sibling_mask]
        return self.subhalos[sibling_mask]

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
            host_mask = sib & (self.subhalos['Rank'] == 0)
        if return_value == 'mask':
            return host_mask
        if return_value == 'track':
            return self.reader.GetTrack(
                self.subhalos['TrackId'][host_mask][0])
        if return_value == 'trackid':
            return self.subhalos['TrackId'][host_mask][0]
        if return_value == 'index':
            return self._range[host_mask][0]
        return self.subhalos[host_mask]

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
            return self._range[self.subhalos['TrackId'] == trackid]
        return self._range[self.subhalos['TrackId'] == trackid][0]

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
        cent = (self.data['Rank'][self.data['TrackId'] == trackId] == 0)
        if hasattr(trackid, '__iter__'):
            return cent
        return cent[0]

    def _distance(self, group, axes='xyz', rank_key='Rank'):
        xyz = np.array([group[i] for i in axes]).T
        return np.sum((xyz-xyz[group[rank_key] == 0])**2, axis=1)**0.5

    def distance2host(self, projection='xyz', append_key='HostHaloDistance',
                      position_name='ComovingMostBoundPosition', read=True):
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
            ``self.subhalos``
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
        sub = self.subhalos
        aux = sub.join(
            sub[sub['Rank'] == 0][columns].set_index('HostHaloId'),
            on='HostHaloId', rsuffix='_h')
        #self.subhalos['HostHaloDistance'] = np.sum(
        dist = np.sum(
            (aux[columns[:-1]].values - aux[auxcols].values)**2, axis=1)**0.5
        if append_key:
            self.subhalos[append_key] = dist
        else:
            return dist



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
        host_mask = sib & (self.subhalos['Rank'] == 0)
        if return_value == 'mask':
            return host_mask
        if return_value == 'track':
            return self.reader.GetTrack(
                self.subhalos['TrackId'][host_mask][0])
        if return_value == 'trackid':
            return self.subhalos['TrackId'][host_mask][0]
        if return_value == 'index':
            return self._range[host_mask][0]
        return self.subhalos[host_mask]

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
