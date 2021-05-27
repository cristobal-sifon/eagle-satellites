from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from icecream import ic
import numpy as np
import six

from HBTReader import HBTReader

from .simulation import BaseSimulation, Simulation
from .subhalo import BaseSubhalo, Subhalos


class TrackArray(BaseSubhalo):

    def __init__(self, track_ids, sim):
        if not np.iterable(track_ids):
            track_ids = np.array([track_ids])
        self.track_ids = track_ids
        # load simulation
        if isinstance(sim, six.string_types):
            self.sim = Simulation(sim)
        else:
            self.sim = sim
        super(TrackArray, self).__init__(self.track_ids, self.sim)

    def infalls(self):
        return


class Track(BaseSubhalo):

    def __init__(self, trackid, sim):
        """
        Parameters
        ----------
        trackid : int or output of ``reader.GetTrack``
            ID of the track to be loaded or the track itself
        sim : ``Simulation`` object or ``str``
            simulation containing the track. If ``str``, should be
            simulation label (see ``Simulation.mapping``)

        To load a track from Aspotle/V1_LR, do
        >>> track = Track(trackid, 'LR')
        or
        >>> track = Track(trackid, Simulation('LR'))

        """
        # load simulation
        if isinstance(sim, six.string_types):
            self.sim = Simulation(sim)
        else:
            self.sim = sim
        # load reader and track
        self.reader = HBTReader(self.sim.path)
        if isinstance(trackid, (int,np.integer)):
            self.trackid = trackid
            self.track = self._load_track()
        elif isinstance(trackid, np.ndarray):
            self.track = trackid
            self.trackid = self.track['TrackId']
            if hasattr(self.trackid, '__iter__'):
                self.trackid = self.trackid[0]
        else:
            msg = f'argument track {trackid} of wrong type ({type(trackid)})'
            raise ValueError(msg)
        super(Track, self).__init__(self.track, self.sim)
        # other attributes
        #self.current_host = self.hosts[-1]
        self.hostid = np.array(self.track['HostHaloId'])
        self.scale = self.track['ScaleFactor']
        self.z = 1/self.scale - 1
        self._first_satellite_snapshot = self._none_value
        self._first_satellite_snapshot_index = self._none_value
        self._infall_snapshot = None
        self._infall_snapshot_index = None
        self._last_central_snapshot = None
        self._last_central_snapshot_index = None
        #self._Mbound = None
        #self._MboundType = None
        self._zcentral = None
        self._zinfall = None

    def __repr__(self):
        return f'Track({self.trackid}, {self.sim.label})'

    def __str__(self):
        return f'Track ID {self.trackid} in {self.sim.name}'

    @property
    def icent(self):
        """alias for ``self.last_central_snapshot_index``"""
        return self.last_central_snapshot_index

    @property
    def isat(self):
        """alias for ``self.first_satellite_snapshot_index``"""
        return self.first_satellite_snapshot_index

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
    def infall_snapshot(self):
        if self._infall_snapshot is None:
            self._infall_snapshot = \
                self.track['Snapshot'][self.infall_snapshot_index]
        return self._infall_snapshot

    @property
    def infall_snapshot_index(self):
        if self._infall_snapshot_index is None:
            self._infall_snapshot_index \
                = self._range[self.host != self.current_host][-1]
        return self._infall_snapshot_index

    @property
    def last_central_snapshot(self):
        if self._last_central_snapshot is None:
            #key = 'SnapshotIndexOfLastIsolation'
            #self._last_central_snapshot = \
                #self.reader.GetSub(self.trackid)[key]
            self._last_central_snapshot = \
                self.track['Snapshot'][self.track['Rank'] == 0][-1]
        return self._last_central_snapshot


    @property
    def last_central_snapshot_index(self):
        if self._last_central_snapshot_index is None:
            #self._last_central_snapshot_index = \
                #self._range[self.track['Snapshot'] == \
                            #self.last_central_snapshot][0]
            self._last_central_snapshot_index = \
                self._range[self.track['Rank'] == 0][-1]
        return self._last_central_snapshot_index

    """
    @property
    def Mbound(self):
        if self._Mbound is None:
            self._Mbound = 1e10 * self.track['Mbound']
        return self._Mbound

    @property
    def MboundType(self):
        if self._MboundType is None:
            self._MboundType = 1e10 * self.track['MboundType']
        return self._MboundType
    """

    @property
    def zcentral(self):
        if self._zcentral is None:
            if self.track['Rank'][-1] == 0:
                self._zcentral = self.z[-1]
            else:
                self._zcentral = \
                    1/self.track['ScaleFactor'][self._last_central_snapshot_index] - 1
        return self._zcentral

    @property
    def zinfall(self):
        if self._zinfall is None:
            self._zinfall = self.z[self.host != self.current_host][-1]
        return self._zinfall

    def _load_track(self):
        """Load track, accounting for missing snapshots"""
        track = self.reader.GetTrack(self.trackid)
        return track

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

    def lookback_time(self, z=None, scale=None, snapshot=None):
        """This should probably be in Simulation"""
        if z is not None:
            return self.sim.cosmology.lookback_time(z)
        if scale is not None:
            return self.sim.cosmology.lookback_time(1/scale - 1)
        if snapshot is not None:
            z = 1/self.track['ScaleFactor'][snapshot] - 1
            return self.sim.cosmology.lookback_time(z)
        return self.sim.cosmology.lookback_time(self.z)

    """
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
    """

    def _mergers(self, output='index'):
        """Identify merging events

        A merging event is defined as one in which the host halo at a
        particular snapshot is both different and more massive than the
        host halo at the previous snapshot

        Parameters
        ----------

        """
        return
