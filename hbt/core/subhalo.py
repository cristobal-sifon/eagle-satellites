from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import six

from HBTReader import HBTReader

from .simulation import Simulation


class BaseSubhaloSample():

    def __init__(self, catalog, sim):
        """
        Parameters
        ----------
        catalog : ``Track.track`` or ``SubhaloSample.data``
        sim : ``Simulation``
        """
        self.catalog = catalog
        self.sim = sim
        self._Mbound = None
        self._MboundType = None
        self.hosts = self.catalog['HostHaloId']
        # private attributes
        self.__range = None

    @property
    def _range(self):
        if self.__range is None:
            self.__range = np.arange(self.catalog.size, dtype=int)
        return self.__range

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


class SubhaloSample(BaseSubhaloSample):
    """Class to manage a sample of subhalos at a given snapshot

    """

    def __init__(self, data, sim):
        """Define a SubhaloSample object

        Note that a ``SubhaloSample`` object is defined within the
        snapshot where the subhalo catalog was defined, and contains
        information about that snapshot only

        Parameters
        ----------
        data : output of ``reader.LoadSubhalos``
        sim : ``Simulation``
        """
        self.data = data
        # load simulation
        if isinstance(sim, six.string_types):
            self.sim = Simulation(sim)
        else:
            self.sim = sim
        super(SubhaloSample, self).__init__(self.data, self.sim)

    def host_track(self, subtrackid):
        """Return the TrackId of the host halo

        Parameters
        ----------
        subtrackid : int
            subhalo track ID, for which to find the host

        Returns
        -------
        host_track : int
            track ID of the host halo
        """
        return self.data['TrackId'][self.siblings(subtrackid) \
                                    & (self.data['Rank'] == 0)][0]

    def index(self, trackid):
        """Index of a given trackid in the subhalo catalog"""
        return self._range[self.data['TrackId'] == trackid][0]

    def siblings(self, trackid, same_depth=False):
        """Siblings of a given subhalo

        Find all subhalos hosted by the same halo as a given subhalo by
        matching their HostHaloId.

        Parameters
        ----------
        trackid : int
            TrackId of the reference subhalo
        same_depth : bool, optional
            Whether to only return subhalos of the same depth as the
            reference, or all subhalos regardless of their depth

        Returns
        -------
        siblings : array of int
            Indices of siblings in ``self.data``

        Examples
        --------
        To idenfity all the siblings of a ``Track`` object at snapshot
        ``isnap``, do:
        >>> track = Track(trackid, sim)
        >>> subs = Subhalo(reader.LoadSubhalos(isnap), sim)
        >>> isiblings = subs.siblings(track.trackid)
        >>> sibling_tracks = subs.data['TrackId'][isiblings]
        """
        idx = self.index(trackid)
        mask = (self.hosts == self.hosts[idx])
        if same_depth:
            mask = mask & (self.data['Depth'] == self.data['Depth'][idx])
        return self._range[mask][0]


class Track(BaseSubhaloSample):

    def __init__(self, trackid, sim):
        """
        Parameters
        ----------
        trackid : int
            ID of the track to be loaded
        sim : ``Simulation`` object or ``str``
            simulation containing the track. If ``str``, should be
            simulation label (see ``Simulation.mapping``)

        To load a track from Aspotle/V1_LR, do
        >>> track = Track(trackid, 'LR')
        or
        >>> track = Track(trackid, Simulation('LR'))

        Maybe I could change this to ``Subhalo``, given that I'm loading
        the track through the reader anyway. Then I could work both within
        a snapshot and across them by merging ``Track`` with ``SubhaloSample``
        """
        self.trackid = trackid
        # load simulation
        if isinstance(sim, six.string_types):
            self.sim = Simulation(sim)
        else:
            self.sim = sim
        # load reader and track
        self.reader = HBTReader(self.sim.path)
        self.track = self.reader.GetTrack(self.trackid)
        super(Track, self).__init__(self.track, self.sim)
        # other attributes
        self.current_host = self.hosts[-1]
        self.scale = self.track['ScaleFactor']
        self.z = 1/self.scale - 1
        self._infall_snapshot = None
        self._infall_snapshot_index = None
        self._last_central_snapshot = None
        self._last_central_snapshot_index = None
        self._zcentral = None
        self._zinfall = None

    """
    @property
    def infall_snapshot(self):
        if self._infall_snapshot is None:
            self._infall_snapshot = \
                self.track['Snapshot'][self.infall_snapshot_index]
        return self._infall_snapshot

    @property
    def infall_snapshot_index(self):
        if self._infall_snapshot_index is None:
            #self._infall_snapshot_index \
                #= self._range[self.host != self.current_host][-1]
            
        return self._infall_snapshot_index
    """

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
    def zcentral(self):
        if self._zcentral is None:
            if self.track['Rank'][-1] == 0:
                self._zcentral = self.z[-1]
            else:
                self._zcentral = \
                    1/self.track['ScaleFactor']\
                        [self._last_central_snapshot_index] - 1
        return self._zcentral

    """
    @property
    def zinfall(self):
        if self._zinfall is None:
            self._zinfall = self.z[self.host != self.current_host][-1]
        return self._zinfall
    """

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

    def mergers(self, output='index'):
        """Identify merging events

        A merging event is defined as one in which the host halo at a
        particular snapshot is both different and more massive than the
        host halo at the previous snapshot

        Parameters
        ----------

        """
        return
