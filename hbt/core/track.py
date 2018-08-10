from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import six

from HBTReader import HBTReader

from .simulation import Simulation


class Track():

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
        # other attributes
        self.host = self.track['HostHaloId']
        self.current_host = self.host[-1]
        self.scale = self.track['ScaleFactor']
        self.z = 1/self.scale - 1
        self._infall_snapshot = None
        self._infall_snapshot_index = None
        self._last_central_snapshot = None
        self._last_central_snapshot_index = None
        self._Mbound = None
        self._MboundType = None
        self._zcentral = None
        self._zinfall = None
        # private attributes
        self.__range = None

    @property
    def _range(self):
        if self.__range is None:
            self.__range = np.arange(self.track['Snapshot'].size, dtype=int)
        return self.__range

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

    def lookback_time(self, z=None, scale=None, snapshot=None):
        if z is not None:
            return self.sim.cosmology.lookback_time(z)
        if scale is not None:
            return self.sim.cosmology.lookback_time(1/scale - 1)
        if snapshot is not None:
            z = 1/track['ScaleFactor'][snapshot] - 1
            return self.sim.cosmology.lookback_time(z)
        return self.sim.cosmology.lookback_time(self.z)

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

    def mergers(self, output='index'):
        """Identify merging events

        A merging event is defined as one in which the host halo at a
        particular snapshot is both different and more massive than the
        host halo at the previous snapshot

        Parameters
        ----------

        """
        return
