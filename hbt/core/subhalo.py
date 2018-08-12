from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import six

from HBTReader import HBTReader

from .simulation import Simulation


class BaseSubhalo():

    def __init__(self, catalog, sim):
        """
        Parameters
        ----------
        catalog : ``Track.track`` or ``Subhalo.data``
        sim : ``Simulation``
        """
        self.catalog = catalog
        self.sim = sim
        self._Mbound = None
        self._MboundType = None
        # private attributes
        self.__range = None

    @property
    def _range(self):
        if self.__range is None:
            self.__range = np.arange(self.catalog.size, dtype=int)
        return self.__range

    @property
    def hosts(self):
        return self.catalog['HostHaloId']

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


class Subhalo(BaseSubhalo):
    """Class to manage a sample of subhalos at a given snapshot

    """

    def __init__(self, trackid, sim, isnap=-1):
        """
        Parameters
        ----------
        trackid : int
            ID of the track to be loaded
        sim : ``Simulation`` object or ``str``
            simulation containing the track. If ``str``, should be
            simulation label (see ``Simulation.mapping``)
        isnap : int, optional
            snapshot index. Default is -1, which refers to the last
            snapshot. All time-specific quantities are calculated at,
            or relative to, snapshot ``isnap``

        To load a subhalo and its track from Aspotle/V1_LR, do
        >>> track = Subhalo(trackid, 'LR')
        or
        >>> track = Subhalo(trackid, Simulation('LR'))
        """
        self._isnap = isnap
        # without the underscore so that the reader is defined as well
        # (which happens in the setter)
        self.sim = sim
        # load reader and track
        self.trackid = trackid
        super(Subhalo, self).__init__(self.track, self.sim)
        # other attributes
        #self.current_host = self.hosts[-1]
        self._infall_snapshot = None
        self._infall_snapshot_index = None
        self._last_central_snapshot = None
        self._last_central_snapshot_index = None
        self._zcentral = None
        self._zinfall = None

    @property
    def future(self):
        """All snapshots in the future of ``self.isnap``

        If ``self.isnap=-1``, returns ``None``
        """
        if self.isnap == -1:
            return None
        return self.track[self.isnap+1:]

    @property
    def host_halo_id(self):
        """HostHaloId"""
        return self.present['HostHaloId']

    @property
    def host_track_id(self):
        return self.track['TrackId'][self.siblings['Rank'] == 0][0]

    @property
    def isnap(self):
        return self._isnap

    @isnap.setter
    def isnap(self, isnap):
        self._isnap = isnap

    @property
    def icent(self):
        """Index of the last snapshot up to ``self.isnap`` when the
        subhalo was a central subhalo"""
        if self.track['Rank'][self.isnap] == 0:
            return self.isnap
        return self.past['Rank'][self.past['Rank'] == 0][-1]

    @property
    def past(self):
        return self.track[:self.isnap]

    @property
    def present(self):
        return self.track[self.isnap]

    @property
    def scale(self):
        return self.track['ScaleFactor']

    @property
    def siblings(self):
        """Track IDs of all subhalos hosted by the same halo at
        present"""
        return self.present['TrackId'][self.present['HostHaloId'] \
                                       == self.current_host][0]

    @property
    def sim(self):
        return self._sim

    @sim.setter
    def sim(self, sim):
        if isinstance(sim, six.string_types):
            self._sim = Simulation(sim)
        else:
            self._sim = sim
        # update the reader for consistency
        self.reader = HBTReader(self._sim.path)

    @property
    def trackid(self):
        return self._trackid

    @trackid.setter
    def trackid(self, trackid):
        self._trackid = trackid
        self.track = self.reader.GetTrack(self._trackid)

    @property
    def z(self):
        return 1/self.scale - 1

    @property
    def zcent(self):
        if self.is_central():
            return self.z[self.isnap]
        else:
            return self.z[:self.isnap][self.icent]

    def is_central(self):
        return (self.track[self.isnap]['Rank'] == 0)


    def index(self, trackid):
        """Index of a given trackid in the subhalo catalog"""
        return self._range[self.data['TrackId'] == trackid][0]


class Track(BaseSubhalo):
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
