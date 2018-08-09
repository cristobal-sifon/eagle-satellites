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
        >>> track = Track(trackid) Simulation('LR'))

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
        self._last_central_snapshot = None
        self._zcentral = None
        self._zinfall = None

    @property
    def infall_snapshot(self):
        if self._infall_snapshot is None:
            self._infall_snapshot \
                = self.track['Snapshot'][self.host != self.current_host][-1]
        return self._infall_snapshot

    @property
    def last_central_snapshot(self):
        if self._last_central_snapshot is None:
            self._last_central_snapshot  = \
                self.reader.GetSub(self.trackid)['SnapshotIndexOfLastIsolation']

    @property
    def zcentral(self):
        if self._zcentral is None:
            if self.track['Rank'] == 0:
                self._zcentral = self.z[-1]
            else:
                self._zcentral = \
                    1/self.track['ScaleFactor'][self._last_central_snapshot] - 1
        return self._zcentral

    @property
    def zinfall(self):
        if self._zinfall is None:
            self._zinfall = self.z[self.host != self.current_host][-1]
        return self._zinfall
