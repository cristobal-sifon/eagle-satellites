from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
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

    def __init__(self, catalog, sim):
        """
        Parameters
        ----------
        catalog : ``Track.track`` or ``Subhalos.data``
        sim : ``Simulation``
        """
        self.catalog = catalog
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
    def Mbound(self):
        if self._Mbound is None:
            self._Mbound = 1e10 * self.catalog['Mbound']
        return self._Mbound

    @property
    def MboundType(self):
        if self._MboundType is None:
            self._MboundType = 1e10 * self.catalog['MboundType']
        return self._MboundType

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

    def __init__(self, subhalos, sim):
        """
        Parameters
        ----------
        subhalos : output of ``reader.LoadSubhalos``
            subhalo sample, defined at a specific snapshot
        sim : ``Simulation`` object or ``str``
            simulation containing the track. If ``str``, should be
            simulation label (see ``Simulation.mapping``)
        """
        self.subhalos = subhalos
        super(Subhalos, self).__init__(self.subhalos, sim)
        self._central_idx = None
        self._central_mask = None
        self._centrals = None
        self._satellite_idx = None
        self._satellite_mask = None
        self._satellites = None

    ### attributes ###

    @property
    def centrals(self):
        if self._centrals is None:
            self._centrals = self.subhalos[self.central_mask]
        return self._centrals

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
    def satellites(self):
        if self._satellites is None:
            self._satellites = self.subhalos[self.subhalos['Rank'] > 0]
        return self._satellites

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
        idx = self.index(trackid)
        sibling_mask = (self.subhalos['HostHaloId'] \
                        == self.subhalos['HostHaloId'][idx])
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


class Track(BaseSubhalo):

    def __init__(self, track, sim):
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
        self.track = track
        self._trackid = self.track['TrackId']
        super(Track, self).__init__(self.track, sim)
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

    """
    @property
    def zinfall(self):
        if self._zinfall is None:
            self._zinfall = self.z[self.host != self.current_host][-1]
        return self._zinfall
    """

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
        """This should probably be in Simulation"""
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
