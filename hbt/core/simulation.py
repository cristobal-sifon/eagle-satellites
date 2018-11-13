from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from astropy.io import ascii
from astropy.cosmology import FlatLambdaCDM
from glob import glob
import numpy as np
import os
import six

from HBTReader import HBTReader


class BaseSimulation(object):
    """
    This class is imported by external objects such as Subhalo or
    Track. For now, it only provides the capability to load a
    Simulation object

    """

    def __init__(self, sim):
        # initialize simulation and reader. This cannot be modified
        # within the same instance
        if isinstance(sim, six.string_types):
            self.sim = Simulation(sim)
        else:
            self.sim = sim
        self.reader = HBTReader(self.sim.path)


class Simulation(object):

    def __init__(self, label):
        self.label = label
        self._cosmology = None
        self._family = None
        self._formatted_name = None
        self._snapshots = None
        self._snapshot_indices = None
        self._virial_snapshots = None
        self.initialize_tree()
        print('Loaded {0} from {1}'.format(self.formatted_name, self.path))

    ### attributes ###

    @property
    def cosmology(self):
        if self._cosmology is None:
            cosmo = {
                'apostle': FlatLambdaCDM(H0=70.4, Om0=0.272, Ob0=0.0455),
                'eagle': FlatLambdaCDM(H0=67.77, Om0=0.307, Ob0=0.04825)
                }
            self._cosmology = cosmo[self.family]
        return self._cosmology

    @property
    def family(self):
        if self._family is None:
            self._family = self.name.split('/')[0]
        return self._family

    @property
    def formatted_name(self):
        if self._formatted_name is None:
            _name = self.name.split('/')
            if self.name.startswith('apostle'):
                _name[0] = _name[0].capitalize()
            elif self.name.startswith('eagle'):
                _name[0] = _name[0].upper()
            self._formatted_name = '/'.join([_name[0], _name[1].upper()])
        return self._formatted_name

    @property
    def masstypes(self):
        return np.array(['gas', 'dm', 'disk', 'bulge', 'stars', 'bh'])

    @property
    def masstype_pandas_columns(self):
        return np.array(['MboundType{0}'.format(self._masstype_index(i))
                         for i in self.masstypes])

    @property
    def masstype_labels(self):
        return np.array(['gas', 'DM', 'disk', 'bulge', 'stars', 'BH'])

    @property
    def _mapping(self):
        return {'LR': 'apostle/V1_LR',
                'MR': 'apostle/V1_MR',
                'HR': 'apostle/V1_HR',
                'L25': 'eagle/L0025N0376',
                'L100': 'eagle/L0100N1504'}

    @property
    def name(self):
        return self._mapping[self.label]

    @property
    def path(self):
        return os.path.join(self.root, self.name, 'subcat')

    @property
    def plot_path(self):
        return os.path.join('plots', self.name.replace('/', '_'))

    @property
    def redshift(self):
        snaplist = os.path.join(self.root, 'subcat', 'snapshotlist.txt')
        snaplist = ascii.read(snaplist)
        return np.array(
            [x.split('_')[2][1:].replace('p', '.') for x in snaplist],
            dtype=float)

    @property
    def root(self):
        return '/cosma/home/durham/jvbq85/data/HBT/data'

    @property
    def snapshots(self):
        if self._snapshots is None:
            try:
                snaps = [
                    i.split('/')[-1].split('_')[1].split('.')[0]
                    for i in sorted(glob(os.path.join(self.path,'SubSnap*')))]
            except IndexError:
                snaps = []
                for i in sorted(os.listdir(self.path)):
                    try:
                        snaps.append(int(i))
                    except ValueError:
                        pass
            self._snapshots = np.array(snaps, dtype=int)
        return self._snapshots

    @property
    def snapshot_indices(self):
        if self._snapshot_indices is None:
            self._snapshot_indices = np.arange(
                self.snapshots.size, dtype=int)
        return self._snapshot_indices

    @property
    def virial_path(self):
        return os.path.join(self.path, 'HaloSize')

    @property
    def virial_snapshots(self):
        if self._virial_snapshots is None:
            lsdir = glob(os.path.join(self.virial_path, 'HaloSize_*.hdf5'))
            print(lsdir[0])
            print(lsdir[0].split('/')[-1].split('.')[0].split('_')[1])
            self._virial_snapshots = np.sort(
                [int(i.split('/')[-1].split('.')[0].split('_')[1])
                 for i in glob(os.path.join(
                    self.virial_path, 'HaloSize_*.hdf5'))])
        return self._virial_snapshots

    ### methods ###

    def masstype_pandas_column(self, mtype):
        """Name of the column containing requested mass type

        Parameters
        ----------
        mtype : str
            mass type. See ``self.masstypes``

        Returns
        -------
        column : str
            column name
        """
        return self.masstype_pandas_columns[self._masstype_index(mtype)]

    def snapshot_index(self, snap):
        """Given a snapshot number, return the index to which it
        corresponds

        Parameters
        ----------
        snap : int
            snapshot number

        Returns
        -------
        isnap : int
            snapshot index
        """
        return self.snapshot_indices[self.snapshots == snap][0]

    def initialize_tree(self):
        if not os.path.isdir(self.plot_path):
            os.makedirs(self.plot_path)

    def masslabel(self, latex=True, **kwargs):
        """Return label of a given mass type

        Paramters
        ---------
        mtype : str, optional
            name of the mass type. Must be one of ``self.masstypes`` or
            'total'
        index : int, optional
            index of the mass type. An index of -1 corresponds to total
            mass. Otherwise see ``self.masstypes``
        latex : bool, optional
            whether the label should be in latex or plain text format

        Returns
        -------
        label : str
            mass label in latex format
        """
        if latex:
            return r'M_\mathrm{{{0}}}'.format(self.masstype(**kwargs))
        else:
            return 'M{0}'.format(self.masstype(**kwargs).lower())

    def masstype(self, mtype=None, index=None):
        assert mtype is not None or index is not None, \
            'must provide either ``mtype`` or ``index``'
        if mtype is not None:
            if mtype.lower() in ('total', 'mbound'):
                return 'total'
            return self.masstype_labels[self._masstype_index(mtype)]
        if index == -1:
            return 'total'
        return self.masstype_labels[index]

    ### private methods ###

    def _masstype_index(self, mtype):
        rng = np.arange(self.masstypes.size, dtype=int)
        return rng[self.masstypes == mtype.lower()][0]


