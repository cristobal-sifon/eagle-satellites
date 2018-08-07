from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os


class Simulation:

    def __init__(self, label):
        self.label = label
        self._formatted_name = None
        self._snapshots = None
        self.initialize()

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
        return np.array(['gas', 'halo', 'disk', 'bulge', 'stars', 'boundary'])

    @property
    def mapping(self):
        return {'LR': 'apostle/V1_LR',
                'MR': 'apostle/V1_MR',
                'HR': 'apostle/V1_HR',
                'L25': 'eagle/L0025N0376',
                'L100': 'eagle/L0100N1504'}

    @property
    def name(self):
        return self.mapping[self.label]

    @property
    def path(self):
        return os.path.join(self.root, self.name, 'subcat')

    @property
    def plot_path(self):
        return os.path.join('plots', self.name.replace('/', '_'))

    @property
    def root(self):
        return '/cosma/home/jvbq85/data/HBT/data'

    @property
    def snapshots(self):
        if self._snapshots is None:
            try:
                self._snapshots = np.array(
                    [i.split('/')[-1].split('_')[1].split('.')[0]
                     for i in sorted(glob(
                         os.path.join(self.path,'SubSnap*')))])
            except IndexError:
                snaps = []
                for i in sorted(os.listdir(self.path)):
                    try:
                        snaps.append(int(i))
                    except ValueError:
                        pass
                self._snapshots = np.array(snaps)
        return self._snapshots

    def initialize(self):
        if not os.path.isdir(self.plot_path):
            os.makedirs(self.plot_path)

    def masstype_index(self, mtype):
        rng = np.arange(self.masstypes.size, dtype=int)
        return rng[self.masstypes == mtype.lower()][0]


