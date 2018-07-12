import os


class Simulation:

    def __init__(self, label):
        """
        add snapshots list and redshifs (see explore.py lines 188,189)
        """
        self.label = label
        self.initialize()

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

    def initialize(self):
        if not os.path.isdir(self.plot_path):
            os.makedirs(self.plot_path)
