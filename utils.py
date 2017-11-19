from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from astropy.io import ascii
from numpy import argmin

def snapz(snapshot=None, z=None):
    """
    Given a snapshot number, return the snapshot redshift; or
    given a redshift, return the closest snapshot and its redshift.

    If both are provided, `z` is ignored.
    """
    assert not (snapshot is None and z is None), \
        'Please provide a snapshot number or redshift'
    data = ascii.read('snapshots.txt')
    if snapshot is not None:
        assert int(snapshot) == snapshot and 0 <= snapshot <= 28, \
            'The snapshot number must be an integer 0 <= snapshot <= 28'
        j = (data['SnapNum'] == snapshot)
        print('in function: z = {0}'.format(data[j]['z']))
        return data['z'][j]
    j = argmin(abs(data['z'] - z))
    return data['SnapNum'][j], data['z'][j]
