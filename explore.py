#!/usr/bin/env python
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import os
from astropy.io import ascii
from matplotlib import pyplot as plt

# my code
import plottools
plottools.update_rcParams()


def main(sim='RefL0100N1504', snapshot=26):
    path = os.path.join('data', sim, 'snapshot{0}'.format(snapshot))
    group = ascii.read(os.path.join(path, 'groups.txt'))
    sat = ascii.read(os.path.join(path, 'satellites.txt'))
    print(group)
    return


main()


