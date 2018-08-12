from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
import sys
from time import time
if sys.version_info[0] == 2:
    range = xrange

from plottools.plotutils import savefig, update_rcParams
update_rcParams()

from HBTReader import HBTReader
from core import hbt_tools
from core.simulation import Simulation
from core.subhalo import SubhaloSample
from core.track import Track


def main():

    args = hbt_tools.parse_args()
    sim = Simulation(args.simulation)

    to = time()
    reader = HBTReader(sim.path)
    print('Loaded reader in {0:.1f} seconds'.format(time()-to))

    subs = reader.LoadSubhalos(-1)
    sat = SubhaloSample(subs[subs['Rank'] > 0], sim)
    cen = SubhaloSample(subs[subs['Rank'] == 0], sim)

    mstarbins = np.logspace(8, 11, 31)
    mstar = (mstarbins[:-1]+mstarbins[1:]) / 2
    mavg_sat = np.histogram(
        sat.mass('stars'), mstarbins, weights=sat.mass('total'))[0]
    mavg_sat = mavg_sat / np.histogram(sat.mass('stars'), mstarbins)[0]
    mavg_cen = np.histogram(
        cen.mass('stars'), mstarbins, weights=cen.mass('total'))[0]
    mavg_cen = mavg_cen / np.histogram(cen.mass('stars'), mstarbins)[0]
    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(
        mstar, mavg_cen, 'k', dashes=(8,8), lw=3, label='Central subhalos')
    ax.plot(mstar, mavg_sat,'C0-', lw=2, label='Satellite subhalos')
    ax.set_xscale('log')
    ax.set_yscale('log')
    output = os.path.join(sim.plot_path, 'Mtotal_Mstars.pdf')
    savefig(output, fig=fig)

    return


main()




