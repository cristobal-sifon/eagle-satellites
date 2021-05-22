from glob import glob
from icecream import ic
import numpy as np
import os
from time import time
from tqdm import tqdm

from HBTReader import HBTReader

# local
from hbtpy import hbt_tools
from hbtpy.simulation import Simulation
from hbtpy.subhalo import Subhalos
from hbtpy.track import TrackArray


def main():
    args = hbt_tools.parse_args()
    sim = Simulation(args.simulation)
    ic(f'Loaded {sim.name} with {sim.snapshots.size} snapshots!')

    to = time()
    reader = HBTReader(sim.path)
    print('Loaded reader in {0:.1f} seconds'.format(time()-to))

    isnap = -1
    ti = time()
    subs = reader.LoadSubhalos(isnap)
    print(f'Loaded subhalos in {time()-ti:.2f} s!')

    subs = Subhalos(
        subs, sim, isnap, exclude_non_FoF=True,
        load_distances=False, load_velocities=False)
    subhalos = subs.catalog
    #ic(np.sort(subs.colnames))

    #for track in s
    tracks = TrackArray(subs.satellites['TrackId'], sim)
    ntracks = tracks.track_ids.size
    ic(ntracks)

    nsnap = sim.snapshots.size
    idx_infall, idx_sat, idx_cent = -99 * np.ones((3,ntracks), dtype=int)

    output = os.path.join(sim.data_path, 'history.npy')
    # skipping i_infall for now. I need to include the host halo track
    # (see subhalo.HostHaloTrack)
    for i in tqdm(range(nsnap, -1, -1)):
        #ti = time()
        # load subhalos in current snapshot
        subs_i = reader.LoadSubhalos(i, selection=('TrackId','Rank'))
        # find test track IDs in the current snapshot
        isin = np.isin(tracks.track_ids, subs_i['TrackId'])
        isin_reverse = np.isin(subs_i['TrackId'], tracks.track_ids)
        # in the current snapshot, of those that exist in the last one,
        # separate into centrals and satellites
        snap_sat = (subs_i['Rank'][isin_reverse] > 0)
        snap_cent = (subs_i['Rank'][isin_reverse] == 0)
        # ... but only change the indices of those that have never been
        # recorded as centrals before (i.e., in later snapshots)
        idx_cent_unassigned = (idx_cent[isin] == -99)
        try:
            idx_sat[isin][idx_cent_unassigned & snap_sat] = i
            idx_cent[isin][idx_cent_unassigned & snap_cent] = i
        except ValueError as err:
            ic(idx_cent.shape)
            ic(idx_sat.shape)
            ic(isin.shape)
            ic(isin.sum())
            ic(idx_cent_unassigned.shape)
            ic(idx_cent_unassigned.sum())
            ic(snap_sat.shape)
            ic(snap_sat.sum())
            ic(snap_cent.shape)
            ic(snap_cent.sum())
            raise ValueError(err)
        #print(f'{time()-ti:.2f} s')
        #print()
        if i % 10 == 0:
            data = [tracks.track_ids, idx_infall, idx_sat, idx_cent]
            np.save(output, data)
            #break
    return


if __name__ == '__main__':
    main()
