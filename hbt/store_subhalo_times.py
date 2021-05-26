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
    sats = subs.satellites
    cents = subs.centrals
    ic(sats['TrackId'].shape)
    ic(cents['TrackId'].shape)
    ti = time()
    #j = np.arange(subhalos.shape[0], dtype=int)
    #host_idx = np.array(
        #[j[(subhalos['Rank'] == 0) & (subhalos['HostHaloId'] == hostid)][0]
         #for hostid in sats['HostHaloId']])
    host_halo_id = sats['HostHaloId'].to_numpy(dtype=np.int32)
    cent_track_id = cents['TrackId'].to_numpy(dtype=np.int32)
    host_track_id = np.array(
        [cent_track_id[cents['HostHaloId'] == hhi][0]
         for hhi in host_halo_id])
    print(f'host tracks in {time()-ti:.2f} s')
    ic(host_track_id[:5])

    # this is unnecessary really
    tracks = TrackArray(subs.satellites['TrackId'], sim)
    track_ids = tracks.track_ids
    ntracks = track_ids.size
    # for later use
    jtracks = np.arange(ntracks, dtype=int)
    ic(ntracks)

    tracks_cent = TrackArray(subs.centrals['TrackId'], sim)
    ntracks_cent = tracks_cent.track_ids.size
    ic(ntracks_cent)

    nsnap = sim.snapshots.size
    idx_infall, idx_sat, idx_cent = -99 * np.ones((3,ntracks), dtype=int)

    output = os.path.join(sim.data_path, 'history.npy')
    for i in tqdm(range(nsnap, -1, -1)):
        subs_i = reader.LoadSubhalos(
            i, selection=('TrackId','Rank','HostHaloId'))
        # snapshot 281 is missing
        if subs_i.size == 0:
            continue
        # find test track IDs in the current snapshot
        isin = np.isin(tracks.track_ids, subs_i['TrackId'])
        isin_reverse = np.isin(subs_i['TrackId'], tracks.track_ids)
        # let's make sure!
        assert np.array_equal(
            tracks.track_ids[isin], subs_i['TrackId'][isin_reverse])
        isin_reverse = np.arange(isin_reverse.size)[isin_reverse]
        # in the current snapshot, of those that exist in the last one,
        # separate into centrals and satellites
        snap_sat = (subs_i['Rank'][isin_reverse] > 0)
        snap_cent = (subs_i['Rank'][isin_reverse] == 0)
        # only change isat, icent of those that have never been
        # recorded as centrals before (i.e., in later snapshots)
        idx_cent_unassigned = (idx_cent[isin] == -99)
        #try:
        idx_sat[isin][idx_cent_unassigned & snap_sat] = i
        idx_cent[isin][idx_cent_unassigned & snap_cent] = i
        # I think this doesn't happen anymore
        """
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
        """
        # now find out whether the hosts of these satellites are
        # still the same as they are in our test snapshot
        # if not, record i_infall
        # satellites in our test snapshot
        sat_i = subs_i[isin_reverse][snap_sat]
        track_id_i = sat_i['TrackId']
        # the hosts of these satellites
        host_halo_id_i = subs_i['HostHaloId'][isin_reverse][snap_sat]
        # all centrals in the snapshot
        cent_i = subs_i[subs_i['Rank'] == 0]
        # only those that host our satellites
        cents_used = np.isin(cent_i['HostHaloId'], sat_i['HostHaloId'])
        # HostHaloId of these centrals
        cent_halo_id_i = cent_i['HostHaloId'][cents_used]
        # and their track ids
        cent_track_id_i = cent_i['TrackId'][cents_used]
        # finally, get the track id of each central associated
        # with every one of our satellites
        host_track_id_i = np.array(
            [cent_track_id_i[cent_halo_id_i == hhii][0]
             for hhii in host_halo_id_i])
        # now we need to match them...
        same_host = np.zeros(ntracks, dtype=bool)
        # I'm only interested in those hosts that are still in
        # our test snapshot
        isinc = isin & np.isin(host_track_id, host_track_id_i)
        rng_isinc = jtracks[isinc]
        for ii, test_track, test_host \
                in zip(rng_isinc, track_ids[isinc], host_track_id[isinc]):
            ji = (track_id_i == test_track)
            same_host[ii] = (host_track_id_i[ji] == test_host)
        # now I just need to update the indices of those that are not in the
        # same host anymore (later test whether this reverses at any point)
        idx_infall[~same_host] = i+1
        #return
        #if i == nsnap-2: break
        if i % 10 == 0:
            data = [tracks.track_ids, idx_infall, idx_sat, idx_cent]
            np.save(output, data)
            #break
    return


if __name__ == '__main__':
    main()

