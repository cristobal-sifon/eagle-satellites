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
    #ic.disable()
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
    idx_infall, idx_sat, idx_cent \
        = -99 * np.ones((3,ntracks), dtype=np.uint8)
    t_infall, t_sat, t_cent \
        = -99 * np.ones((3,ntracks), dtype=np.float16)
    z_infall, z_sat, z_cent \
        = -99 * np.ones((3,ntracks), dtype=np.float16)
    Mbound_infall, Mbound_sat, Mbound_cent \
        = -99 * np.ones((3,ntracks), dtype=np.float16)
    Mstar_infall, Mstar_sat, Mstar_cent \
        = -99 * np.ones((3,ntracks), dtype=np.float16)
    Mdm_infall, Mdm_sat, Mdm_cent \
        = -99 * np.ones((3,ntracks), dtype=np.float16)
    TrackId_current_host, TrackId_previous_host \
        = -99 * np.ones((2,ntracks), dtype=np.uint16)

    path = os.path.join(sim.data_path, 'history')
    os.makedirs(path, exist_ok=True)
    history_keys = ('idx','t','z','Mbound','Mstar','Mdm')
    output = {key: os.path.join(path, f'history_{key}.npy')
              for key in history_keys}
    # this one contains track IDs of our subhalos and their current
    # and previous hosts
    history_trackids = os.path.join(path, 'history_trackids.npy')
    # save a file with all column names
    output_log = os.path.join(path, 'history.log')
    with open(output_log, 'w') as f:
        filename = 'history_trackids'
        columns = 'TrackId,TrackId_current_host,TrackId_previous_host'
        print(f'{filename:<20s}  {columns}', file=f)
        for key in history_keys:
            filename = os.path.split(
                output.get(key, f'history_{key}'))[1]
            columns = ','.join(
                [f'{key}_{inst}' for inst in ('infall','sat','cent')])
            print(f'{filename:<20s}  {columns}', file=f)
    print(f'Saved column names to {output_log}')
    # for use below (but we didn't want to include it in the loop above)
    output['trackids'] = history_trackids
    ## now calculate things!
    # this one is to store values across snapshots. The three
    # arrays are Mbound, Mstar, Mdm
    #Mbound_all = [[], [], []]
    snaps = np.arange(nsnap, -1, -1, dtype=np.uint16)
    for i in tqdm(snaps):
        subs_i = reader.LoadSubhalos(
            i, selection=('TrackId','Rank','HostHaloId',
                          'Mbound','MboundType'))
        # snapshot 281 is missing
        if subs_i.size == 0:
            continue
        z_i = sim.redshift(i)
        t_i = sim.cosmology.lookback_time(z_i)
        # find test track IDs in the current snapshot
        isin = np.isin(tracks.track_ids, subs_i['TrackId'])
        isin_reverse = np.isin(subs_i['TrackId'], tracks.track_ids)
        # let's make sure!
        assert np.array_equal(
            tracks.track_ids[isin], subs_i['TrackId'][isin_reverse])
        isin_reverse_rng = np.arange(isin_reverse.size)[isin_reverse]
        # in the current snapshot, of those that exist in the last one,
        # separate into centrals and satellites
        snap_sat = (subs_i['Rank'][isin_reverse_rng] > 0)
        snap_cent = (subs_i['Rank'][isin_reverse_rng] == 0)
        # only change isat, icent of those that have never been
        # recorded as centrals before (i.e., in later snapshots)
        idx_cent_unassigned = (idx_cent[isin] == -99)
        satmask = idx_cent_unassigned & snap_sat
        if satmask.sum() > 0:
            idx_sat[isin][satmask] = i
            z_sat[isin][satmask] = z_i
            t_sat[isin][satmask] = t_i
            Mbound_sat[isin][satmask] \
                = subs_i['Mbound'][isin_reverse_rng][snap_sat]
            Mstar_sat[isin][satmask] \
                = subs_i['MboundType'][isin_reverse_rng,4][snap_sat]
            Mdm_sat[isin][satmask] \
                = subs_i['MboundType'][isin_reverse_rng,1][snap_sat]
        centmask = idx_cent_unassigned & snap_cent
        if centmask.sum() > 0:
            idx_cent[isin][centmask] = i
            z_cent[isin][centmask] = z_i
            t_cent[isin][centmask] = t_i
            Mbound_cent[isin][centmask] \
                = subs_i['Mbound'][isin_reverse_rng][snap_cent]
            Mstar_cent[isin][centmask] \
                = subs_i['MboundType'][isin_reverse_rng,4][snap_cent]
            Mdm_cent[isin][centmask] \
                = subs_i['MboundType'][isin_reverse_rng,1][snap_cent]
        # now find out whether the hosts of these satellites are
        # still the same as they are in our test snapshot
        # if not, record i_infall
        # satellites in our test snapshot
        sat_i = subs_i[isin_reverse_rng][snap_sat]
        track_id_i = sat_i['TrackId']
        # the hosts of these satellites
        host_halo_id_i = subs_i['HostHaloId'][isin_reverse_rng][snap_sat]
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
        ic(host_track_id_i.shape)
        # now we need to match them...
        same_host = np.zeros(ntracks, dtype=bool)
        # I'm only interested in those that have not been identified in
        # later snapshots
        inf_cand = (idx_infall == -99)
        ic(inf_cand.shape, inf_cand.sum())
        change_host = (host_track_id_i != host_track_id) & (idx_infall == -99)
        ic(change_host.shape, change_host.sum())
        if change_host.sum() > 0:
            idx_infall[change_host] = i+1
            z_infall_i = sim.redshift(i+1)
            z_infall[change_host] = z_infall_i
            t_infall[change_host] = sim.cosmology.lookback_time(z_infall_i)
        continue
        # same_host[inf_cand] = [()]
        # I'm only interested in those hosts that are still in
        # our test snapshot
        isinc = isin & np.isin(host_track_id, host_track_id_i)
        rng_isinc = jtracks[isinc]
        for ii, test_track, test_host \
                in zip(rng_isinc, track_ids[isinc], host_track_id[isinc]):
            if not same_host[ii]:
                continue
            ji = (track_id_i == test_track)
            if ji.sum() > 0:
                same_host[ii] = (host_track_id_i[ji] == test_host)
        # now I just need to update the indices of those that are not in the
        # same host anymore (later test whether this reverses at any point)
        continue
        if (~same_host).sum() > 0:
            idx_infall[~same_host] = i+1
            z_infall_i = sim.redshift(i+1)
            z_infall[~same_host] = z_infall_i
            t_infall[~same_host] = sim.cosmology.lookback_time(z_infall_i)
            # now masses
            # same_host_i = np.zeros(track_id_i.size, dtype=bool)
            # isinr = isin_reverse & np.isin(host_track_id_i, host_track_id)
            # ic(isinr.shape, isinr.sum())
            # rng_isinr = np.arange(track_id_i.size, dtype=np.int32)
            # for ii, track_i, host_i \
            #         in zip(rng_isinr, track_id_i[isinr],
            #                host_track_id_i[isinr]):
            #     ji = (track_ids == track_i)
            #     if ji.sum() > 0:
            #         same_host_i[ii] = (host_track_id[ji] == host_i)
            # ic(same_host_i.shape, same_host_i.sum(), (~same_host_i).sum())
            # # check later
            # ic(Mbound_infall.shape)
            # ic(same_host.shape, (~same_host).sum())
            # ic(subs_i['Mbound'].shape)
            # ic(isin_reverse.shape, isin_reverse.dtype)
            # ic(snap_cent.shape, snap_cent.sum())
            # Mbound_infall[~same_host] \
            #     = subs_i['Mbound'][isin_reverse][snap_cent]
            # Mstar_infall[~same_host] \
            #     = subs_i['MboundType'][isin_reverse,4][snap_cent]
            # Mdm_infall[~same_host] \
            #     = subs_i['MboundType'][isin_reverse,1][snap_cent]
        #but is this sorted the right way?
        #TrackId_previous_host[~same_host] = host_track_id_i[isinc]
        #return
        #if i == nsnap-3: break
        if i < nsnap and i % 20 == 0:
            np.save(output['trackids'],
                    [tracks.track_ids, TrackId_current_host,
                     TrackId_previous_host])
            np.save(output['idx'], [idx_infall, idx_sat, idx_cent])
            np.save(output['t'], [t_infall, t_sat, t_cent])
            np.save(output['z'], [z_infall, z_sat, z_cent])
            np.save(output['Mbound'], [Mbound_infall, Mbound_sat, Mbound_cent])
            np.save(output['Mstar'], [Mstar_infall, Mstar_sat, Mstar_cent])
            np.save(output['Mdm'], [Mdm_infall, Mdm_sat, Mdm_cent])
            #break
    return


if __name__ == '__main__':
    main()
