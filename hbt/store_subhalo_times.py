from glob import glob
import h5py
from icecream import ic
import numpy as np
import os
import pandas as pd
from time import time
from tqdm import tqdm
import sys

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
    print(f'Loaded {subs.shape[0]} subhalos in {time()-ti:.2f} s!')

    subs = Subhalos(
        subs, sim, isnap, exclude_non_FoF=True,
        logMmin=9, logM200Mean_min=12,
        load_distances=False, load_velocities=False,
        load_hosts=False, load_history=False)
    subhalos = subs.catalog
    sats = subs.satellites
    cents = subs.centrals
    ic(sats['TrackId'].shape)
    ic(cents['TrackId'].shape)
    host_halo_id = sats['HostHaloId'].to_numpy(dtype=np.int32)
    ic(host_halo_id.shape)
    cent_track_id = cents['TrackId'].to_numpy(dtype=np.int32)
    ti = time()
    host_track_id_today = np.array(
        [cent_track_id[cents['HostHaloId'] == hhi][0]
         for hhi in host_halo_id])
    print(f'host tracks in {time()-ti:.2f} s')
    ic(host_track_id_today[:5])

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

    #groups = ('trackids', 'infall', 'sat', 'cent')
    groups = ('trackids', 'first_infall', 'last_infall', 'sat', 'cent')
    names = [('TrackId', 'TrackId_current_host',
              'TrackId_previous_host')] \
        + 4*[('isnap', 'time', 'z', 'Mbound', 'Mstar', 'Mdm', 'Mgas')]
    dtypes = [3*[np.int32]] + 4*[[np.int16] + 6*[np.float16]]
    trackcols, firstinfcols, lastinfcols, satcols, centcols \
        = [[f'{group}/{name}' for name in grnames]
            for group, grnames in zip(groups, names)]
    ic(satcols)
    # DataFrame
    history = {
        f'{gr}/{name}': -np.ones(ntracks, dtype=dt)
        for gr, grnames, dtype in zip(groups, names, dtypes)
        for name, dt in zip(grnames, dtype)}
    history['trackids/TrackId'] = track_ids
    history['trackids/TrackId_current_host'] = host_track_id_today
    history = pd.DataFrame(history).reset_index()
    ic(np.sort(history.columns))
    ic(history.shape)

    # save everything to an hdf5 file
    path_history = os.path.join(sim.data_path, 'history')
    hdf = os.path.join(path_history, 'history.h5')
    ## now calculate things!
    snaps = np.arange(nsnap, -1, -1, dtype=np.uint16)

    z, t = np.zeros((2,snaps.size))
    selection = ('TrackId','Rank','HostHaloId', 'Mbound','MboundType')
    if not args.debug: ic.disable()
    for i, snap in enumerate(tqdm(snaps)):
        ic(i, snap)
        subs_i = reader.LoadSubhalos(snap, selection=selection)
        # snapshot 281 is missing
        if subs_i.size == 0:
            continue
        subs_i = ({key: subs_i[key] for key in selection[:-1]},
                  {f'MboundType{j}': 1e10 * subs_i['MboundType'][:,j]
                   for j in (0,1,4)})
        subs_i['Mbound'] = 1e10 * subs_i['Mbound']
        subs_i = {**subs_i[0], **subs_i[1]}
        subs_i = pd.DataFrame(subs_i)
        z[i] = sim.redshift(snap)
        t[i] = sim.cosmology.lookback_time(z[i]).to('Gyr').value
        # first match subhaloes now and then
        this = history.merge(
            subs_i, left_on='trackids/TrackId', right_on='TrackId',
            how='left', suffixes=('_test','_snap'))
        if i == 0:
            ic(np.sort(this.columns))
        ic(this.shape)
        # this one gets updated all the time
        still_sat = (this['Rank'] > 0)
        ic(still_sat.shape, still_sat.sum())
        history.loc[still_sat, satcols[:3]] = [snap, t[i], z[i]]
        history = assign_masses(history, this, still_sat, 'sat')
        if i == 0:
            continue
        #history['sat/Mbound'].loc[still_sat] = this[
        # this one is updated only once per subhalo
        new_cent = (this['cent/isnap'] == -1) & (this['Rank'] == 0)
        ic(new_cent.shape, new_cent.sum())
        history.loc[new_cent, centcols[:3]] = [snap, t[i], z[i]]
        history = assign_masses(history, this, new_cent, 'cent')
        ic()
        ic(this.shape)

        ## infall
        # first map HostHaloId to TrackId
        this_cent_mask = (subs_i['Rank'] == 0) & (subs_i['HostHaloId'] > -1)
        this_cent = subs_i[['TrackId','HostHaloId']].loc[this_cent_mask]
        ic(this_cent)
        this = this.merge(
            this_cent, on='HostHaloId', how='left',
            suffixes=('_sat','_cent'))
        ic(np.sort(this.columns))
        # first update the masses for all subhalos for which we have
        # not registered infall already
        jinfall_last = (this['last_infall/isnap'] == -1)
        ic(jinfall_last.shape, jinfall_last.sum())
        history = assign_masses(history, this, jinfall_last, 'last_infall')
        # if they are no longer in the same host, then register
        # the other columns. Once we've done this their masses
        # will never be updated again
        ic(this.shape, history.shape)
        host_track = history['trackids/TrackId_current_host']
        try:
            jinfall_last = jinfall_last & (this['TrackId_cent'] != host_track)
        except ValueError as e:
            print(jinfall_last.shape, jinfall_last.sum(),
                  this['TrackId_cent'].shape, host_track.shape)
            raise ValueError(e)
        ic(jinfall_last.sum())
        history.loc[jinfall_last, lastinfcols[:3]] = [snap+1, t[i-1], z[i-1]]
        # assign these same values to first_infall
        history = assign_masses(history, this, jinfall_last, 'first_infall')
        history.loc[jinfall_last, firstinfcols[:3]] = [snap+1, t[i-1], z[i-1]]
        # here we record the first time each subhalo fell into its
        # current host, i.e., the same as above but the info can be
        # updated once set
        left_host_and_back = (history['last_infall/isnap'] != -1) \
            & (this['TrackId_cent'] == host_track)
        ic(left_host_and_back.sum())
        history.loc[left_host_and_back, firstinfcols[:3]] = [snap, t[i], z[i]]
        history = assign_masses(
            history, this, left_host_and_back, 'first_infall')
        # TrackId_previous_host refers to the host prior to the first
        # time the subhalo entered its current host (does it make a
        # difference?), i.e. those for which we've already registered
        # an infall in the *immediate* previous iteration
        # this will be updated each time the subhalo leaves
        jinfall_first = (history['first_infall/isnap'] == snap+1) \
            & (this['TrackId_cent'] != host_track)
        ic(jinfall_first.sum())
        history.loc[jinfall_first, 'trackids/TrackId_previous_host'] \
            = this['TrackId_cent'][jinfall_first]

        #ic(np.unique(history['cent/isnap'], return_counts=True))
        #ic(np.sort(history['sat/Mbound']))
        #ic((history > -1).sum())
        #ic(history.mean())
        if i % 100 == 0:
            store_h5(hdf, groups, names, history)
            #store_npy(path, groups, names, history)
            #break

    #ic(history)
    store_h5(hdf, groups, names, history)
    print(f'Finished! Stored everything to {hdf}.')

    return


def assign_masses(history, this, mask, group):
    history.loc[mask, f'{group}/Mbound'] = this.loc[mask, 'Mbound']
    history.loc[mask, f'{group}/Mgas'] = this.loc[mask, 'MboundType0']
    history.loc[mask, f'{group}/Mdm'] = this.loc[mask, 'MboundType1']
    history.loc[mask, f'{group}/Mstar'] = this.loc[mask, 'MboundType4']
    return history


def store_h5(hdf, groups, names, data):
    ic(np.sort(data.columns))
    ic(names)
    with h5py.File(hdf, 'w') as out:
        for group, grnames in zip(groups, names):
            ic(grnames)
            gr = out.create_group(group)
            for name in grnames:
                col = f'{group}/{name}'
                gr.create_dataset(
                    name, data=data[col].to_numpy(), dtype=data[col].dtype)
    ic(f'Saved to {hdf}')
    return


def store_npy(path, groups, names, history):
    np.save(output['trackids'],
            [tracks.track_ids, TrackId_current_host,
             TrackId_previous_host])
    np.save(output['idx'], [idx_infall, idx_sat, idx_cent])
    np.save(output['t'], [t_infall, t_sat, t_cent])
    np.save(output['z'], [z_infall, z_sat, z_cent])
    np.save(output['Mbound'], [Mbound_infall, Mbound_sat, Mbound_cent])
    np.save(output['Mstar'], [Mstar_infall, Mstar_sat, Mstar_cent])
    np.save(output['Mdm'], [Mdm_infall, Mdm_sat, Mdm_cent])
    return

if __name__ == '__main__':
    main()
