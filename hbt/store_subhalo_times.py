import h5py
from icecream import IceCreamDebugger
import numpy as np
import os
import pandas as pd
from time import time
from tqdm import tqdm
import warnings

from HBTReader import HBTReader

# local
from hbtpy import hbt_tools
from hbtpy.simulation import Simulation
from hbtpy.subhalo import Subhalos

warnings.simplefilter('ignore', UserWarning)


icnames = ('birth', 'cent', 'sat', 'infall', 'history', 'masses', 'base')
ics = {name: IceCreamDebugger() for name in icnames}
ic0 = ics['base']

def main():
    args = hbt_tools.parse_args(
        args=[('--no-store', {'dest': 'store', 'action': 'store_false'})] \
            + [(f'--{name}', {'action': 'store_true'}) for name in icnames]
    )
    # individual ic instances
    for name in icnames:
        if not args.__dict__.get(name):
            ics[name].disable()
    if args.history:
        for name in icnames[:4]:
            ics[name].enable()
    if args.debug:
        ic0.enable()

    sim = Simulation(args.simulation)
    print(f'Loaded {sim.name} with {sim.snapshots.size} snapshots!')
    reader = HBTReader(sim.path)
    isnap = -1
    ti = time()
    subs = reader.LoadSubhalos(isnap)
    print(f'Loaded {subs.shape[0]} subhalos in {time()-ti:.2f} s!')

    subs = Subhalos(
        subs, sim, isnap, exclude_non_FoF=True,
        logMmin=9, logM200Mean_min=12, logMstar_min=None,
        load_distances=False, load_velocities=False,
        load_hosts=False, load_history=False)
    subhalos = subs.catalog
    sats = subs.satellites
    cents = subs.centrals
    ic0(sats['TrackId'].shape)
    ic0(cents['TrackId'].shape)
    track_ids = subs.satellites['TrackId']
    host_halo_id = sats['HostHaloId'].to_numpy(dtype=np.int32)
    ic0(host_halo_id, host_halo_id.shape)
    cent_track_id = cents['TrackId'].to_numpy(dtype=np.int32)
    ic(cent_track_id, cent_track_id.shape)
    ti = time()
    host_track_id_today = np.array(
        [cent_track_id[cents['HostHaloId'] == hhi][0]
         for hhi in host_halo_id])
    print(f'host tracks in {time()-ti:.2f} s')
    ic0(host_track_id_today[:5])


    groups = ('trackids', 
              'max_Mbound', 'max_Mstar', 'max_Mdm', 'max_Mgas',
              #'max_Mstar/Mbound', 'max_Mdm/Mbound', 'max_Mgas/Mbound',
              #'max_Mbound/Mstar', 'max_Mgas/Mstar',
              'birth', 'first_infall', 'last_infall', 'sat', 'cent')
    names = [('TrackId', 'TrackId_current_host', 'TrackId_previous_host')] \
            + (len(groups)-1) \
                * [('isnap', 'time', 'z', 'Mbound',
                    'Mstar', 'Mdm', 'Mgas', 'M200Mean')]
    dtypes = [3*[np.int32]] + (len(groups)-1)*[[np.int16] + 7*[np.float32]]
    cols = {group: [f'{group}:{name}' for name in grnames]
            for group, grnames in zip(groups, names)}
    # DataFrame
    history = {
        f'{gr}:{name}': -np.ones(subs.nsat, dtype=dt)
        for gr, grnames, dtype in zip(groups, names, dtypes)
        for name, dt in zip(grnames, dtype)}
    ic0(history)
    history['trackids:TrackId'] = track_ids
    history['trackids:TrackId_current_host'] = host_track_id_today
    history = pd.DataFrame(history).reset_index()
    ic0(np.sort(history.columns))
    ic0(history.shape)
    # testing
    jtr = (history['trackids:TrackId'] == 31158)
    jtr_cols = ['trackids:TrackId', 'max_Mstar:isnap',
                'max_Mstar:time', 'max_Mstar:Mstar']
    ic0(jtr.sum())

    # save everything to an hdf5 file
    path_history = os.path.join(sim.data_path, 'history')
    hdf = os.path.join(path_history, 'history.h5')
    if os.path.isfile(hdf):
        os.system(f'cp -p {hdf} {hdf}.bak')
        os.system(f'rm {hdf}')

    ## now calculate things!
    snaps = np.sort(sim.snapshots)[::-1]

    z, t = np.zeros((2,snaps.size))
    selection = ('TrackId', 'Rank', 'HostHaloId', 'Mbound', 'MboundType')
    failed = []
    for i, snap in enumerate(tqdm(snaps)):
        ics['history'](i, snap)
        subs_i = reader.LoadSubhalos(snap)#, selection=selection)
        # snapshot 281 is missing
        if subs_i.size == 0:
            ics['history']('no data')
            continue
        subs_i = Subhalos(
            subs_i, sim, snap, logMmin=None, logM200Mean_min=None,
            logMstar_min=None, load_any=False, verbose_when_loading=False)
        ic0(type(subs_i), subs_i.shape)
        z[i] = sim.redshift(snap)
        t[i] = sim.cosmology.lookback_time(z[i]).to('Gyr').value

        # first match subhaloes now and then
        this = history.merge(
            subs_i.catalog, left_on='trackids:TrackId', right_on='TrackId',
            how='left', suffixes=('_test', '_snap'))
        ics['history'](this.shape)
        this = Subhalos(
            this, subs_i.sim, subs_i.isnap, load_any=False, logMmin=None,
            logM200Mean_min=None, logMstar_min=None, exclude_non_FoF=False)
        if i == 0:
            ic(np.sort(this.columns))
        ics['history'](this.shape, subs_i.shape, history.shape)

        ## sat
        # this one gets updated all the time
        still_sat = np.array(this['Rank'] > 0)
        ics['sat'](still_sat.sum())
        history.loc[still_sat, cols['sat'][:3]] = [snap, t[i], z[i]]
        history = assign_masses(history, this, still_sat, 'sat')

        ## cent
        # this one is updated only once per subhalo
        if i > 0:
            new_cent = (this['cent:isnap'] == -1) & (this['Rank'] == 0)
            ics['cent'](new_cent.sum())
            # ics['cent'](this['TrackId'][new_cent])
            # ics['cent'](history['trackids:TrackId'][new_cent])
            #return
            history.loc[new_cent, cols['cent'][:3]] = [snap, t[i], z[i]]
            history = assign_masses(history, this, new_cent, 'cent')
            all_cent_history = (history['cent:isnap'] > -1)
            ev = 'cent'
            ic_cols = ['trackids:TrackId'] \
                + [f'{ev}:{x}' for x in ('isnap','z','time','Mbound','Mstar')]
            #ics['cent'](history.loc[all_cent_history, ic_cols])
        #continue

        history = max_mass(history, this, groups, cols, snap, t[i], z[i])
        ics['masses'](this.catalog.loc[jtr, ['TrackId','MboundType4']])
        ics['masses'](history.loc[jtr, jtr_cols])

        ## birth
        # this is updated all the time while the subhalo exists
        # doing this every time so we can record the mass at birth
        # and so that satellites that already existed in the first snapshot
        # do get assigned a birth time
        exist = np.in1d(history['trackids:TrackId'], subs_i['TrackId'])
        ics['birth'](exist.sum(), (1-exist).sum())
        history.loc[exist, cols['birth'][:3]] = [snap, t[i], z[i]]
        history = assign_masses(history, this, exist, 'birth')

        ## infall
        # first map HostHaloId to TrackId
        # seems like I need to ask for HostHaloId > -1 to avoid
        # multiple matches but what if a present-day satellite
        # was last a central with HostHaloId = -1?
        this_cent_mask = (subs_i['Rank'] == 0) & (subs_i['HostHaloId'] > -1)
        this_cent = subs_i.catalog.loc[this_cent_mask, ['TrackId','HostHaloId']]
        ics['infall'](this_cent_mask.shape, this_cent_mask.sum())
        ics['infall'](this.shape)
        this = this.merge(
            this_cent, on='HostHaloId', how='left',
            suffixes=('_sat','_cent'))
        ics['infall'](this.shape)
        # first update the masses for all subhalos for which we have
        # not registered infall already
        jinfall_last = (this['last_infall:isnap'] == -1).values
        ics['infall'](jinfall_last.shape, jinfall_last.sum())
        history = assign_masses(history, this, jinfall_last, 'last_infall')
        # if they are no longer in the same host, then register
        # the other columns. Once we've done this their masses
        # will never be updated again
        host_track = history['trackids:TrackId_current_host']
        try:
            jinfall_last = jinfall_last & (this['TrackId_cent'] != host_track)
        except ValueError as e:
            ics['infall'](
                jinfall_last.shape, jinfall_last.sum(),
                this['TrackId_cent'].shape, host_track.shape)
            print(f'*** ValueError({e}) ***')
            failed.append(snap)
            if len(failed) == 3:
                print(f'First failed: {failed}. Exiting.')
                return
            raise ValueError(e)
            continue
        ics['infall'](jinfall_last.sum())
        history.loc[jinfall_last, cols['last_infall'][:3]] \
            = [snap+1, t[i-1], z[i-1]]
        # assign these same values to first_infall
        history = assign_masses(history, this, jinfall_last, 'first_infall')
        history.loc[jinfall_last, cols['first_infall'][:3]] \
            = [snap+1, t[i-1], z[i-1]]
        # here we record the first time each subhalo fell into its
        # current host, i.e., the same as above but the info can be
        # updated once set
        left_host_and_back = (history['last_infall:isnap'] != -1) \
            & (this['TrackId_cent'] == host_track)
        ics['infall'](left_host_and_back.sum())
        history.loc[left_host_and_back, cols['first_infall'][:3]] \
            = [snap, t[i], z[i]]
        history = assign_masses(
            history, this, left_host_and_back, 'first_infall')
        # TrackId_previous_host refers to the host prior to the first
        # time the subhalo entered its current host (does it make a
        # difference?), i.e. those for which we've already registered
        # an infall in the *immediate* previous iteration
        # this will be updated each time the subhalo leaves
        jinfall_first = (history['first_infall:isnap'] == snap+1) \
            & (this['TrackId_cent'] != host_track)
        ics['infall'](jinfall_first.sum())
        history.loc[jinfall_first, 'trackids:TrackId_previous_host'] \
            = this['TrackId_cent'][jinfall_first]

        #ic(np.unique(history['cent/isnap'], return_counts=True))
        #ic(np.sort(history['sat/Mbound']))
        #ic((history > -1).sum())
        #ic(history.mean())
        if (args.store or True) and i % 20 == 0:
            store_h5(hdf, groups, names, history)
            #store_npy(path, groups, names, history)
            #break
        if args.test and i >= 3:
            break

    print(f'Failed: {failed}, i.e., {len(failed)} times')
    if args.store or args.test:
        store_h5(hdf, groups, names, history)
        print(f'Finished! Stored everything to {hdf}.')

    return


def assign_masses(history, this, mask, group):
    idx = np.arange(mask.size, dtype=int)[mask]
    for m in ('Mbound', 'Mgas', 'Mdm', 'Mstar'):
        history.loc[mask, f'{group}:{m}'] = this[m][mask]
    return history


def max_mass(history, this, events, cols, snap, ti, zi):
    #ics['masses']()
    #ic(this)
    for event in events:
        if event[:4] == 'max_':
            m = event[4:]
            #ics['masses'](event, m)
            # define before splitting m
            histdata = history[f'{event}:{m}']
            m = m.split('/')
            thisdata = this[m[0]]
            if len(m) == 2:
                thisdata = thisdata / this[m[1]]
            m = '/'.join(m)
            # greater or eaqual than because I'm going backwards
            # in time (i.e., I want to record the earliest snapshot
            # where the maximum mass is obtained)
            gtr = (thisdata >= histdata)
            # this is working
            history.loc[gtr, cols[event][:3]] = [snap, ti, zi]
            # but this is not
            history = assign_masses(history, this, gtr, event)
            #history.loc[gtr, key] = thisdata.loc[gtr]
            #if m == 'Mbound':
                #ics['masses'](history.loc[gtr, cols[event]])
    return history


def store_h5(hdf, groups, names, data):
    with h5py.File(hdf, 'w') as out:
        for group, grnames in zip(groups, names):
            gr = out.create_group(group)
            for name in grnames:
                col = f'{group}:{name}'
                gr.create_dataset(
                    name, data=data[col].to_numpy(), dtype=data[col].dtype)
    ic0(f'Saved to {hdf}')
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
