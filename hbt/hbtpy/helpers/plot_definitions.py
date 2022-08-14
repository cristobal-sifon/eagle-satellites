"""
Global plotting definitions (bad idea, I know!)
"""
import numpy as np

ccolor = 'C1'
scolor = 'k'

massnames = ['Mbound', 'Mstar', 'Mdm', 'Mgas', 'Mass', 'M200', 'MVir']
units = {'mass': r'\mathrm{M}_\odot', 'time': '\mathrm{Gyr}',
         'distance': 'h^{-1}\mathrm{Mpc}'}
xbins = {
    'ComovingMostBoundDistance': np.logspace(-1.7, 0.5, 9),
    #'logComovingMostBoundDistance': np.logspace(-2, 0.5, 9),
    'M200Mean': np.logspace(13, 14.7, 6),
    'mu': np.logspace(-5, 0, 9),
    'Mstar': np.logspace(9, 11.5, 10),
    'Mgas': np.logspace(8, 12, 10),
    'Mbound': np.logspace(9, 13, 10),
    'time': np.arange(0, 13.5, 2),
    'z': np.array([0, 0.5, 1, 1.5, 2, 3, 5]),
    # some common ratios
    'ComovingMostBoundDistance/R200Mean': np.logspace(-1.7, 0.5, 9),
    'ComovingMostBoundDistance/R200MeanComoving': \
        np.append([0.02], np.logspace(-1.5, 0.5, 7)),
        #np.logspace(-1.7, 0.5, 9)
    }
ylims = {'Mbound/history:first_infall:Mbound': (5e-3, 2),
         'Mdm/history:first_infall:Mdm': (5e-4, 2),
         'Mstar/history:first_infall:Mstar': (0.1, 20),
         }
binlabel = {
    #'ComovingMostBoundDistance': '$R_\mathrm{com}$ ($h^{-1}$Mpc)',
    'ComovingMostBoundDistance': 'R',
    'R200Mean': r'R_\mathrm{200m}',
    'R200MeanComoving': r'R_\mathrm{200m}',
    'M200Mean': r'M_\mathrm{200m}',
    'mu': r'\mu',
    'Mbound': r'm_\mathrm{sub}',
    'Mstar': 'm_{\u2605}',
    'Mdm': r'm_\mathrm{DM}',
    'Mgas': r'm_\mathrm{gas}',
    'LastMaxMass': 'm_\mathrm{sub,max}',
    'time': 't_\mathrm{lookback}',
    'z': 'z',
    }
_xx = 'ComovingMostBoundDistance'
for i in '012':
    xbins[f'{_xx}{i}'] = xbins[_xx]
    binlabel[f'{_xx}{i}'] = f'{binlabel[_xx]}_{i}'
    xbins[f'{_xx}{i}/R200Mean'] = xbins[f'{_xx}/R200Mean']

events = {'last_infall': 'acc', 'first_infall': 'infall',
          'cent': 'cent', 'sat': 'sat'}
for event, event_label in events.items():
    h = f'history:{event}'
    infall_label = f'\mathrm{{{event_label}}}'
    binlabel[f'{h}:Mbound'] = rf'm_\mathrm{{sub}}^{infall_label}'
    binlabel[f'{h}:Mstar'] = f'm_{{\u2605}}^{infall_label}'
    binlabel[f'{h}:Mdm'] = fr'm_\mathrm{{DM}}^{infall_label}'
    binlabel[f'{h}:Mgas'] = rf'm_\mathrm{{gas}}^{infall_label}'
    binlabel[f'{h}:time'] = rf't_\mathrm{{lookback}}^{infall_label}'
    binlabel[f'{h}:z'] = rf'z^{infall_label}'
    xbins[f'{h}:Mbound/{h}:Mstar'] = np.logspace(1, 2.7, 6)
    xbins[f'Mstar/{h}:Mbound'] = np.logspace(-4, -1, 6)
    xbins[f'{h}:z'] = xbins['z']
axlabel = {}
for key, label in binlabel.items():
    ismass = np.any([mn in key for mn in massnames])
    if ismass or 'time' in key:
        label = label.replace('$', '')
        unit_key = 'mass' if ismass else (
            'time' if 'time' in key else 'distance')
        un = units[unit_key].replace('$', '')
        axlabel[key] = rf'${label}\,({un})$'
    elif 'distance' in key.lower():
        un = units['distance'].replace('$', '')
        axlabel[key] = rf'${label}\,({un})$'
    else:
        axlabel[key] = label
