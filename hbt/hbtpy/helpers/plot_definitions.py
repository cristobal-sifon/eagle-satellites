"""
Global plotting definitions (bad idea, I know!)
"""
import numpy as np

ccolor = 'C3'
scolor = 'C1'

massnames = ['Mbound', 'Mstar', 'Mdm', 'Mgas', 'Mass', 'M200', 'MVir']
units = {'mass': r'\mathrm{M}_\odot', 'time': '\mathrm{Gyr}',
         'distance': 'h^{-1}\mathrm{Mpc}'}
xbins = {
    'ComovingMostBoundDistance': np.logspace(-2, 0.5, 9),
    #'logComovingMostBoundDistance': np.logspace(-2, 0.5, 9),
    'M200Mean': np.logspace(13, 14.7, 6),
    'mu': np.logspace(-5, 0, 9),
    'Mstar': np.logspace(7.7, 11.3, 9),
    'Mbound': np.logspace(8, 12.5, 11),
    'time': np.arange(0, 13.5, 2),
    }
binlabel = {
    #'ComovingMostBoundDistance': '$R_\mathrm{com}$ ($h^{-1}$Mpc)',
    'ComovingMostBoundDistance': 'R',
    'R200Mean': r'R_\mathrm{200m}',
    'M200Mean': r'M_\mathrm{200m}',
    'mu': r'\mu',
    'Mbound': r'm_\mathrm{sub}',
    'Mstar': r'm_{\!\star}',
    'Mdm': r'm_\mathrm{DM}',
    'Mgas': r'm_\mathrm{gas}',
    'LastMaxMass': 'm_\mathrm{sub,max}',
    'time': 't_\mathrm{lookback}'
    }
_xx = 'ComovingMostBoundDistance'
for i in '012':
    xbins[f'{_xx}{i}'] = xbins[_xx]
    binlabel[f'{_xx}{i}'] = f'{binlabel[_xx]}_{i}'

events = {'last_infall': 'acc', 'first_infall': 'infall',
          'cent': 'cent', 'sat': 'sat'}
for event, event_label in events.items():
    h = f'history:{event}'
    infall_label = f'\mathrm{{{event_label}}}'
    binlabel[f'{h}:Mbound'] = rf'm_\mathrm{{sub}}^{infall_label}'
    binlabel[f'{h}:Mstar'] = rf'm_\star^{infall_label}'
    binlabel[f'{h}:Mdm'] = fr'm_\mathrm{{DM}}^{infall_label}'
    binlabel[f'{h}:Mgas'] = rf'm_\mathrm{{gas}}^{infall_label}'
    binlabel[f'{h}:time'] = rf't_\mathrm{{lookback}}^{infall_label}'
    xbins[f'{h}:Mbound/{h}:Mstar'] = np.logspace(1, 2.7, 6)
    xbins[f'Mstar/{h}:Mbound'] = np.logspace(-4, -1, 6)
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
