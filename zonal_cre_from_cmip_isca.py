from __future__ import print_function
import numpy as np
import xarray as xr
import os
import matplotlib.pyplot as plt
#from itertools import product
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
#import cmaps
import calendar
import proplot as plot
from analysis_functions import get_unique_line_labels


def cmp_zonal_mean_CREs(ds_arr, line_labels, obs_cre, fig_name,
        line_styles=None, obs_names=None, cmip_dict=None, obs_dict_keys=None):

    var_names = ['toa_sw_cre', 'toa_lw_cre', 'toa_net_cre',]
    var_titles = ['TOA SW CRE', 'TOA LW CRE', 'TOA net CRE']
    ylims = [[-110, 5], [-5, 50], [-60, 25]]

    obs_dict = {}
    if obs_names is None:
        obs_names = ['CERES_EBAF Ed4.0', 'ERA-Interim']
    for obs_nm in obs_names:
        obs_dict[obs_nm] = obs_cre[obs_nm]

    plot.rc.margin = 0.05

    plot.close('all')
    fig, axes = plot.subplots(nrows=3, ncols=1, axwidth=5, aspect=(2.5, 1), share=1) #   

    if obs_dict_keys is None:
        obs_dict_keys = var_names
    if line_styles is None:
        line_styles = ['-']*len(ds_arr)
    
    xlabel = 'Latitude'
    ylabel = r'Flux (Wm$^{-2}$)'

    lines = []
    for i, (ax, var_nm, title, obs_dict_key, ylim) in enumerate(zip(axes, 
                            var_names, var_titles, obs_dict_keys, ylims)):
        # Add CRE from CMIP models (with MME)
        if cmip_dict is not None:
            for kk, (mod_nm, dt_dict) in enumerate(cmip_dict.items()):
                lats = dt_dict[obs_dict_key].lat
                dims = dt_dict[obs_dict_key].dims
                if 'time' in dims:
                    tmp_cre_dt = dt_dict[obs_dict_key].mean(('time'))
                else:
                    tmp_cre_dt = dt_dict[obs_dict_key]
                
                if 'lon' in dims:
                    zm_cre = tmp_cre_dt.mean(('lon'))
                else:
                    zm_cre = tmp_cre_dt

                if 'MME' in mod_nm:
                    l = ax.plot(lats, zm_cre, 'k-', linewidth=2, label='Multimodel mean')
                else:
                    l = ax.plot(lats, zm_cre, 'darkgray', linewidth=1, label='CMIP5 models')
                lines.extend(l)

        # Add CRE from observation and reanalysis to figure
        if obs_dict is not None:
            for kk, (obs_nm, dt_dict) in enumerate(obs_dict.items()):
                l = ax.plot(dt_dict[obs_dict_key].lat, 
                        dt_dict[obs_dict_key].mean(('time', 'lon')),
                        'C'+str(6+kk)+'-', linewidth=2, label=obs_nm.replace('_', '-'))
                lines.extend(l)

        for kk, (ds, label, line_style) in enumerate(zip(ds_arr, line_labels, line_styles)):
            lats = ds.lat
            dt = ds[var_nm].mean(('time', 'lon'))
            l = ax.plot(lats, dt, color='C'+str(kk), linestyle=line_style, linewidth=2, label=label)
            lines.extend(l)

        ax.format(title='('+chr(97+i)+') '+title, ytickminor=True, ylim=ylim)

    axes.format(xlabel=xlabel, ylabel=ylabel, 
        xlim=(-90, 90), xlocator=plot.arange(-90, 91, 30), xminorlocator=10,
        xformatter='deglat', xtickminor=False, grid=False)

    new_lines, new_labels = get_unique_line_labels(lines)
    fig.legend(new_lines, new_labels, loc='b', frameon=False, ncol=4)

    fig.savefig(fig_name, bbox_inches='tight', pad_inches=0.1, transparent=False)

