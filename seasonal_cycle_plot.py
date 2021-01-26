from __future__ import print_function
import os
import numpy as np
import xarray as xr
import proplot as plot
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import cmaps
import calendar
from scipy import interpolate
from analysis_functions import add_datetime_info, get_unique_line_labels, get_global_mean

def polar_cloud_amount_seasonal_cycle(ds_arr, exp_names, figname, obs_cf_dict=None, obs_cf_nm=None):
    '''
    cloud amount change in polar region: low and total cloud amount
    '''
    plot.close('all')
    fig, axes = plot.subplots(nrows=1, aspect=(1.8, 1), ncols=2,  share=1)

    # for north polar region
    months = np.arange(1,13,1)
    var_names = ['low_cld_amt', 'tot_cld_amt']
    titles = ['Low cloud amount', 'Total cloud amount']

    lines = []
    
    line_styles = ['C2-', 'C2--', 'C3-', 'C3--']
    for kk, (ax, var_nm, title) in enumerate(zip(axes, var_names, titles)):
        for ds, exp_nm, l_style in zip(ds_arr, exp_names, line_styles):
            lats = ds.lat
            lons = ds.lon

            l_lat = np.logical_and(ds.lat>60, ds.lat<=90)
            cc_polar = ds[var_nm].where(l_lat, drop=True)
            cc_polar_mon = cc_polar.groupby('month').mean('time')  
            cc_polar_gm = get_global_mean(cc_polar_mon)

            l = ax.plot(months, cc_polar_gm, l_style, label=exp_nm)
            lines.extend(l)
        
        ## plot observations
        if obs_cf_dict is not None:
            obs_cf = obs_cf_dict[var_nm]
            add_datetime_info(obs_cf)
            l_lat = np.logical_and(obs_cf.lat>60, obs_cf.lat<=90)
            cc_polar = obs_cf.where(l_lat, drop=True)
            cc_polar_mon = cc_polar.groupby('month').mean('time')  
            cc_polar_gm = get_global_mean(cc_polar_mon)

            if obs_cf_nm is None:
                obs_cf_nm = 'Obs'
            l = ax.plot(months, cc_polar_gm, 'k:', label=obs_cf_nm)
            lines.extend(l)

        ax.set_title('('+chr(97+kk)+') '+title, loc='left')
        ax.set_xticks(months)
        mon_labels = [calendar.month_abbr[i][0] for i in months]
        ax.set_xticklabels(mon_labels)

    axes.format(xlabel='Month', ylabel='Cloud amount (%)', 
        xlim=(1, 12), xlocator=plot.arange(1, 13, 1), xminorlocator=1,
        ylim=(0, 100), xtickminor=False, grid=False)

    new_lines, new_labels = get_unique_line_labels(lines)
    axes[-1].legend(new_lines, new_labels, loc='lc', ncol=1)
 
    fig.tight_layout()
    fig.savefig(figname, bbox_inches='tight', pad_inches=0.1, transparent=False)
    plot.rc.reset()

def cre_seasonal_cycle(obs_dt, ds_arr, exp_names, figname):

    dst_dt_arr = []
    title_arr = []

    cre_nms = ['toa_lw_cre', 'toa_sw_cre', 'toa_net_cre']
    cre_strs = ['LW CRE', 'SW CRE', 'Net CRE']

    for cre_nm, cre_str in zip(cre_nms, cre_strs):
        # Read obs and interpolate
        dims = obs_dt[cre_nm].dims
        obs_times = obs_dt[cre_nm][dims[0]]
        obs_lats = obs_dt[cre_nm][dims[1]]
        obs_lons = obs_dt[cre_nm][dims[2]]
        # interpolate
        lats = ds_arr[0].lat
        lons = ds_arr[0].lon
        times = ds_arr[0].time

        # get months
        months = np.arange(1,13,1)
        obs_var = np.zeros((len(obs_times), len(lats), len(lons)))
        # print('obs time=', len(obs_times))
        for tt in range(len(obs_times)):
            fint = interpolate.interp2d(obs_lons, obs_lats, obs_dt[cre_nm][tt,])
            obs_var[tt,] = fint(lons, lats)
        obs_var = xr.DataArray(obs_var, coords=[months, lats, lons], dims=['month', 'lat', 'lon'])
        #obs_var.coords['mask']=(('lat','lon'), ds_mask.land_mask.values)
        
        dst_dt_arr.append(obs_var)
        title_arr.append('Obs ('+cre_str+')')
        
        for nm, ds in zip(exp_names, ds_arr):
            dst_dt = ds[cre_nm].groupby('month').mean('time')
            dst_dt_arr.append(dst_dt)
            title_arr.append(nm+' ('+cre_str+')')

    nrows = 2
    ncols = 3

    ## Evaluate an existing colormap from 0.5 (midpoint) to 1 (upper end)
    cmap_tmp = plt.get_cmap('RdBu_r')
    colors = cmap_tmp(np.linspace(0.5, 1, cmap_tmp.N // 2))
    ## Create a new colormap from those colors
    cmap_Rd = LinearSegmentedColormap.from_list('Upper Half', colors)

    # Seasonal Cycle
    cnlevels_arr = [np.arange(0, 61, 5), np.arange(-150, 1, 10), np.arange(-90, 91, 10),]# np.arange(0, 310, 20)]
    extend_strs = ['max', 'min', 'both',] # 'max']
    cmaps_arr = [cmap_Rd, cmaps.MPL_Blues_r, 'RdBu_r']
    units = ['W/m$^2$', 'W/m$^2$', 'W/m$^2$',] 
    
    plot.close()
    nrows = 3
    ncols = 3
    fig, axes = plot.subplots(nrows=nrows, ncols=ncols, aspect=(1.8, 1))

    for kk, (ax, dst_dt, title) in enumerate(zip(axes[:], dst_dt_arr, title_arr)):
        nn = kk // ncols
        cnlevels = cnlevels_arr[nn]
        extend = extend_strs[nn]
        unit = units[nn]
        cmap = cmaps_arr[nn]

        time_1d = np.arange(1, 13, 1)

        time_2d, lat_2d = np.meshgrid(time_1d, dst_dt.lat)
        cs = ax.contourf(time_2d, lat_2d, np.swapaxes(np.nanmean(dst_dt, axis=2), 1, 0),
                         levels=cnlevels, cmap=cmap, extend=extend)
        if np.mod(kk, ncols) == ncols-1:
            cbar = ax.colorbar(cs, loc='r', label='Wm$^{-2}$', labelsize=9)
            cbar.ax.tick_params(labelsize=9)

        xlabels = [calendar.month_abbr[x][0] for x in range(1,13)]
        ax.set_xticks(np.arange(1,13,1))
        ax.set_xticklabels(xlabels)
        ax.set_title('('+chr(97+kk)+') '+title, loc='left')

    axes.format(xlabel='Month', ylabel='Latitude', 
        xlim=[1, 12], xlocator=np.arange(1, 13, 1), xminorlocator=1, xtickminor=False,
        ylim=(-90, 90), ylocator=plot.arange(-90, 91, 30), yminorlocator=10, 
        ytickminor=False, yformatter='deglat', grid=False)

    fig.tight_layout()
    
    fig.savefig(figname, bbox_inches='tight', pad_inches=0.1, transparent=False)
    # fig.savefig(figname.replace('.pdf', '.png'), bbox_inches='tight', pad_inches=0.05, transparent=False)