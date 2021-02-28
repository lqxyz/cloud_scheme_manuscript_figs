from __future__ import print_function
import numpy as np
import xarray as xr
import pandas as pd
import cmaps
import warnings
warnings.simplefilter(action='ignore')
import matplotlib.pyplot as plt
import proplot as plot
from matplotlib import cm
from mpl_toolkits.basemap import Basemap
import string
from scipy import interpolate

def interpolate_obs_dict(obs_dict, lats, lons):
    obs_remap_dict = {}
    for nm, dt in obs_dict.items():
        obs_lats = dt.lat
        obs_lons = dt.lon
        if not(len(lats)==len(obs_lats) and len(lons)==len(obs_lons)):
            try:
                times = dt.time
                ntime = len(times)
                intp_dt = np.zeros((ntime,len(lats),len(lons)))
                for t in range(ntime):
                    fintp = interpolate.interp2d(obs_lons, obs_lats, np.array(dt[t,]))
                    intp_dt[t,:,:] = fintp(lons, lats)
                obs_remap_dict[nm] = xr.DataArray(intp_dt, coords=[times,lats,lons], dims=['time','lat','lon'])
            except:
                #print('Interp no time!')
                intp_dt = np.zeros((len(lats),len(lons)))
                fintp = interpolate.interp2d(obs_lons, obs_lats, np.array(dt))
                intp_dt = fintp(lons, lats)
                obs_remap_dict[nm] = xr.DataArray(intp_dt, coords=[lats,lons], dims=['lat','lon'])
        else:
            # print('no interp')
            obs_remap_dict[nm] = dt
    return obs_remap_dict

def plot_latlon_with_map(ax, dt, cmap='RdBu_r', cnlevels=None,
            extend='neither', add_cbar=True,
            cbar_loc='bottom', cbar_pad=0.3, title=None):
    """
    Plot the single lat/lon map in ax.
    """
    lats = dt.lat.values
    lons = dt.lon.values
    lons_2d, lats_2d = np.meshgrid(lons, lats)
    m = Basemap(lon_0=180, ax=ax, resolution='c') #(projection='robin', lon_0=0., ax=ax, resolution='c')
    m.drawcoastlines()
    m.drawparallels(np.arange(-60.,61.,30.),   labels=[1,0,0,0], linewidth=0)
    m.drawmeridians(np.arange(-180.,181.,60.), labels=[0,0,0,1], linewidth=0)
    xi, yi = m(lons_2d, lats_2d)
    if cnlevels is None:
        cs = m.contourf(xi, yi, dt, cmap=cmap)
    else:
        cs = m.contourf(xi, yi, dt, cmap=cmap, levels=cnlevels, extend=extend)
    if add_cbar:
        m.colorbar(cs, location=cbar_loc, pad=cbar_pad)
    ax.set_title(title, loc='left')

    return cs

def  plot_multiple_latlon_maps(dt_arr, title_arr, nrows=2, ncols=3,
        units_arr=None, units=None, cmap_arr=None, cmap=None,
        cnlevels_arr=None, cnlevels=None, extend_arr=None, extend=None, 
        width=5, height=3.5, title_add_gm=False, fig_name=None):
    """
    Plot multiple lat/lon maps.
    """

    plot.close()
    fig, axes = plot.subplots(nrows=nrows, ncols=ncols) #, axwidth=5, aspect=(1.5, 1))

    for kk, (ax, dt, title) in enumerate(zip(axes, dt_arr, title_arr)):
        # Determine the cmap, cnlevels and extend
        if cmap_arr is not None:
            i_cmap = cmap_arr[kk]
        else:
            if cmap is None:
                print('cmap should not be None if cmap_arr is None.')
            else:
                i_cmap = cmap

        if cnlevels_arr is not None:
            i_cnlevels = cnlevels_arr[kk]
        else:
            i_cnlevels = cnlevels

        if extend_arr is not None:
            i_extend = extend_arr[kk]
        else:
            if extend is None:
                print('extend should not be None if extend_arr is None.')
            else:
                i_extend = extend
        
        if units_arr is not None:
            i_units = units_arr[kk]
        else:
            if units is None:
                print('units should not be None if units_arr is None.')
            else:
                i_units = units

        # Prepare the title string
        prefix = '('+string.ascii_lowercase[kk]+') '+title.replace('cldamt', 'cloud amount')
        if title_add_gm:
            coslat = np.cos(np.deg2rad(dt.lat))
            dt_gm = np.average(dt.mean('lon'), axis=0, weights=coslat)
            val_str = ' (%.2f'%(dt_gm) + i_units + ')'
        else:
            val_str = ''
        i_title = prefix + val_str
        
        # call func to plot single map
        cs = plot_latlon_with_map(ax, dt, cmap=i_cmap, cnlevels= i_cnlevels, 
                extend=i_extend, title=i_title, add_cbar=False)
        if kk==2 or kk==5 or kk==8:
            if kk==2 and 'amt' in title.lower():
                mm = 4
            else:
                mm = 2
            axes[kk].colorbar(cs, loc='r', ticks=i_cnlevels[::mm], label=i_units, width='1em')

    # save and show figure
    if fig_name is not None:
        fig.tight_layout()
        fig.savefig(fig_name, bbox_inches='tight', pad_inches=0.05, transparent=False)
        fig.savefig(fig_name.replace('.pdf', '.png'), bbox_inches='tight', pad_inches=0.05, transparent=False)
        fig.savefig(fig_name.replace('.pdf', '.eps'), bbox_inches='tight', pad_inches=0.05, transparent=False)

    plt.show()

def cmp_spatial_patterns_from_exps_and_obs(ds_arr, obs_dict, var_name, fig_name, 
            exp_names=['LS', 'FD', 'ALL'],
            obs_name='Obs', coeff=1.0, title_nm=None, cmap1=None, cmap2=None, l_title_add_gm=False):

    dt_arr = []
    title_arr = []

    if 'amt' in var_name:
        var_name1 = var_name
    elif 'sw_up' in var_name or 'olr' in var_name or 'flux' in var_name:
        var_name1 = 'soc_' + var_name
    else:
        var_name1 = var_name

    ds_arr_tm = []
    for ds in ds_arr:
        ds_arr_tm.append(ds[var_name1].mean('time') * coeff)

    # ----- FIRST ROW ----- #
    for ds, exp_nm in zip(ds_arr_tm, exp_names):
        dt_arr.append(ds)
        title_arr.append(title_nm+' ('+exp_nm+')')
    
    # ----- SECOND ROW ----- #
    obs_dict1 = interpolate_obs_dict(obs_dict, ds_arr[0].lat, ds_arr[0].lon)
    if 'lwp' in var_name or 'iwp' in var_name or 'cwp' in var_name:
        obs = obs_dict1[var_name] * coeff
    else:
        obs = obs_dict1[var_name].mean('time') * coeff

    dt_arr.append(obs)
    title_arr.append(title_nm+' ('+obs_name+')')

    # exp diff
    dt_arr.append(ds_arr_tm[1]-ds_arr_tm[0])
    title_arr.append(title_nm+' ('+exp_names[1] + ' - '+ exp_names[0]+')')

    dt_arr.append(ds_arr_tm[2]-ds_arr_tm[1])
    title_arr.append(title_nm+' ('+exp_names[2] + ' - '+ exp_names[1]+')')

    # ----- THIRD ROW ----- #
    # mod - obs
    for i in range(3):
        dt_arr.append(ds_arr_tm[i] - obs)
        if 'obs' in obs_name.lower():
            obs_nm = 'obs'
        else:
            obs_nm = obs_name
        title_arr.append(title_nm + ' ('+exp_names[i] + ' - ' + obs_nm + ')') 

    nrows = 3
    ncols = 3
    extend_arr = ['neither'] * 4 + ['both'] * 5

    cmap_arr = [cmaps.BlueWhiteOrangeRed ] * len(dt_arr)
    if cmap1 is not None:
        cmap_arr[0:4] = [cmap1] * 4
    else:
        cmap_arr[0:4] = [cmaps.amwg256] * 4
    if cmap2 is not None:
        cmap_arr[4:] = [cmap2] * 5
    
    units = ''
    if 'low_cld_amt' in var_name or 'high_cld_amt' in var_name or 'mid_cld_amt' in var_name:
        cnlevels_arr =  [np.arange(0,101,5)] * 4 + [np.arange(-40,41,5)] * 2 + [np.arange(-60,61,10)] * 3
        units = '%'
    if 'tot_cld_amt' in var_name:
        cnlevels_arr =  [np.arange(0,101,5)] * 4 + [np.arange(-40,41,5)] * 2 + [np.arange(-60,61,10)] * 3
        units = '%'
    if 'toa_sw_up' in var_name:
        cnlevels_arr =  [np.arange(60,171,10)] * 4 + [np.arange(-30,31,5)] * 2 + [np.arange(-60,61,10)] * 3
        units = 'Wm$^{-2}$'
    if 'olr' in var_name:
        cnlevels_arr =  [np.arange(100,301,15)] * 4 + [np.arange(-20,21,5)] * 2 + [np.arange(-60,61,10)] * 3
        units = 'Wm$^{-2}$'
        extend_arr = ['both']*9
    if 'toa_net_flux' in var_name:
        cnlevels_arr =  [np.arange(-140,141,20)] * 4 + [np.arange(-40,41,5)] * 2 + [np.arange(-60,61,10)] * 3
        units = 'Wm$^{-2}$'
    if 'toa_sw_cre' in var_name:
        cnlevels_arr =  [np.arange(-120,1,10)] * 4 + [np.arange(-30,31,5)] * 2 + [np.arange(-60,61,10)] * 3
        units = 'Wm$^{-2}$'
        extend_arr = ['min'] * 4 + ['both'] * 5
    if 'toa_lw_cre' in var_name:
        cnlevels_arr =  [np.arange(0,71,5)] * 4 + [np.arange(-30,31,5)] * 2 + [np.arange(-30,31,5)] * 3
        units = 'Wm$^{-2}$'
        extend_arr = ['max'] * 4 + ['both'] * 5
    if 'toa_net_cre' in var_name:
        cnlevels_arr =  [np.arange(-90,91,10)] * 4 + [np.arange(-30,31,5)] * 2 + [np.arange(-60,61,10)] * 3
        units = 'Wm$^{-2}$'
    if 'cwp' in var_name:
        cnlevels_arr =  [np.arange(0,360,20)] * 4 + [np.arange(-60,61,10)] * 2 + [np.arange(-150,151,20)] * 3
        units = 'g m$^{-2}$'
        extend_arr = ['max'] * 4 + ['both'] * 5
    if 'lwp' in var_name:
        cnlevels_arr =  [np.arange(0,200,20)] * 4 + [np.arange(-60,61,10)] * 2 + [np.arange(-120,121,20)] * 3
        units = 'g m$^{-2}$'
        extend_arr = ['max'] * 4 + ['both'] * 5

    plot_multiple_latlon_maps(dt_arr, title_arr, nrows=nrows, ncols=ncols, units=units,
                cmap_arr=cmap_arr, cnlevels_arr=cnlevels_arr, extend_arr=extend_arr,
                title_add_gm=l_title_add_gm, fig_name=fig_name)
