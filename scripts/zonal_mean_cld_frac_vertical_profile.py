from __future__ import print_function
import os
import numpy as np
import xarray as xr
import proplot as plot
import matplotlib.pyplot as plt
import copy
from analysis_functions import add_datetime_info, get_unique_line_labels, z_to_p

def zonal_mean_cld_frac_vertical_profile(ds_arr, exp_names, line_styles, figname=None):
    
    # ------------- Read CALIPSO dataset ------------- #
    base_dir = '/scratch/ql260/obs_datasets/'
    cf_calipso = xr.open_dataset(os.path.join(base_dir, 'GOCCP_v3',
                '3D_CloudFraction330m_200606-201711_avg_CFMIP2_sat_3.1.2.nc'),
                decode_times=False, autoclose=True)
    add_datetime_info(cf_calipso)
    clcalipso = cf_calipso.clcalipso.where(np.logical_and(cf_calipso.year>=2007,
                cf_calipso.year<=2015), drop=True)
    p_calipso = z_to_p(cf_calipso.alt_mid * 1e3)
    cf_zm_calipso1 = clcalipso.mean(('time', 'longitude'))
    cf_zm_calipso = np.ma.MaskedArray(cf_zm_calipso1, mask=np.isnan(cf_zm_calipso1))
    coslat2 = np.cos(np.deg2rad(clcalipso.latitude))
    calipso_cf_profile = np.ma.average(cf_zm_calipso, axis=1, weights=coslat2)

    # ------------- ERA interim reanalysis ------------- #
    fnm_cld_frac = os.path.join(base_dir, 'ecmwf_data', 'ecmwf_cld_frac_1979_2017_t42.nc')
    ds_cf_era = xr.open_dataset(fnm_cld_frac)
    cld_frac = ds_cf_era.cc.mean('time')
    cld_frac_zm_profile = cld_frac.mean('lon')
    levels_era = ds_cf_era.level
    coslat_era = np.cos(np.deg2rad(ds_cf_era.lat))
    # Calculate zonal mean profile
    cf_zm_era = np.ma.MaskedArray(cld_frac_zm_profile, mask=np.isnan(cld_frac_zm_profile))
    cf_profile_era = np.ma.average(cf_zm_era, axis=1, weights=coslat_era)

    lats_arr = [clcalipso.latitude, ds_cf_era.lat]
    levels_arr = [p_calipso, ds_cf_era.level]
    cf_zm_arr = [cf_zm_calipso, cf_zm_era]

    # ===================================== #
    plot.close('all')
    fig, axes = plot.subplots(nrows=3, ncols=3, aspect=(1.2, 1), share=1) #axwidth=5, 

    ylim = [0, 1000]
    # global zonal mean
    ax = axes[0]
    var_name = 'cf'
    lines = []
    for ds, exp_name, line_style in zip(ds_arr, exp_names, line_styles):
        pfull = ds.pfull
        coslat = np.cos(np.deg2rad(ds.lat))
        cf = ds[var_name]
        cf_ma = np.ma.MaskedArray(cf, mask=np.isnan(cf))
        cf_zm = np.ma.average(cf_ma, axis=(0,3))
        coslat = np.cos(np.deg2rad(ds.lat))
        cf_zm_profile = np.ma.average(cf_zm, axis=1, weights=coslat)

        l = ax.plot(cf_zm_profile*1e2, pfull.values, linestyle=line_style, linewidth=1, label=exp_name)
        lines.extend(l)
        ax.set_ylabel('Pressure (hPa)')
        ax.set_title('(a) Cloud fraction (%)')
        ax.set_xlim([0,30])
        ax.set_ylim(ylim)
        ax.invert_yaxis()

        # Append for zonal mean plot
        cf_zm_arr.append(cf_zm)
        lats_arr.append(ds.lat)
        levels_arr.append(pfull)

    # Add obsrvation from GOCCP-CALIPSO
    l = ax.plot(calipso_cf_profile*1e2, p_calipso, 'C6-', linewidth=2, label='CALIPSO')
    lines.extend(l)
    l = ax.plot(cf_profile_era*1e2, levels_era, 'C7-', linewidth=2, label='ERA interim')
    lines.extend(l)

    new_lines, new_labels = get_unique_line_labels(lines)
    fig.legend(new_lines, new_labels, ncol=3, loc='b', cols=(1,2))

    # =========================================================================== #
    # Plot 2d cloud fraction profile
    cf_labels = copy.deepcopy(exp_names)
    cf_labels.insert(0, 'CALIPSO-GOCCP')
    cf_labels.insert(1, 'ERA-Interim')

    cnlevels = np.arange(0, 101, 5)
    colormap='Dusk'
    for k, (ax, cf, label, lats, levels) in enumerate(zip(axes[1:], cf_zm_arr, cf_labels, lats_arr, levels_arr)):
        lats_2d, plevel_2d = np.meshgrid(lats, levels)
        cs = ax.contourf(lats_2d, plevel_2d, cf*1e2, levels=cnlevels, cmap=colormap)
        ax.set_title('('+chr(97+1+k)+') '+label)
        ax.set_ylim(ylim)
        ax.invert_yaxis()

    axes[1:].format(xlabel='Latitude', ylabel='Pressure (hPa)', 
        xlim=(-90, 90), xlocator=plot.arange(-60, 61, 30), xminorlocator=30, 
        xformatter='deglat', xtickminor=False)

    axes.format(yminorlocator=100, grid=False)

    #https://proplot.readthedocs.io/en/latest/api/proplot.figure.Figure.colorbar.html?highlight=colorbar
    fig.colorbar(cs, loc='b', col=3, label='Cloud fraction (%)') 

    fig.tight_layout()
    if figname is None:
        plt.show()
    else:
        fig.savefig(figname, bbox_inches='tight', pad_inches=0.05, transparent=False)
