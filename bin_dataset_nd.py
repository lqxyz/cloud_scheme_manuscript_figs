# Divide the data into different bins according to
# omega500, ELF, LTS or EIS

from __future__ import print_function
import numpy as np
import xarray as xr
import os
import sys
from scipy import interpolate
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore') #, category=FutureWarning)
import copy


def bin_4d_data(ds, bins, grp_time_var='year', bin_var_nm='ELF'):
    """
    ds: a xarray dataset containing several variables
    variables dimension: (time, pfull, lat, lon)
    """
    if grp_time_var is not None:
        ds_grp = ds.groupby(grp_time_var).mean('time')
        ntime = len(ds_grp[grp_time_var])
    else:
        ds_grp = ds
        ntime = len(ds.time)

    nlev = len(ds.pfull)
    nbins = len(bins)
    pdf_omega = np.zeros((ntime, nbins-1))
    ds_bin_mean = {}

    for var in ds.variables:
        if len(ds[var].shape) == 4:
            ds_bin_mean[var] = np.zeros((ntime, nlev, nbins-1))

    for i in range(ntime):
        for j in range(nlev):
            if grp_time_var is not None:
                ds_i = ds_grp.isel({grp_time_var:i, 'pfull':j})
            else:
                ds_i = ds_grp.isel({'time':i, 'pfull':j})
            if j==0:
                pdf_omega[i, :] = np.histogram(ds_i[bin_var_nm], bins=bins, density=True)[0]

            grouped = ds_i.groupby_bins(bin_var_nm, bins).mean()
 
            for var in ds.variables:
                if len(ds[var].shape) == 4:
                    ds_bin_mean[var][i,j,:] = grouped[var]

    return pdf_omega, ds_bin_mean


def bin_3d_data(ds, bins, grp_time_var='year', bin_var_nm='ELF'):
    """
    ds: a xarray dataset containing several variables
    """
    if grp_time_var is not None:
        ds_grp = ds.groupby(grp_time_var).mean('time')
        ntime = len(ds_grp[grp_time_var])
    else:
        ds_grp = ds
        ntime = len(ds.time)

    nbins = len(bins)
    pdf_omega = np.zeros((ntime, nbins-1))
    ds_bin_mean = {}

    for var in ds.variables:
        if len(ds[var].shape) == 3:
            ds_bin_mean[var] = np.zeros((ntime, nbins-1))

    for i in range(ntime):
        if grp_time_var is not None:
            ds_i = ds_grp.isel({grp_time_var: i})
        else:
            ds_i = ds.isel({'time':i})
        pdf_omega[i,:] = np.histogram(ds_i[bin_var_nm], bins=bins, density=True)[0]
        grouped = ds_i.groupby_bins(bin_var_nm, bins).mean()

        for var in ds.variables:
            v = grouped.variables.get(var)
            #if len(np.shape(v)) == 3:
            if len(ds[var].shape) == 3:
                ds_bin_mean[var][i,:] = v #grouped.variables.get(var)

    return pdf_omega, ds_bin_mean


def select_4d_data(ds_m, bin_data_dict, ds_mask, bins,
      bin_var_nm='ELF', land_sea='ocean', grp_time_var='year'):
    # 4d data
    t_var_names = ['dt_tg_condensation', 'dt_tg_diffusion',  'dt_tg_convection']
    q_var_names = [v.replace('_tg_', '_qg_') for v in t_var_names]

    cal_total_tendency(ds_m, t_var_names)
    cal_total_tendency(ds_m, q_var_names)

    t_var_names.append('dt_tg_sum_cond_diffu_conv')
    q_var_names.append('dt_qg_sum_cond_diffu_conv')

    for varnm in t_var_names:
        bin_data_dict[varnm] = ds_m[varnm]
    for varnm in q_var_names:
        bin_data_dict[varnm] = ds_m[varnm]

    four_d_varnames = ['cf', 'rh', 'sphum', 'theta', 'qcl_rad', 'omega',
                        'soc_tdt_lw', 'soc_tdt_sw', 'soc_tdt_rad',
                        'diff_m', 'diff_t']
    for vn in four_d_varnames:
        bin_data_dict[vn] = ds_m[vn]

    ds_bin_m = xr.Dataset(bin_data_dict,
                coords={'time': ds_m.time, 'pfull': ds_m.pfull, 
                        'lat': ds_m.lat, 'lon': ds_m.lon})

    ds_bin_m.coords['mask'] = (('lat', 'lon'), ds_mask.land_mask.values)

    for varname, da in ds_bin_m.data_vars.items():
        if land_sea == 'ocean': # or 'ocn' in land_sea:
            ds_bin_m[varname] = da.where(ds_bin_m.mask==0)
        if land_sea == 'land':
            ds_bin_m[varname] = da.where(ds_bin_m.mask==1)

    pdf_m, ds_bin_mean_m = bin_4d_data(ds_bin_m, bins, 
            grp_time_var=grp_time_var, bin_var_nm= bin_var_nm)

    # Write data in xarray dataset format
    bins_coord = (bins[0:-1]+bins[1:]) / 2.0

    if grp_time_var is not None:
        ds_grp = ds_m.groupby(grp_time_var).mean('time')
        coords = {grp_time_var: ds_grp[grp_time_var], 'pfull':ds_m.pfull, 'bin':bins_coord}
        dims = (grp_time_var, 'pfull', 'bin')
    else:
        coords = {'time':ds_bin_m.time, 'pfull':ds_m.pfull, 'bin':bins_coord}
        dims = ('time', 'pfull', 'bin')

    return pdf_m, ds_bin_mean_m, dims, coords
    

def select_3d_data(ds_m, bin_data_dict, ds_mask, bins,
      bin_var_nm='ELF', land_sea='ocean', grp_time_var='year'):


    three_d_varnames = ['soc_olr', 'soc_olr_clr', 'flux_lhe', 'flux_t',
                    'toa_sw_cre', 'toa_lw_cre', 'toa_net_cre',
                    'tot_cld_amt', 'low_cld_amt', 'mid_cld_amt', 'high_cld_amt', #] #, 'soc_tot_cloud_cover',
                    'z_pbl']
    for vn in three_d_varnames:
        bin_data_dict[vn] = ds_m[vn]

    ds_bin_m = xr.Dataset(bin_data_dict, coords={'time': ds_m.time, 'lat': ds_m.lat, 'lon': ds_m.lon})

    ds_bin_m.coords['mask'] = (('lat', 'lon'), ds_mask.land_mask.values)

    for varname, da in ds_bin_m.data_vars.items():
        if land_sea == 'ocean': # or 'ocn' in land_sea:
            ds_bin_m[varname] = da.where(ds_bin_m.mask==0)
        if land_sea == 'land':
            ds_bin_m[varname] = da.where(ds_bin_m.mask==1)
    
    pdf_m, ds_bin_mean_m = bin_3d_data(ds_bin_m, bins, 
            grp_time_var=grp_time_var, bin_var_nm=bin_var_nm)

    # Write data in xarray dataset format
    bins_coord = (bins[0:-1]+bins[1:]) / 2.0

    if grp_time_var is not None:
        ds_grp = ds_m.groupby(grp_time_var).mean('time')
        coords = {grp_time_var: ds_grp[grp_time_var], 'bin':bins_coord}
        dims = (grp_time_var, 'bin')
    else:
        coords = {'time':ds_bin_m.time, 'bin':bins_coord}
        dims = ('time', 'bin')

    return pdf_m, ds_bin_mean_m, dims, coords


def select_3d_obs_data(ds_m, bin_data_dict, ds_mask, bins,
      bin_var_nm='ELF', land_sea='ocean', grp_time_var='year'):
    three_d_varnames = [ 'toa_sw_cre', 'toa_lw_cre', 'toa_net_cre',
                    'tot_cld_amt', 'low_cld_amt', 'mid_cld_amt', 'high_cld_amt']
    for vn in three_d_varnames:
        bin_data_dict[vn] = ds_m[vn]

    ds_bin_m = xr.Dataset(bin_data_dict, coords={'time': ds_m.time, 'lat': ds_m.lat, 'lon': ds_m.lon})

    ds_bin_m.coords['mask'] = (('lat', 'lon'), ds_mask.land_mask.values)

    for varname, da in ds_bin_m.data_vars.items():
        if land_sea == 'ocean': # or 'ocn' in land_sea:
            ds_bin_m[varname] = da.where(ds_bin_m.mask==0)
        if land_sea == 'land':
            ds_bin_m[varname] = da.where(ds_bin_m.mask==1)
    
    pdf_m, ds_bin_mean_m = bin_3d_data(ds_bin_m, bins, 
            grp_time_var=grp_time_var, bin_var_nm=bin_var_nm)

    # Write data in xarray dataset format
    bins_coord = (bins[0:-1]+bins[1:]) / 2.0

    if grp_time_var is not None:
        ds_grp = ds_m.groupby(grp_time_var).mean('time')
        coords = {grp_time_var: ds_grp[grp_time_var], 'bin':bins_coord}
        dims = (grp_time_var, 'bin')
    else:
        coords = {'time':ds_bin_m.time, 'bin':bins_coord}
        dims = ('time', 'bin')

    return pdf_m, ds_bin_mean_m, dims, coords


def bin_obs_data(ds, s_lat=-30, n_lat=30, bin_var_nm='omega500',
        grp_time_var='year', bins=np.arange(0,1.1,0.1), land_sea='global'):

    """ Return binned data for isca dataset based on certain variable
        such as vertical pressure velocity at 500hPa (omega500), ELF,
        EIS and LTS...
    """
    ds_m = ds.where(np.logical_and(ds.lat>=s_lat, ds.lat<=n_lat), drop=True)

    ds_mask = xr.open_dataset('era_land_t42.nc', decode_times=False)
    ds_mask = ds_mask.where(np.logical_and(ds_mask.lat>=s_lat,ds_mask.lat<=n_lat), drop=True)
    #ds_m.coords['mask'] = (('lat', 'lon'), ds_mask.land_mask.values)

    bin_data_dict = {'omega500': ds_m.omega500} 

    vars_dict = {}

    ## 3d variables
    bin_data_dict2 = copy.deepcopy(bin_data_dict)
    pdf_m, ds_bin_mean_m, dims, coords2 = select_3d_obs_data(ds_m, bin_data_dict2, ds_mask,
        bins, bin_var_nm=bin_var_nm, land_sea=land_sea, grp_time_var=grp_time_var)
    for key, val in ds_bin_mean_m.items():
        vars_dict[key] = (dims, val)
    
    vars_dict['pdf_omega'] = (dims, pdf_m)

    ds_bin_mean_m_array = xr.Dataset(vars_dict, coords=coords2)

    return ds_bin_mean_m_array


def bin_isca_exp_data(ds, s_lat=-30, n_lat=30, bin_var_nm='omega500', bin_var=None,
        grp_time_var='year', bins=np.arange(0,1.1,0.1), land_sea='global'):

    """ Return binned data for isca dataset based on certain variable
        such as vertical pressure velocity at 500hPa (omega500), ELF,
        EIS and LTS...
    """
    ds_m = ds.where(np.logical_and(ds.lat>=s_lat, ds.lat<=n_lat), drop=True)

    ds_mask = xr.open_dataset('era_land_t42.nc', decode_times=False)
    ds_mask = ds_mask.where(np.logical_and(ds_mask.lat>=s_lat,ds_mask.lat<=n_lat), drop=True)
    #ds_m.coords['mask'] = (('lat', 'lon'), ds_mask.land_mask.values)

    omega_coeff = 3600. * 24. / 100.
    omega_m = ds_m.omega * omega_coeff

    fint = interpolate.interp1d(np.log(ds_m.pfull), omega_m, kind='linear', axis=1)
    omega500_m = fint(np.log(np.array([500])))
    omega500_m = xr.DataArray(omega500_m[:,0,:,:], coords=[ds_m.time, ds_m.lat, ds_m.lon],
                dims=['time', 'lat', 'lon'])

    if bin_var is None:
        bin_data_dict = {'omega500': omega500_m}
    else:
        omega500_obs_t = np.zeros_like(omega500_m)
        omega500_obs_lat_range = bin_var.where(np.logical_and(bin_var.lat>=s_lat, bin_var.lat<=n_lat), drop=True)
        for t in range(len(ds_m.time)):
            omega500_obs_t[t,:,:] = omega500_obs_lat_range
        omega500_obs_t = xr.DataArray(omega500_obs_t, coords=[ds_m.time, ds_m.lat, ds_m.lon],
                dims=['time', 'lat', 'lon'])
        bin_data_dict = {'omega500': omega500_obs_t} 

    vars_dict = {}

    ## 3d variables
    bin_data_dict2 = copy.deepcopy(bin_data_dict)
    pdf_m, ds_bin_mean_m, dims, coords2 = select_3d_data(ds_m, bin_data_dict2, ds_mask,
        bins, bin_var_nm=bin_var_nm, land_sea=land_sea, grp_time_var=grp_time_var)
    for key, val in ds_bin_mean_m.items():
        vars_dict[key] = (dims, val)
    
    vars_dict['pdf_omega'] = (dims, pdf_m)

    ds_bin_mean_m_array = xr.Dataset(vars_dict, coords=coords2)

    return ds_bin_mean_m_array
