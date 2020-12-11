from __future__ import print_function
import os
import numpy as np
import xarray as xr
import json
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from analysis_functions import get_global_mean


if __name__ == "__main__":
    P = os.path.join
    base_dir = '/scratch/ql260'

    group_models = [('BNU', 'BNU-ESM'),
        ('CCCma', 'CanESM2'), # NO amip
        #('CMCC', 'CMCC-CESM'), # No amip # time no units
        ('CNRM-CERFACS', 'CNRM-CM5'),
        ('CSIRO-BOM', 'ACCESS1-0'),
        ('FIO', 'FIO-ESM'),
        ('INM', 'inmcm4'), 
        ('IPSL', 'IPSL-CM5A-LR'),
        ('IPSL', 'IPSL-CM5A-MR'),
        ('MIROC', 'MIROC-ESM'),
        ('MOHC', 'HadCM3'),
        #('MOHC', 'HadGEM2-ES'), # 200511
        ('MPI-M', 'MPI-ESM-LR'),
        ('MPI-M', 'MPI-ESM-MR'),
        ('MRI', 'MRI-CGCM3'),
        ('MRI', 'MRI-ESM1'),
        ('NASA-GISS', 'GISS-E2-H'),
        #('NASA-GISS', 'GISS-E2-R'), # time no units
        ('NCAR', 'CCSM4'),
        ('NCC', 'NorESM1-M'),
        ('NIMR-KMA', 'HadGEM2-AO'),
        ('NOAA-GFDL', 'GFDL-CM3'),
        ('NOAA-GFDL', 'GFDL-ESM2M'), 
        ('NSF-DOE-NCAR', 'CESM1-CAM5') ]
    
    '''
    group_models = [('BNU', 'BNU-ESM'),
        ('CNRM-CERFACS', 'CNRM-CM5'),
        ('CSIRO-BOM', 'ACCESS1-0'),
        ('INM', 'inmcm4'), 
        ('IPSL', 'IPSL-CM5A-LR'),
        ('IPSL', 'IPSL-CM5A-MR'), 
        ('MPI-M', 'MPI-ESM-LR'), 
        ('MPI-M', 'MPI-ESM-MR'),
        ('MRI', 'MRI-CGCM3'), 
       # ('NASA-GISS', 'GISS-E2-R'),
        ('NCAR', 'CCSM4'), 
        ('NCC', 'NorESM1-M'),
        ('NOAA-GFDL', 'GFDL-CM3'),
        ('NSF-DOE-NCAR', 'CESM1-CAM5'), ]

    '''
    variables = ["rlut", "rsut", "rlutcs", "rsutcs",]

    start_year = 1996
    end_year = 2005

    exp_nm = "historical"  # 'amip'    
    res_label = 'r128x64'

    hist_cre = {}
    hist_zm_cre = {}
    hist_gm_cre = {}

    # Add MME to the dataset as well
    nmodels = len(group_models)
    nlat = 64
    nlon = 128

    sw_cre_sum = np.zeros((nlat, nlon))
    lw_cre_sum = np.zeros((nlat, nlon))
    #net_cre_sum = np.zeros((nlat, nlon))

    for (group, model_nm) in group_models:
        var_dict = {}
        for var in variables:
            dt_dir = P(base_dir, 'cmip5_data', exp_nm, var)
            filename = '_'.join([model_nm, exp_nm, var, str(start_year),
                             str(end_year), res_label + '.nc'])
            ds = xr.open_dataset(P(dt_dir, filename), decode_times=False)
            var_dict[var] = ds[var]
        var_dict['toa_sw_cre'] = (var_dict["rsutcs"] - var_dict["rsut"]).mean('time')

        sw_cre_sum = sw_cre_sum + var_dict['toa_sw_cre']

        if 'ccsm4' in model_nm.lower():
            ## The lats of rlutcs and rlut have small differences.
            lwcre = var_dict["rlutcs"].values - var_dict["rlut"].values
            times = var_dict["rlutcs"].time
            lats = var_dict["rlutcs"].lat
            lons = var_dict["rlutcs"].lon
            var_dict['toa_lw_cre'] = (xr.DataArray(lwcre, coords=[times,lats,lons],
                                     dims=['time','lat','lon'])).mean('time')
        else:
            var_dict['toa_lw_cre'] = (var_dict["rlutcs"] - var_dict["rlut"]).mean('time')
        
        lw_cre_sum = lw_cre_sum + var_dict['toa_lw_cre']
        var_dict['toa_net_cre'] = var_dict['toa_sw_cre'] + var_dict['toa_lw_cre']

        # zonal and global mean cres
        latlon_var_dict = {}
        zm_var_dict = {}
        gm_var_dict = {}
        for k in ['toa_sw_cre', 'toa_lw_cre', 'toa_net_cre']:
            latlon_var_dict[k] = var_dict[k]
            zm_var_dict[k] = var_dict[k].mean('lon')
            gm_var_dict[k] = get_global_mean(var_dict[k])

        hist_cre[model_nm] = latlon_var_dict
        hist_zm_cre[model_nm] = zm_var_dict
        hist_gm_cre[model_nm] = gm_var_dict

    # Add MME to dictionary
    lats = ds.lat
    lons = ds.lon
    sw_cre_sum = xr.DataArray(sw_cre_sum, coords=[lats,lons], dims=['lat','lon'])
    lw_cre_sum = xr.DataArray(lw_cre_sum, coords=[lats,lons], dims=['lat','lon'])
    
    dict_MME = {}
    dict_MME['toa_sw_cre'] = sw_cre_sum / nmodels
    dict_MME['toa_lw_cre'] = lw_cre_sum / nmodels
    dict_MME['toa_net_cre'] =  dict_MME['toa_sw_cre'] +  dict_MME['toa_lw_cre']

    zm_var_dict_MME = {}
    gm_var_dict_MME = {}
    for k in ['toa_sw_cre', 'toa_lw_cre', 'toa_net_cre']:
        zm_var_dict_MME[k] = dict_MME[k].mean('lon')
        gm_var_dict_MME[k] = get_global_mean(dict_MME[k])

    hist_cre['MME'] = dict_MME
    hist_zm_cre['MME'] = zm_var_dict_MME
    hist_gm_cre['MME'] = gm_var_dict_MME

    # ===================================================================== #
    #                              Save dataset
    # ===================================================================== #

    print('Saving data...')
    save_dt_dir = './data'
    if not os.path.exists(save_dt_dir):
        os.makedirs(save_dt_dir)

    cre_fn = P(save_dt_dir, '_'.join(['cmip5', exp_nm, 'cre', res_label + '.npy']))
    zm_cre_fn = P(save_dt_dir, '_'.join(['cmip5', exp_nm, 'zm_cre', res_label + '.npy']))
    gm_cre_fn = P(save_dt_dir, '_'.join(['cmip5', exp_nm, 'gm_cre', res_label + '.npy']))
    
    np.save(cre_fn, hist_cre)
    print(cre_fn + ' saved.')
    np.save(zm_cre_fn, hist_zm_cre)
    print(zm_cre_fn + ' saved.')
    np.save(gm_cre_fn, hist_gm_cre)
    print(gm_cre_fn + ' saved.')
