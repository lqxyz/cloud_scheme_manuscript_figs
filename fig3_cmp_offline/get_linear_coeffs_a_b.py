from __future__ import print_function
from __future__ import division
import os
import sys
sys.path.append('../scripts')
import numpy as np
import xarray as xr
import timeit
import warnings
warnings.filterwarnings("ignore")
from analysis_functions import add_datetime_info
from fit_coeff_rhc_functions import fit_cld_profile_paras


def cld_frac_wrt_rh_linear_a_b(rh, a, b):
    cf = a * rh + b
    cf[cf>1.0] = 1.0
    cf[cf<0.0] = 0.0
    return cf


if __name__ == "__main__":
    P = os.path.join
    #dt_dir = '/scratch/ql260/era5/data_2017' # On gv3
    dt_dir = '/disca/share/ql260/era5/data_2017'

    saved_dt_dir = './data'
    if not os.path.exists(saved_dt_dir):
        os.mkdir(saved_dt_dir)
   
    resolution_name = ['T42',] 
    resolution_str = ['r128x64',]

    mon_str = '01'
    for res_nm, res_str in zip(resolution_name, resolution_str):
        start = timeit.default_timer()
        print('resolution is '+ str(res_nm))
        if '1_deg' is res_nm:
            fn = 'era5_cld_rh_2017_01.nc'
        else:
            fn = 'era5_cld_rh_2017_01_'+res_str+'.nc'
        
        ds = xr.open_dataset(P(dt_dir, fn), decode_times=False)
        try:
            ds = ds.rename({'latitude':'lat', 'longitude':'lon'})
        except:
            print('Already lat and lon, do nothing...')

        rh = ds.r
        add_datetime_info(rh)

        cld_frac = ds.cc
        add_datetime_info(cld_frac)

        #lat_arr = [(-90, 90), (30, 60), (-60, -30), (60, 90), (-90, -60), (-30, 30)]
        #lat_str_arr = ['90S_90N', '30N_60N', '30S_60S', '60N_90N', '60S_90S', '30S_30N']
        lat_arr = [(-90, 90)]
        lat_str_arr = ['90S_90N']

        for (slat, nlat), lat_str in zip(lat_arr, lat_str_arr):
            print('lat range is:', slat, nlat)

            levels = ds.level
            times = ds.time
            lats = ds.lat

            l_lat = np.logical_and(lats>=slat, lats<=nlat)
            para = np.zeros((len(times), len(levels), 2))
            
            start1 = timeit.default_timer()
            for i in range(len(times)):
                print('time='+str(i))
                rh_t = rh[i,:,:,:].where(l_lat, drop=True)
                cld_frac_t = cld_frac[i,:,:,:].where(l_lat, drop=True)

                para_ab_i = fit_cld_profile_paras(cld_frac_wrt_rh_linear_a_b, rh_t, cld_frac_t, 
                            fit_rhc=False, left_boundary=[0, -np.inf], right_boundary=[np.inf, 0])
                          
                para[i,:,:] = np.squeeze(para_ab_i)
                if np.mod(i, 10) == 0:
                    stop = timeit.default_timer() 
                    print('Time has passed: ', stop - start1)
                
            para = xr.DataArray(para, coords=[times, levels, np.array([0,1])], 
                                dims=['time', 'level', 'coeffs'], name='para_ab')
            ds_a = xr.Dataset({'para_ab':para}, coords={'time':times, 'level':levels, 
                                'coeffs':np.array([0,1])})
 
            ds_a.to_netcdf(P(saved_dt_dir, 'fit_linear_para_ab_2017' + mon_str +
                             '_' + res_nm + '_' + lat_str+'.nc'), 
                            format='NETCDF3_CLASSIC', mode='w')
            print(res_nm+' file is saved')
            
            stop = timeit.default_timer() 
            print('Time has passed: ', stop - start)
