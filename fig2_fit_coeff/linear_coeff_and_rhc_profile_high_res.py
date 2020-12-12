from __future__ import print_function
from __future__ import division
import os
import sys
sys.path.append('../')
import numpy as np
import xarray as xr
import timeit
import warnings
warnings.filterwarnings("ignore")
from calendar import monthrange
from analysis_functions import add_datetime_info
from fit_coeff_rhc_functions import fit_cld_profile_paras, cld_frac_wrt_rh_linear, cld_frac_spookie


if __name__ == "__main__":
    P = os.path.join
    #dt_dir = '/scratch/ql260/era5/data_2017_high_res' # On gv3
    dt_dir = '/disca/share/ql260/era5/data_2017_high_res'

    dst_var = sys.argv[1] # should b 'ceoff_a' or 'rhc'
    print(dst_var)

    if dst_var.lower()!='coeff_a' and dst_var.lower()!='rhc':
        raise ValueError("The first argument should be 'coeff_a' or 'rhc'.")

    saved_dt_dir = P('./data', dst_var + '_data', 'high_res_0.75')
    if not os.path.exists(saved_dt_dir):
        os.makedirs(saved_dt_dir)

    resolution_name = ['0.75_deg']
    resolution_str = ['r480x240']

    for mon in range(0, 12):
        mon_str = str(mon+1).zfill(2)
        print(mon_str)
        ndays = monthrange(2017, mon+1)[1]

        for res_nm, res_str in zip(resolution_name, resolution_str):
            start = timeit.default_timer()
            # print('resolution is '+ str(res_nm))
            fn_arr = [P(dt_dir, mon_str+'_'+res_str, 'era5_cld_rh_2017_'+mon_str+'_'+str(day+1).zfill(2)+'_'+res_str+'.nc') for day in range(0,ndays)]
            ds = xr.open_mfdataset(fn_arr, decode_times=False)
            try:
                ds = ds.rename({'latitude':'lat', 'longitude':'lon'})
            except:
                print('Already lat and lon, do nothing...')

            rh = ds.r
            add_datetime_info(rh)

            cld_frac = ds.cc
            add_datetime_info(cld_frac)
    
            # lat_arr = [(-90, 90), (30, 60), (-60, -30), (60, 90), (-90, -60), (-30, 30)]
            # lat_str_arr = ['90S_90N', '30N_60N', '30S_60S', '60N_90N', '60S_90S', '30S_30N']
            lat_arr = [(-90, 90)]
            lat_str_arr = ['90S_90N']

            for (slat, nlat), lat_str in zip(lat_arr, lat_str_arr):
                # print('lat range is:', slat, nlat)
                levels = ds.level
                times = ds.time
                lats = ds.lat

                l_lat = np.logical_and(lats>=slat, lats<=nlat)
                
                fitted_var = np.zeros((len(times), len(levels)))

                start1 = timeit.default_timer()
                for i in range(len(times)):
                    print('time='+str(i))
                    rh_t = rh[i,:,:,:].where(l_lat, drop=True)
                    cld_frac_t = cld_frac[i,:,:,:].where(l_lat, drop=True)

                    if dst_var.lower()=='coeff_a':
                        fitted_var_i = fit_cld_profile_paras(cld_frac_wrt_rh_linear,  rh_t, cld_frac_t, fit_rhc=False)
                    if dst_var.lower()=='rhc':
                        fitted_var_i = fit_cld_profile_paras(cld_frac_spookie,  rh_t, cld_frac_t, fit_rhc=True)
    
                    fitted_var[i,:] = np.squeeze(fitted_var_i)

                    if np.mod(i, 10) == 0:
                        stop = timeit.default_timer() 
                        print('Time has passed: ', stop - start1)
                    
                fitted_var = xr.DataArray(fitted_var, coords=[times, levels], dims=['time', 'level'], name='fitted_var')
                
                if dst_var.lower()=='coeff_a':
                    fitted_var_nm = 'para_a'
                if dst_var.lower()=='rhc':
                    fitted_var_nm = 'rhc'
                ds_var = xr.Dataset({fitted_var_nm:fitted_var}, coords={'time':times, 'level':levels})

                ds_a.to_netcdf(P(saved_dt_dir, '_'.join(['fit', dst_var, '2017' + mon_str, res_nm, lat_str + '.nc'])),
                                format='NETCDF3_CLASSIC', mode='w')
                print(res_nm+' file is saved')
                
                stop = timeit.default_timer() 
                print('Time has passed: ', stop - start)
