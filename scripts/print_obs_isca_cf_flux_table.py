from __future__ import print_function
import numpy as np
import pandas as pd
from isca_cre_cwp import get_gm

def print_obs_isca_cf_flux_table(ds_arr, ds_names, cf_obs_dict, flux_obs_dict, file_name=None, float_fmt='%.2f'):
    row_names = [r'Low cloud amount (%)', r'Middle cloud amount (%)',
                r'High cloud amount (%)', r'Total cloud amount (%)',
                r'TOA net SW flux', r'TOA net LW flux',
                r'TOA net flux',
                r'TOA SW CRE', r'TOA LW CRE',
                r'TOA net CRE', r'Cloud water path']

    N = len(ds_arr)+1
    col_names = ['Obs']
    for ds_nm in ds_names:
        col_names.append(ds_nm)
    
    table =  np.zeros((len(row_names), N), dtype='double')

    # =============== Obs results ================ #
    var_names = ['low_cld_amt',  'mid_cld_amt',
                'high_cld_amt', 'tot_cld_amt',
                'toa_net_sw', 'olr', 'toa_net_flux',
                'toa_sw_cre', 'toa_lw_cre', 
                'toa_net_cre', 'cwp', ]
   
    for i, key in enumerate(var_names[0:4]):
        val = ma_gm(cf_obs_dict[key]) #get_gm(cf_obs_dict[key])
        table[i, 0] = val

    for i, key in enumerate(var_names[4:]):
        if 'cwp' not in key:
            val = get_gm(flux_obs_dict[key])
        else:
            val = ma_gm(flux_obs_dict[key])
        if 'cwp' in key:
            table[i+4, 0] = val*1e3
        else:
            table[i+4, 0] = val

    # =============== Model results ================ #
    def get_model_flux_arr(ds):
        # ---------- TOA -------------- #
        toa_net_sw = get_gm(ds.soc_toa_sw)
        olr = get_gm(ds.soc_olr)
        toa_net_flux = toa_net_sw - olr

        toa_sw_cre = get_gm(ds.toa_sw_cre)
        toa_lw_cre = get_gm(ds.toa_lw_cre)
        toa_net_cre = get_gm(ds.toa_net_cre)

        cwp = get_gm(ds.cwp)*1e3

        mod_arr = [toa_net_sw, olr, toa_net_flux,
                    toa_sw_cre, toa_lw_cre, toa_net_cre, cwp,]
        return mod_arr

    def get_model_cf_arr(ds):
        try:
            low_ca = get_gm(ds.low_cld_amt_mxr)
            mid_ca = get_gm(ds.mid_cld_amt_mxr)
            high_ca = get_gm(ds.high_cld_amt_mxr)
            tot_ca = get_gm(ds.tot_cld_amt_mxr)
        except:
            low_ca = get_gm(ds.low_cld_amt)
            mid_ca = get_gm(ds.mid_cld_amt)
            high_ca = get_gm(ds.high_cld_amt)
            tot_ca = get_gm(ds.tot_cld_amt)
        mod_arr = [low_ca, mid_ca, high_ca, tot_ca]
        return mod_arr

    for nn, ds in enumerate(ds_arr):
        mod_cf_arr = get_model_cf_arr(ds)
        mod_flux_arr = get_model_flux_arr(ds)
        mod_arr = mod_cf_arr + mod_flux_arr
        #mod_dt_arrs.append(mod_arr)
        for i, dt in enumerate(mod_arr):
            table[i, nn+1] = dt

    tbl = pd.DataFrame(data=table, index=row_names, columns=col_names)
    if file_name is None:
        print(tbl.to_latex(float_format=float_fmt))
    else:
        tbl.to_latex(buf=file_name, float_format=float_fmt)

    return tbl


def ma_gm(dt):
    dt_ma = np.ma.array(dt, mask=np.isnan(dt))
    lats = dt.lat
    coslat = np.cos(np.deg2rad(lats))
    dims = dt.dims
    lon_dim = dims.index('lon')
    lat_dim = dims.index('lat')
    if 'time' not in dims:
        dt_gm = np.ma.average(np.ma.average(dt_ma, axis=lon_dim), axis=lat_dim, weights=coslat)
    else:
        time_dim = dims.index('time')
        dt_gm = np.ma.average(np.ma.average(dt_ma, axis=(time_dim,lon_dim)), axis=lat_dim-1, weights=coslat)
    return dt_gm
