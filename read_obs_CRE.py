from __future__ import print_function
import numpy as np
import xarray as xr
import os
from scipy.integrate import trapz
from analysis_functions import add_datetime_info


def rename_dims_to_standards(ds, new_dims=('time','lat','lon')):
    dim_dict = {}
    for old_dim, new_dim in zip(ds.dims, new_dims):
        dim_dict[old_dim] = new_dim
    return ds.rename(dim_dict)

    
def read_toa_cre_obs(dataset_name='ALL', base_dir='/scratch/ql260/obs_datasets/'):
    def calc_and_add_cre_to_dict(toa_cre, ds_name, lw_clr, lw_cld, 
        sw_clr, sw_cld, cwp, lwp=None, iwp=None):
        dt = {}
        dt['toa_lw_cre'] = rename_dims_to_standards(lw_clr - lw_cld)
        dt['toa_sw_cre'] = rename_dims_to_standards(sw_clr - sw_cld)
        dt['toa_net_cre'] = dt['toa_lw_cre'] + dt['toa_sw_cre']
        dt['cwp'] = rename_dims_to_standards(cwp)
        if lwp is not None:
            dt['lwp'] = rename_dims_to_standards(lwp)
        if iwp is not None:
            dt['iwp'] = rename_dims_to_standards(iwp)
        toa_cre[ds_name] = dt

    # ================== Read CERES ================== #
    def read_ceres_ebaf_ed4p1():
        ds_name = 'CERES_EBAF Ed4.1'
        obs_dir = 'CERES'
        file_nm = 'CERES_EBAF-TOA_Ed4.1_Subset_200101-201812.nc'
        ds = xr.open_dataset(os.path.join(base_dir, obs_dir, file_nm), decode_times=False)
        add_datetime_info(ds)
        # clear sky upward lw
        lw_clr = ds.toa_lw_clr_c_mon.groupby('month').mean('time')
        # cloudy sky upward lw
        lw_cld = ds.toa_lw_all_mon.groupby('month').mean('time')
        # clear sky upward sw
        sw_clr = ds.toa_sw_clr_c_mon.groupby('month').mean('time')
        # cloudy sky upward sw
        sw_cld = ds.toa_sw_all_mon.groupby('month').mean('time')
        # cloud water
        file_nm = 'CERES_SYN1deg-Month_Terra-Aqua-MODIS_Ed4.1_Subset_200101-201812_lwp_iwp.nc'
        ds = xr.open_dataset(os.path.join(base_dir, obs_dir, file_nm), decode_times=False)
        add_datetime_info(ds)
        lwp = ds.lwp_total_mon.groupby('month').mean('time')
        iwp = ds.iwp_total_mon.groupby('month').mean('time')
        cwp = (lwp+iwp) / 1.0e3 # kg/m^-2
        lwp = lwp / 1.0e3
        iwp = iwp / 1.0e3
        calc_and_add_cre_to_dict(toa_cre, ds_name, lw_clr, lw_cld, sw_clr, sw_cld, cwp, lwp=lwp, iwp=iwp)

    def read_ceres_ebaf_ed4():
        ds_name = 'CERES_EBAF Ed4.0'
        obs_dir = 'CERES'
        file_nm = 'CERES_EBAF-TOA_Ed4.0_Subset_201701-201712.nc'
        dt = {}
        ds = xr.open_dataset(os.path.join(base_dir, obs_dir, file_nm), decode_times=False)
        dt['toa_lw_cre'] = ds.toa_cre_lw_mon
        dt['toa_sw_cre'] = ds.toa_cre_sw_mon
        dt['toa_net_cre'] = dt['toa_lw_cre'] + dt['toa_sw_cre']
        file_nm = 'CERES_SYN1deg-Month_Terra-Aqua-MODIS_Ed4A_Subset_201701-201712.nc'
        ds = xr.open_dataset(os.path.join(base_dir, obs_dir, file_nm), decode_times=False)
        lwp = ds.lwp_total_mon
        iwp = ds.iwp_total_mon
        cwp = (lwp+iwp) / 1.0e3 # kg/m^-2
        dt['cwp'] = cwp
        dt['lwp'] = lwp / 1.0e3
        dt['iwp'] = iwp / 1.0e3
        toa_cre[ds_name] = dt

    def read_ceres_ebaf_ed28():
        ds_name = 'CERES_EBAF Ed2.8'
        obs_dir = 'CERES'
        file_nm = 'CERES_EBAF-TOA_Ed2.8_Subset_201601-201612.nc'
        dt = {}
        ds = xr.open_dataset(os.path.join(base_dir, obs_dir, file_nm), decode_times=False)
        dt['toa_lw_cre'] = ds.toa_cre_lw_mon
        dt['toa_sw_cre'] = ds.toa_cre_sw_mon
        dt['toa_net_cre'] = dt['toa_lw_cre'] + dt['toa_sw_cre']
        file_nm = 'CERES_SYN1deg-Month_Terra-Aqua-MODIS_Ed3A_Subset_201601-201612.nc'
        ds = xr.open_dataset(os.path.join(base_dir, obs_dir, file_nm), decode_times=False)
        lwp = ds.lwp_total_mon
        iwp = ds.iwp_total_mon
        cwp = (lwp+iwp) / 1.0e3 # kg/m^-2
        dt['cwp'] = cwp
        dt['lwp'] = lwp / 1.0e3
        dt['iwp'] = iwp / 1.0e3
        toa_cre[ds_name] = dt

    # ================== Read NCEP ================== #
    def read_ncep():
        obs_dir = 'NCEP'
        ds_name = 'NCEP'
        # clear sky upward lw
        file_nm = 'csulf.ntat.gauss.2017.nc'
        ds = xr.open_dataset(os.path.join(base_dir, obs_dir, file_nm), decode_times=False)
        add_datetime_info(ds)
        lw_clr = ds.csulf.groupby('month').mean('time')
        # cloudy sky upward lw
        file_nm = 'ulwrf.ntat.gauss.2017.nc'
        ds = xr.open_dataset(os.path.join(base_dir, obs_dir, file_nm), decode_times=False)
        add_datetime_info(ds)
        lw_cld = ds.ulwrf.groupby('month').mean('time')
        # clear sky upward sw
        file_nm = 'csusf.ntat.gauss.2017.nc'
        ds = xr.open_dataset(os.path.join(base_dir, obs_dir, file_nm), decode_times=False)
        add_datetime_info(ds)
        sw_clr = ds.csusf.groupby('month').mean('time')
        # cloudy sky upward sw
        file_nm = 'uswrf.ntat.gauss.2017.nc'
        ds = xr.open_dataset(os.path.join(base_dir, obs_dir, file_nm), decode_times=False)
        add_datetime_info(ds)
        sw_cld = ds.uswrf.groupby('month').mean('time')
        # cloud water
        file_nm = 'cldwtr.eatm.2010.nc'
        ds = xr.open_dataset(os.path.join(base_dir, obs_dir, file_nm), decode_times=False)
        add_datetime_info(ds)
        cwp = ds.cldwtr.groupby('month').mean('time')
        calc_and_add_cre_to_dict(toa_cre, ds_name, lw_clr, lw_cld, sw_clr, sw_cld, cwp)

    # ================== Read CFSR ================== #
    def read_ncep_cfsr():
        obs_dir = 'NCEP/CFSR'
        ds_name = 'CFSR'
        file_nm = 'pgbl01.gdas.2010.grb2.nc'
        ds = xr.open_dataset(os.path.join(base_dir, obs_dir, file_nm), decode_times=False)
        # clear sky upward lw
        lw_clr = ds.CSULF_L8_FcstAvg
        # cloudy sky upward lw
        lw_cld = ds.ULWRF_L8_FcstAvg
        # clear sky upward sw
        sw_clr = ds.CSUSF_L8_FcstAvg # .mean('time1')
        # cloudy sky upward sw
        sw_cld = ds.USWRF_L8_FcstAvg #.mean('time1')
        # cloud water
        cwp = ds.C_WAT_L200_Avg #.mean('time0')
        calc_and_add_cre_to_dict(toa_cre, ds_name, lw_clr, lw_cld, sw_clr, sw_cld, cwp)

    # ================== Read JRA55 ================== #
    def read_jra55():
        obs_dir = 'JRA55'
        ds_name = 'JRA55'
        # clear sky upward lw
        file_nm = 'fcst_phy2m.162_csulf.reg_tl319.201001_201012.liu357967.nc'
        ds = xr.open_dataset(os.path.join(base_dir, obs_dir, file_nm), decode_times=False)
        lw_clr = ds.CSULF_GDS4_NTAT_S130 #.mean('initial_time0_hours')
        # cloudy sky upward lw
        file_nm = 'fcst_phy2m.212_ulwrf.reg_tl319.201001_201012.liu357967.nc'
        ds = xr.open_dataset(os.path.join(base_dir, obs_dir, file_nm), decode_times=False)
        lw_cld = ds.ULWRF_GDS4_NTAT_S130 #.mean('initial_time0_hours')
        # clear sky upward sw
        file_nm = 'fcst_phy2m.160_csusf.reg_tl319.201001_201012.liu357967.nc'
        ds = xr.open_dataset(os.path.join(base_dir, obs_dir, file_nm), decode_times=False)
        sw_clr = ds.CSUSF_GDS4_NTAT_S130 #.mean('initial_time0_hours')
        # cloudy sky upward sw
        file_nm = 'fcst_phy2m.211_uswrf.reg_tl319.201001_201012.liu357967.nc'
        ds = xr.open_dataset(os.path.join(base_dir, obs_dir, file_nm), decode_times=False)
        sw_cld = ds.USWRF_GDS4_NTAT_S130 #.mean('initial_time0_hours')
        # cloud water
        file_nm = 'fcst_p125.221_cwat.201001_201012.liu357967.nc'
        ds = xr.open_dataset(os.path.join(base_dir, obs_dir, file_nm), decode_times=False)
        grav = 9.87
        q = ds.CWAT_GDS0_ISBL_S113
        dims = list(q.dims)
        shp = list(q.shape)
        shp.pop(1)
        dims.pop(1)
        dim_info = []
        for dim in dims:
            dim_info.append((dim, ds[dim]))
        cwp =  xr.DataArray(np.zeros(shp), dim_info)
        cwp[:] = trapz(q, x=ds.lv_ISBL1, axis=1) * 1e2 / grav # kg/m^2
        #cwp = cwp.mean(dims[0])
        
        calc_and_add_cre_to_dict(toa_cre, ds_name, lw_clr, lw_cld, sw_clr, sw_cld, cwp)

    # ================== Read ERA-Interim ================== #
    def read_era_interim():
        obs_dir = 'ecmwf_data'
        ds_name = 'ERA-Interim'
        file_nm = 'toa_sfc_cld_clear_flux_monthly_2017.nc'
        ds = xr.open_dataset(os.path.join(base_dir, obs_dir, file_nm), decode_times=False)
        add_datetime_info(ds)
        # Change J/m^2 to W/m^2
        time_step = 3600.0 * (ds.time[1]-ds.time[0])
        # clear sky net lw at toa
        lw_clr = ds.ttrc.groupby('month').mean('time') / time_step # .mean('time')
        # cloudy sky net lw at toa
        lw_cld = ds.ttr.groupby('month').mean('time') / time_step
        # clear sky net sw at toa
        sw_clr = ds.tsrc.groupby('month').mean('time') / time_step
        # cloudy sky net sw at toa
        sw_cld = ds.tsr.groupby('month').mean('time') / time_step
        # total cloud water path
        lwp = ds.tclw.groupby('month').mean('time')
        iwp = ds.tciw.groupby('month').mean('time')
        cwp = lwp + iwp
        # !!!! Note: Change the parameter order (cld and clr) to change the sign of CRE
        calc_and_add_cre_to_dict(toa_cre, ds_name, lw_cld, lw_clr, sw_cld, sw_clr, cwp)

    # Control flow
    toa_cre = {}

    if dataset_name.upper()=='ALL':
        read_ceres_ebaf_ed4p1()
        read_ceres_ebaf_ed4()
        read_ceres_ebaf_ed28()
        read_ncep()
        read_ncep_cfsr()
        read_jra55()
        read_era_interim()
    elif dataset_name.upper()=='CERES_EBAF ED4.1':
        read_ceres_ebaf_ed4p1()
    elif dataset_name.upper()=='CERES_EBAF ED4.0':
        read_ceres_ebaf_ed4()
    elif dataset_name.upper()=='CERES_EBAF ED2.8':
        read_ceres_ebaf_ed28()
    elif dataset_name.upper()=='NCEP':
        read_ncep()
    elif dataset_name.upper()=='NCEP/CFSR':
        read_ncep_cfsr()
    elif dataset_name.upper()=='JRA55':
        read_jra55()
    elif dataset_name.upper()=='ERA-INTERIM':
        read_era_interim()
    else:
        print('Available datasets are: CERES_EBAF Ed4.0/4.1, CERES_EBAF Ed2.8, NCEP, NCEP/CFSR, JRA55, ERA-Interim')
 
    return toa_cre   


def read_toa_flux_obs(dataset_name='ALL', base_dir='/scratch/ql260/obs_datasets/'):
    def calc_and_add_flux_to_dict(toa_flux, ds_name, olr, swup, swdn):
        dt = {}     
        dt['olr'] = rename_dims_to_standards(olr)
        dt['toa_sw_up'] = rename_dims_to_standards(swup)
        dt['toa_net_sw'] = rename_dims_to_standards(swdn) - dt['toa_sw_up']
        dt['toa_net_flux'] = dt['toa_net_sw'] - dt['olr']
        toa_flux[ds_name] = dt

    # ================== Read CERES ================== #
    def read_ceres_ebaf_ed4p1():
        ds_name = 'CERES_EBAF Ed4.1'
        obs_dir = 'CERES'
        file_nm = 'CERES_EBAF-TOA_Ed4.1_Subset_200101-201812.nc'
        dt = {}
        ds = xr.open_dataset(os.path.join(base_dir, obs_dir, file_nm), decode_times=False)
        add_datetime_info(ds)
        dt['olr'] = rename_dims_to_standards(ds.toa_lw_all_mon.groupby('month').mean('time')) # outgoing lw flux
        dt['toa_sw_up'] = rename_dims_to_standards(ds.toa_sw_all_mon.groupby('month').mean('time'))
        dt['toa_net_flux'] = rename_dims_to_standards(ds.toa_net_all_mon.groupby('month').mean('time'))
        dt['toa_net_sw'] = dt['toa_net_flux'] + dt['olr']
        toa_flux[ds_name] = dt

    def read_ceres_ebaf_ed4():
        ds_name = 'CERES_EBAF Ed4.0'
        obs_dir = 'CERES'
        file_nm = 'CERES_EBAF-TOA_Ed4.0_Subset_201701-201712.nc'
        dt = {}
        ds = xr.open_dataset(os.path.join(base_dir, obs_dir, file_nm), decode_times=False)
        dt['olr'] = ds.toa_lw_all_mon # outgoing lw flux
        dt['toa_sw_up'] = ds.toa_sw_all_mon
        dt['toa_net_flux'] = ds.toa_net_all_mon
        dt['toa_net_sw'] = ds.toa_net_all_mon + ds.toa_lw_all_mon
        toa_flux[ds_name] = dt

    def read_ceres_ebaf_ed28():
        ds_name = 'CERES_EBAF Ed2.8'
        obs_dir = 'CERES'
        file_nm = 'CERES_EBAF-TOA_Ed2.8_Subset_201601-201612.nc'
        dt = {}
        ds = xr.open_dataset(os.path.join(base_dir, obs_dir, file_nm), decode_times=False)
        add_datetime_info(ds)
        dt['olr'] = ds.toa_lw_all_mon # outgoing lw flux
        dt['toa_sw_up'] = ds.toa_sw_all_mon
        dt['toa_net_flux'] = ds.toa_net_all_mon
        dt['toa_net_sw'] = dt['toa_net_flux'] + dt['olr']

        toa_flux[ds_name] = dt

    # ================== Read NCEP ================== #
    def read_ncep():
        obs_dir = 'NCEP'
        ds_name = 'NCEP'

        # cloudy sky upward lw
        file_nm = 'ulwrf.ntat.gauss.2017.nc'
        ds = xr.open_dataset(os.path.join(base_dir, obs_dir, file_nm), decode_times=False)
        add_datetime_info(ds)
        olr = ds.ulwrf.groupby('month').mean('time')
        # cloudy sky upward sw
        file_nm = 'uswrf.ntat.gauss.2017.nc'
        ds = xr.open_dataset(os.path.join(base_dir, obs_dir, file_nm), decode_times=False)
        add_datetime_info(ds)
        swup = ds.uswrf.groupby('month').mean('time')
        # downward sw flux
        file_nm = 'dswrf.ntat.gauss.2017.nc'
        ds = xr.open_dataset(os.path.join(base_dir, obs_dir, file_nm), decode_times=False)
        add_datetime_info(ds)
        swdn = ds.dswrf.groupby('month').mean('time')

        calc_and_add_flux_to_dict(toa_flux, ds_name, olr, swup, swdn)


    # ================== Read CFSR ================== #
    def read_ncep_cfsr():
        obs_dir = 'NCEP/CFSR'
        ds_name = 'CFSR'
        file_nm = 'pgbl01.gdas.2010.grb2.nc'  # 01, 02 in filename means how many hours to average
        ds = xr.open_dataset(os.path.join(base_dir, obs_dir, file_nm), decode_times=False)
       
        # cloudy sky upward lw
        olr = ds.ULWRF_L8_FcstAvg
        # cloudy sky upward sw
        swup = ds.USWRF_L8_FcstAvg
        # downward shortwave radiation flux at toa
        swdn = ds.DSWRF_L8_FcstAvg

        calc_and_add_flux_to_dict(toa_flux, ds_name, olr, swup, swdn)

    # ================== Read JRA55 ================== #
    def read_jra55():
        obs_dir = 'JRA55'
        ds_name = 'JRA55'

        # cloudy sky upward lw
        file_nm = 'fcst_phy2m.212_ulwrf.reg_tl319.201001_201012.liu357967.nc'
        ds = xr.open_dataset(os.path.join(base_dir, obs_dir, file_nm), decode_times=False)
        olr = ds.ULWRF_GDS4_NTAT_S130 #.mean('initial_time0_hours')
        # cloudy sky upward sw
        file_nm = 'fcst_phy2m.211_uswrf.reg_tl319.201001_201012.liu357967.nc'
        ds = xr.open_dataset(os.path.join(base_dir, obs_dir, file_nm), decode_times=False)
        swup = ds.USWRF_GDS4_NTAT_S130
        # downward sw
        file_nm = 'fcst_phy2m.204_dswrf.reg_tl319.201001_201012.liu357967.nc'
        ds = xr.open_dataset(os.path.join(base_dir, obs_dir, file_nm), decode_times=False)
        swdn = ds.DSWRF_GDS4_NTAT_S130

        calc_and_add_flux_to_dict(toa_flux, ds_name, olr, swup, swdn)

    # ================== Read ERA-Interim ================== #
    def read_era_interim():
        obs_dir = 'ecmwf_data'
        ds_name = 'ERA-Interim'
        file_nm = 'toa_sfc_cld_clear_flux_monthly_2017.nc'
        ds = xr.open_dataset(os.path.join(base_dir, obs_dir, file_nm), decode_times=False)
        add_datetime_info(ds)
        # Change J/m^2 to W/m^2
        time_step = 3600.0 * (ds.time[1]-ds.time[0])

        # cloudy sky net lw at toa
        olr = -ds.ttr.groupby('month').mean('time') / time_step # downward is positive
        # cloudy sky net sw at toa
        net_sw = ds.tsr.groupby('month').mean('time') / time_step
        # downward sw flux
        file_nm = 'toa_incident_solar_2017_mon_step12.nc'
        ds = xr.open_dataset(os.path.join(base_dir, obs_dir, file_nm), decode_times=False)
        add_datetime_info(ds)
        time_step = 3600.0 * (ds.time[1]-ds.time[0])
        swdn = ds.tisr.groupby('month').mean('time') / time_step
        swup = swdn - net_sw

        calc_and_add_flux_to_dict(toa_flux, ds_name, olr, swup, swdn)

    # Control flow
    toa_flux = {}

    if dataset_name.upper()=='ALL':
        read_ceres_ebaf_ed4p1()
        read_ceres_ebaf_ed4()
        read_ceres_ebaf_ed28()
        read_ncep()
        read_ncep_cfsr()
        read_jra55()
        read_era_interim()
    elif dataset_name.upper()=='CERES_EBAF ED4.1':
        read_ceres_ebaf_ed4p1()
    elif dataset_name.upper()=='CERES_EBAF ED4.0':
        read_ceres_ebaf_ed4()
    elif dataset_name.upper()=='CERES_EBAF ED2.8':
        read_ceres_ebaf_ed28()
    elif dataset_name.upper()=='NCEP':
        read_ncep()
    elif dataset_name.upper()=='NCEP/CFSR':
        read_ncep_cfsr()
    elif dataset_name.upper()=='JRA55':
        read_jra55()
    elif dataset_name.upper()=='ERA-INTERIM':
        read_era_interim()
    else:
        print('Available datasets are: CERES_EBAF Ed4.0, CERES_EBAF Ed2.8, NCEP, NCEP/CFSR, JRA55, ERA-Interim')
 
    return toa_flux   


def read_surf_flux_obs(dataset_name='ALL', base_dir='/scratch/ql260/obs_datasets/'):
    """[ ###'toa_sw_up', 'toa_net_sw', 'olr', 'toa_net_flux',###
            'surf_sw_down', 'surf_sw_up', 'surf_net_sw',
            'surf_lw_down', 'surf_lw_up', 'surf_net_lw',
            'surf_net_flux' ]
    """
    def calc_and_add_flux_to_dict(surf_flux, ds_name, surf_sw_up, surf_sw_down, surf_lw_up, surf_lw_down):
        dt = {}     
        dt['surf_sw_up'] = rename_dims_to_standards(surf_sw_up)
        dt['surf_sw_down'] = rename_dims_to_standards(surf_sw_down)
        dt['surf_net_sw'] =  dt['surf_sw_down'] - dt['surf_sw_up']
        
        dt['surf_lw_up'] = rename_dims_to_standards(surf_lw_up)
        dt['surf_lw_down'] = rename_dims_to_standards(surf_lw_down)
        dt['surf_net_lw'] =  dt['surf_lw_down'] - dt['surf_lw_up']
    
        dt['surf_net_flux'] = dt['surf_net_sw'] + dt['surf_net_lw']
        
        surf_flux[ds_name] = dt

    # ================== Read CERES ================== #
    def read_ceres_ebaf_ed4p1():
        ds_name = 'CERES_EBAF Ed4.1'
        obs_dir = 'CERES'
        #file_nm = 'CERES_EBAF-Surface_Ed4.1_Subset_200101-201812.nc'
        file_nm = 'CERES_EBAF_Ed4.1_Subset_200101-201812_surf.nc'
        dt = {}
        ds = xr.open_dataset(os.path.join(base_dir, obs_dir, file_nm), decode_times=False)
        add_datetime_info(ds)
        dt['surf_sw_down'] = rename_dims_to_standards(ds.sfc_sw_down_all_mon.groupby('month').mean('time'))
        dt['surf_sw_up'] = rename_dims_to_standards(ds.sfc_sw_up_all_mon.groupby('month').mean('time'))
        dt['surf_net_sw'] = rename_dims_to_standards(ds.sfc_net_sw_all_mon.groupby('month').mean('time'))

        dt['surf_lw_down'] = rename_dims_to_standards(ds.sfc_lw_down_all_mon .groupby('month').mean('time'))
        dt['surf_lw_up'] = rename_dims_to_standards(ds.sfc_lw_up_all_mon.groupby('month').mean('time'))
        dt['surf_net_lw'] = rename_dims_to_standards(ds.sfc_net_lw_all_mon.groupby('month').mean('time'))

        dt['surf_net_flux'] =   dt['surf_net_sw'] +  dt['surf_net_lw']

        surf_flux[ds_name] = dt

    def read_ceres_ebaf_ed4():
        ds_name = 'CERES_EBAF Ed4.0'
        obs_dir = 'CERES'
        file_nm = 'CERES_EBAF-Surface_Ed4.0_Subset_201701-201712.nc'
        dt = {}
        ds = xr.open_dataset(os.path.join(base_dir, obs_dir, file_nm), decode_times=False)
        dt['surf_sw_down'] = ds.sfc_sw_down_all_mon
        dt['surf_sw_up'] = ds.sfc_sw_up_all_mon
        dt['surf_net_sw'] = ds.sfc_net_sw_all_mon

        dt['surf_lw_down'] = ds.sfc_lw_down_all_mon 
        dt['surf_lw_up'] = ds.sfc_lw_up_all_mon
        dt['surf_net_lw'] = ds.sfc_net_lw_all_mon

        dt['surf_net_flux'] =  ds.sfc_net_sw_all_mon + ds.sfc_net_lw_all_mon

        surf_flux[ds_name] = dt

    def read_ceres_ebaf_ed28():
        ds_name = 'CERES_EBAF Ed2.8'
        obs_dir = 'CERES'
        file_nm = 'CERES_EBAF-Surface_Ed2.8_Subset_201601-201612.nc'
        dt = {}
        ds = xr.open_dataset(os.path.join(base_dir, obs_dir, file_nm), decode_times=False)
        dt['surf_sw_down'] = ds.sfc_sw_down_all_mon
        dt['surf_sw_up'] = ds.sfc_sw_up_all_mon
        dt['surf_net_sw'] = ds.sfc_net_sw_all_mon

        dt['surf_lw_down'] = ds.sfc_lw_down_all_mon 
        dt['surf_lw_up'] = ds.sfc_lw_up_all_mon
        dt['surf_net_lw'] = ds.sfc_net_lw_all_mon

        dt['surf_net_flux'] =  ds.sfc_net_sw_all_mon + ds.sfc_net_lw_all_mon

        surf_flux[ds_name] = dt

    # ================== Read NCEP ================== #
    def read_ncep():
        obs_dir = 'NCEP'
        ds_name = 'NCEP'

        # cloudy sky upward lw at surf
        file_nm = 'ulwrf.sfc.gauss.2017.nc'
        ds = xr.open_dataset(os.path.join(base_dir, obs_dir, file_nm), decode_times=False)
        add_datetime_info(ds)
        surf_lw_up = ds.ulwrf.groupby('month').mean('time')

        # cloudy sky downward lw at surf
        file_nm = 'dlwrf.sfc.mon.2017.nc'
        ds = xr.open_dataset(os.path.join(base_dir, obs_dir, file_nm), decode_times=False)
        add_datetime_info(ds)
        surf_lw_down = ds.dlwrf.groupby('month').mean('time')

        # cloudy sky upward sw at sfc
        file_nm = 'uswrf.sfc.gauss.2017.nc'
        ds = xr.open_dataset(os.path.join(base_dir, obs_dir, file_nm), decode_times=False)
        add_datetime_info(ds)
        surf_sw_up = ds.uswrf.groupby('month').mean('time')
        # downward sw flux
        file_nm = 'dswrf.sfc.mon.2017.nc'
        ds = xr.open_dataset(os.path.join(base_dir, obs_dir, file_nm), decode_times=False)
        add_datetime_info(ds)
        surf_sw_down = ds.dswrf.groupby('month').mean('time')

        calc_and_add_flux_to_dict(surf_flux, ds_name, surf_sw_up, surf_sw_down, surf_lw_up, surf_lw_down)

    # ================== Read CFSR ================== #
    def read_ncep_cfsr():
        obs_dir = 'NCEP/CFSR'
        ds_name = 'CFSR'
        file_nm = 'pgbl01.gdas.2010.grb2.nc'  # 01, 02 in filename means how many hours to average
        ds = xr.open_dataset(os.path.join(base_dir, obs_dir, file_nm), decode_times=False)
       
        # cloudy sky upward lw at surf
        surf_lw_up = ds.ULWRF_L1_FcstAvg
        # cloudy sky downward lw at surf
        surf_lw_down = ds.DLWRF_L1_FcstAvg

        # cloudy sky upward sw at surf
        surf_sw_up = ds.USWRF_L1_FcstAvg
        # downward shortwave radiation flux at toa
        surf_sw_down = ds.DSWRF_L1_FcstAvg

        calc_and_add_flux_to_dict(surf_flux, ds_name, surf_sw_up, surf_sw_down, surf_lw_up, surf_lw_down)

    # ================== Read JRA55 ================== #
    def read_jra55():
        obs_dir = 'JRA55'
        ds_name = 'JRA55'

        # cloudy sky upward lw
        file_nm = 'fcst_phy2m.212_ulwrf.reg_tl319.201001_201012.liu357967.nc'
        ds = xr.open_dataset(os.path.join(base_dir, obs_dir, file_nm), decode_times=False)
        surf_lw_up = ds.ULWRF_GDS4_SFC_S130 #.mean('initial_time0_hours')

        # cloudy sky downward lw
        file_nm = 'fcst_phy2m.205_dlwrf.reg_tl319.201001_201012.liu357967.nc'
        ds = xr.open_dataset(os.path.join(base_dir, obs_dir, file_nm), decode_times=False)
        surf_lw_down = ds.DLWRF_GDS4_SFC_S130 #.mean('initial_time0_hours')

        # cloudy sky upward sw
        file_nm = 'fcst_phy2m.211_uswrf.reg_tl319.201001_201012.liu357967.nc'
        ds = xr.open_dataset(os.path.join(base_dir, obs_dir, file_nm), decode_times=False)
        surf_sw_up = ds.USWRF_GDS4_SFC_S130
        # downward sw
        file_nm = 'fcst_phy2m.204_dswrf.reg_tl319.201001_201012.liu357967.nc'
        ds = xr.open_dataset(os.path.join(base_dir, obs_dir, file_nm), decode_times=False)
        surf_sw_down = ds.DSWRF_GDS4_SFC_S130

        calc_and_add_flux_to_dict(surf_flux, ds_name, surf_sw_up, surf_sw_down, surf_lw_up, surf_lw_down)

    # ================== Read ERA-Interim ================== #
    def read_era_interim():
        obs_dir = 'ecmwf_data'
        ds_name = 'ERA-Interim'
        file_nm = 'toa_sfc_cld_clear_flux_monthly_2017.nc'
        ds = xr.open_dataset(os.path.join(base_dir, obs_dir, file_nm), decode_times=False)
        add_datetime_info(ds)
        # Change J/m^2 to W/m^2
        time_step = 3600.0 * (ds.time[1]-ds.time[0])

        # cloudy sky net lw at surface
        surf_net_lw = -ds.str.groupby('month').mean('time') / time_step # downward is positive
        # downward lw flux
        surf_lw_down = ds.strd.groupby('month').mean('time') / time_step
        # upward lw flux
        surf_lw_up = surf_lw_down - surf_net_lw

        # cloudy sky net sw at surf
        surf_net_sw = ds.ssr.groupby('month').mean('time') / time_step
        # downward sw flux
        surf_sw_down = ds.ssrd.groupby('month').mean('time') / time_step
        # upward sw flux
        surf_sw_up = surf_net_sw - surf_sw_down

        calc_and_add_flux_to_dict(surf_flux, ds_name, surf_sw_up, surf_sw_down, surf_lw_up, surf_lw_down)

    # Control flow
    surf_flux = {}

    if dataset_name.upper()=='ALL':
        read_ceres_ebaf_ed4p1()
        read_ceres_ebaf_ed4()
        read_ceres_ebaf_ed28()
        read_ncep()
        read_ncep_cfsr()
        read_jra55()
        read_era_interim()
    elif dataset_name.upper()=='CERES_EBAF ED4.1':    
        read_ceres_ebaf_ed4p1()
    elif dataset_name.upper()=='CERES_EBAF ED4.0':
        read_ceres_ebaf_ed4()
    elif dataset_name.upper()=='CERES_EBAF ED2.8':
        read_ceres_ebaf_ed28()
    elif dataset_name.upper()=='NCEP':
        read_ncep()
    elif dataset_name.upper()=='NCEP/CFSR':
        read_ncep_cfsr()
    elif dataset_name.upper()=='JRA55':
        read_jra55()
    elif dataset_name.upper()=='ERA-INTERIM':
        read_era_interim()
    else:
        print('Available datasets are: CERES_EBAF Ed4.0/4.1, CERES_EBAF Ed2.8, NCEP, NCEP/CFSR, JRA55, ERA-Interim')
 
    return surf_flux   


def read_surf_flux_clr_obs(dataset_name='ALL', base_dir='/scratch/ql260/obs_datasets/'):
    """[ ###'toa_sw_up', 'toa_net_sw', 'olr', 'toa_net_flux',###
            'surf_sw_down_clr', 'surf_sw_up_clr', 'surf_net_sw_clr',
            'surf_lw_down_clr', 'surf_lw_up_clr', 'surf_net_lw_clr',
            'surf_net_flux_clr' ]
    """
    def calc_and_add_flux_to_dict(surf_flux, ds_name, surf_sw_up_clr, surf_sw_down_clr, surf_lw_up_clr, surf_lw_down_clr):
        dt = {}     
        dt['surf_sw_up_clr'] = rename_dims_to_standards(surf_sw_up_clr)
        dt['surf_sw_down_clr'] = rename_dims_to_standards(surf_sw_down_clr)
        dt['surf_net_sw_clr'] =  dt['surf_sw_down_clr'] - dt['surf_sw_up_clr']
        
        dt['surf_lw_up_clr'] = rename_dims_to_standards(surf_lw_up_clr)
        dt['surf_lw_down_clr'] = rename_dims_to_standards(surf_lw_down_clr)
        dt['surf_net_lw_clr'] =  dt['surf_lw_down_clr'] - dt['surf_lw_up_clr']

        dt['surf_net_flux_clr'] = dt['surf_net_sw_clr'] + dt['surf_net_lw_clr']

        surf_flux[ds_name] = dt

    # ================== Read CERES ================== #
    def read_ceres_ebaf_ed4():
        ds_name = 'CERES_EBAF Ed4.0'
        obs_dir = 'CERES'
        file_nm = 'CERES_EBAF-Surface_Ed4.0_Subset_201701-201712.nc'
        dt = {}
        ds = xr.open_dataset(os.path.join(base_dir, obs_dir, file_nm), decode_times=False)
        dt['surf_sw_down_clr'] = ds.sfc_sw_down_clr_mon
        dt['surf_sw_up_clr'] = ds.sfc_sw_up_clr_mon
        dt['surf_net_sw_clr'] = ds.sfc_net_sw_clr_mon

        dt['surf_lw_down_clr'] = ds.sfc_lw_down_clr_mon 
        dt['surf_lw_up_clr'] = ds.sfc_lw_up_clr_mon
        dt['surf_net_lw_clr'] = ds.sfc_net_lw_clr_mon

        dt['surf_net_flux_clr'] =  ds.sfc_net_sw_clr_mon + ds.sfc_net_lw_clr_mon

        surf_flux[ds_name] = dt

    def read_ceres_ebaf_ed28():
        ds_name = 'CERES_EBAF Ed2.8'
        obs_dir = 'CERES'
        file_nm = 'CERES_EBAF-Surface_Ed2.8_Subset_201601-201612.nc'
        dt = {}
        ds = xr.open_dataset(os.path.join(base_dir, obs_dir, file_nm), decode_times=False)
        dt['surf_sw_down_clr'] = ds.sfc_sw_down_clr_mon
        dt['surf_sw_up_clr'] = ds.sfc_sw_up_clr_mon
        dt['surf_net_sw_clr'] = ds.sfc_net_sw_clr_mon

        dt['surf_lw_down_clr'] = ds.sfc_lw_down_clr_mon 
        dt['surf_lw_up_clr'] = ds.sfc_lw_up_clr_mon
        dt['surf_net_lw_clr'] = ds.sfc_net_lw_clr_mon

        dt['surf_net_flux_clr'] =  ds.sfc_net_sw_clr_mon + ds.sfc_net_lw_clr_mon

        surf_flux[ds_name] = dt

    # ================== Read NCEP ================== #
    def read_ncep():
        obs_dir = 'NCEP'
        ds_name = 'NCEP'

        # Don't find yet, it's the same to cloudy sky one
        # upward lw at surf
        #file_nm = 'ulwrf.sfc.gauss.2017.nc'
        #ds = xr.open_dataset(os.path.join(base_dir, obs_dir, file_nm), decode_times=False)
        #add_datetime_info(ds)
        #surf_lw_up_clr = ds.ulwrf.groupby('month').mean('time')

        # cloudy sky upward lw at surf
        file_nm = 'ulwrf.sfc.gauss.2017.nc'
        ds = xr.open_dataset(os.path.join(base_dir, obs_dir, file_nm), decode_times=False)
        add_datetime_info(ds)
        surf_lw_up = ds.ulwrf.groupby('month').mean('time')

        # cloudy sky downward lw at surf
        file_nm = 'dlwrf.sfc.mon.2017.nc'
        ds = xr.open_dataset(os.path.join(base_dir, obs_dir, file_nm), decode_times=False)
        add_datetime_info(ds)
        surf_lw_down = ds.dlwrf.groupby('month').mean('time')

        file_nm = 'cfnlf.sfc.gauss.2017.nc'
        ds = xr.open_dataset(os.path.join(base_dir, obs_dir, file_nm), decode_times=False)
        add_datetime_info(ds)
        surf_lw_cre = ds.cfnlf.groupby('month').mean('time')
        
        # clear_sky upward lw at surf
        surf_lw_up_clr = surf_lw_cre - (surf_lw_down-surf_lw_down_clr) + surf_lw_up

        # clear sky downward lw at surf
        file_nm = 'csdlf.sfc.mon.2017.nc'
        ds = xr.open_dataset(os.path.join(base_dir, obs_dir, file_nm), decode_times=False)
        add_datetime_info(ds)
        surf_lw_down_clr = ds.csdlf.groupby('month').mean('time')

        # clear sky upward sw at sfc
        file_nm = 'csusf.sfc.gauss.2017.nc'
        ds = xr.open_dataset(os.path.join(base_dir, obs_dir, file_nm), decode_times=False)
        add_datetime_info(ds)
        surf_sw_up_clr = ds.csusf.groupby('month').mean('time')
        # downward sw flux
        file_nm = 'csdsf.sfc.mon.2017.nc'
        ds = xr.open_dataset(os.path.join(base_dir, obs_dir, file_nm), decode_times=False)
        add_datetime_info(ds)
        surf_sw_down_clr = ds.csdsf.groupby('month').mean('time')

        calc_and_add_flux_to_dict(surf_flux, ds_name, surf_sw_up_clr, surf_sw_down_clr, surf_lw_up_clr, surf_lw_down_clr)

    # ================== Read CFSR ================== #
    def read_ncep_cfsr():
        obs_dir = 'NCEP/CFSR'
        ds_name = 'CFSR'
        file_nm = 'pgbl01.gdas.2010.grb2.nc'  # 01, 02 in filename means how many hours to average
        ds = xr.open_dataset(os.path.join(base_dir, obs_dir, file_nm), decode_times=False)
       
        # clear sky upward lw at surf
        surf_lw_up_clr = ds.CSULF_L1_FcstAvg
        # clear sky downward lw at surf
        surf_lw_down_clr = ds.CSDLF_L1_FcstAvg

        # clear sky upward sw at surf
        surf_sw_up_clr = ds.CSUSF_L1_FcstAvg
        # downward shortwave radiation flux at surf
        surf_sw_down_clr = ds.CSDSF_L1_FcstAvg

        calc_and_add_flux_to_dict(surf_flux, ds_name, surf_sw_up_clr, surf_sw_down_clr, surf_lw_up_clr, surf_lw_down_clr)

    # ================== Read JRA55 ================== #
    def read_jra55():
        obs_dir = 'JRA55'
        ds_name = 'JRA55'

        # clear sky upward lw (the same as cloudy?)
        #file_nm = 'fcst_phy2m.162_csulf.reg_tl319.201001_201012.liu357967.nc' # only toa data
        file_nm = 'fcst_phy2m.212_ulwrf.reg_tl319.201001_201012.liu357967.nc'
        ds = xr.open_dataset(os.path.join(base_dir, obs_dir, file_nm), decode_times=False)
        surf_lw_up_clr = ds.ULWRF_GDS4_SFC_S130 #.mean('initial_time0_hours')

        # clear sky downward lw
        file_nm = 'fcst_phy2m.163_csdlf.reg_tl319.201001_201012.liu357967.nc'
        ds = xr.open_dataset(os.path.join(base_dir, obs_dir, file_nm), decode_times=False)
        surf_lw_down_clr = ds.CSDLF_GDS4_SFC_S130 #.mean('initial_time0_hours')

        # clear sky upward sw
        file_nm = 'fcst_phy2m.160_csusf.reg_tl319.201001_201012.liu357967.nc'
        ds = xr.open_dataset(os.path.join(base_dir, obs_dir, file_nm), decode_times=False)
        surf_sw_up_clr = ds.CSUSF_GDS4_SFC_S130
        # downward sw
        file_nm = 'fcst_phy2m.161_csdsf.reg_tl319.201001_201012.liu357967.nc'
        ds = xr.open_dataset(os.path.join(base_dir, obs_dir, file_nm), decode_times=False)
        surf_sw_down_clr = ds.CSDSF_GDS4_SFC_S130

        calc_and_add_flux_to_dict(surf_flux, ds_name, surf_sw_up_clr, surf_sw_down_clr, surf_lw_up_clr, surf_lw_down_clr)

    # ================== Read ERA-Interim ================== #
    def read_era_interim():
        obs_dir = 'ecmwf_data'
        ds_name = 'ERA-Interim'
        file_nm = 'toa_sfc_cld_clear_flux_monthly_2017.nc'
        ds = xr.open_dataset(os.path.join(base_dir, obs_dir, file_nm), decode_times=False)
        add_datetime_info(ds)
        # Change J/m^2 to W/m^2
        time_step = 3600.0 * (ds.time[1]-ds.time[0])

        # clear sky net lw at surface
        surf_net_lw_clr = -ds.strc.groupby('month').mean('time') / time_step # downward is positive
        # downward lw flux 
        #surf_lw_down_clr = ds.strd.groupby('month').mean('time') / time_step
        # upward lw flux
        #surf_lw_up_clr = surf_lw_down_clr - surf_net_lw_clr

        # clear sky net sw at surf
        surf_net_sw_clr = ds.ssrc.groupby('month').mean('time') / time_step
        # downward sw flux
        #surf_sw_down_clr = ds.ssrd.groupby('month').mean('time') / time_step
        # upward sw flux
        #surf_sw_up_clr = surf_net_sw_clr - surf_sw_down_clr

        #calc_and_add_flux_to_dict(surf_flux, ds_name, surf_sw_up_clr, surf_sw_down_clr, surf_lw_up_clr, surf_lw_down_clr)

    # Control flow
    surf_flux = {}

    if dataset_name.upper()=='ALL':
        read_ceres_ebaf_ed4()
        read_ceres_ebaf_ed28()
        read_ncep()
        read_ncep_cfsr()
        read_jra55()
        read_era_interim()
    elif dataset_name.upper()=='CERES_EBAF ED4.0':
        read_ceres_ebaf_ed4()
    elif dataset_name.upper()=='CERES_EBAF ED2.8':
        read_ceres_ebaf_ed28()
    elif dataset_name.upper()=='NCEP':
        read_ncep()
    elif dataset_name.upper()=='NCEP/CFSR':
        read_ncep_cfsr()
    elif dataset_name.upper()=='JRA55':
        read_jra55()
    elif dataset_name.upper()=='ERA-INTERIM':
        read_era_interim()
    else:
        print('Available datasets are: CERES_EBAF Ed4.0, CERES_EBAF Ed2.8, NCEP, NCEP/CFSR, JRA55, ERA-Interim')
 
    return surf_flux   


def read_surf_CRE_obs(dataset_name='ALL', base_dir='/scratch/ql260/obs_datasets/'):
    """Show the LW and SW difference between cloudy and clear sky
    Or downward LW/SW difference"""
    
    def calc_and_add_flux_to_dict(surf_cre, ds_name, surf_sw_up, surf_sw_down, 
                                surf_lw_up, surf_lw_down, surf_sw_up_clr, 
                                surf_sw_down_clr, surf_lw_up_clr, surf_lw_down_clr):
        dt = {}
        dt['surf_lw_down_cre'] = rename_dims_to_standards(surf_lw_down) - \
                                 rename_dims_to_standards(surf_lw_down_clr)
        dt['surf_lw_up_cre'] = rename_dims_to_standards(surf_lw_up) - \
                               rename_dims_to_standards(surf_lw_up_clr)
        dt['surf_sw_down_cre'] = rename_dims_to_standards(surf_sw_down) - \
                                 rename_dims_to_standards(surf_sw_down_clr)
        dt['surf_sw_up_cre'] = rename_dims_to_standards(surf_sw_up) - \
                               rename_dims_to_standards(surf_sw_up_clr)

        dt['surf_lw_cre'] = dt['surf_lw_down_cre'] - dt['surf_lw_up_cre']
        dt['surf_sw_cre'] = dt['surf_sw_down_cre'] - dt['surf_sw_up_cre']
        dt['surf_net_cre'] = dt['surf_lw_cre'] + dt['surf_sw_cre']
        
        surf_cre[ds_name] = dt

    # ================== Read CERES ================== #
    def read_ceres_ebaf_ed4():
        ds_name = 'CERES_EBAF Ed4.0'
        obs_dir = 'CERES'
        file_nm = 'CERES_EBAF-Surface_Ed4.0_Subset_201701-201712.nc'
        dt = {}
        ds = xr.open_dataset(os.path.join(base_dir, obs_dir, file_nm), decode_times=False)
        dt['surf_lw_cre'] = ds.sfc_cre_net_lw_mon
        dt['surf_sw_cre'] = ds.sfc_cre_net_sw_mon
        dt['surf_net_cre'] = ds.sfc_cre_net_tot_mon
        
        dt['surf_lw_down_cre'] = ds.sfc_lw_down_all_mon - ds.sfc_lw_down_clr_mon
        dt['surf_lw_up_cre'] = ds.sfc_lw_up_all_mon - ds.sfc_lw_up_clr_mon
        dt['surf_sw_down_cre'] = ds.sfc_sw_down_all_mon - ds.sfc_sw_down_clr_mon
        dt['surf_sw_up_cre'] = ds.sfc_sw_up_all_mon - ds.sfc_sw_up_clr_mon
    
        """ 
        # check how to calculate net LW CRE at surface
        lwdn = (ds.sfc_lw_down_all_mon-ds.sfc_lw_down_clr_mon).mean()
        lwup = (ds.sfc_lw_up_all_mon-ds.sfc_lw_up_clr_mon).mean()
        print(lwdn.values, lwup.values, lwdn.values-lwup.values, ds.sfc_cre_net_lw_mon.mean().values )
        
        # check how to calculate net SW CRE at surface
        swdn = (ds.sfc_sw_down_all_mon-ds.sfc_sw_down_clr_mon).mean()
        swup = (ds.sfc_sw_up_all_mon-ds.sfc_sw_up_clr_mon).mean()
        print(swdn.values, swup.values, swdn.values-swup.values, ds.sfc_cre_net_sw_mon.mean().values )
        """
        surf_cre[ds_name] = dt

    def read_ceres_ebaf_ed28():
        ds_name = 'CERES_EBAF Ed2.8'
        obs_dir = 'CERES'
        file_nm = 'CERES_EBAF-Surface_Ed2.8_Subset_201601-201612.nc'
        dt = {}
        ds = xr.open_dataset(os.path.join(base_dir, obs_dir, file_nm), decode_times=False)
        dt['surf_lw_cre'] = ds.sfc_cre_net_lw_mon 
        dt['surf_sw_cre'] = ds.sfc_cre_net_sw_mon
        dt['surf_net_cre'] = ds.sfc_cre_net_tot_mon
        
        dt['surf_lw_down_cre'] = ds.sfc_lw_down_all_mon - ds.sfc_lw_down_clr_mon
        dt['surf_lw_up_cre'] = ds.sfc_lw_up_all_mon - ds.sfc_lw_up_clr_mon
        dt['surf_sw_down_cre'] = ds.sfc_sw_down_all_mon - ds.sfc_sw_down_clr_mon
        dt['surf_sw_up_cre'] = ds.sfc_sw_up_all_mon - ds.sfc_sw_up_clr_mon

        surf_cre[ds_name] = dt

    # ================== Read NCEP ================== #
    def read_ncep():
        obs_dir = 'NCEP'
        ds_name = 'NCEP'

        file_nm = 'cfnlf.sfc.gauss.2017.nc'
        ds = xr.open_dataset(os.path.join(base_dir, obs_dir, file_nm), decode_times=False)
        add_datetime_info(ds)
        surf_lw_cre = ds.cfnlf.groupby('month').mean('time')

        #file_nm = 'cfnsf.sfc.gauss.2017.nc'
        #ds = xr.open_dataset(os.path.join(base_dir, obs_dir, file_nm), decode_times=False)
        #add_datetime_info(ds)
        #surf_sw_cre = ds.cfnsf.groupby('month').mean('time')

        # --------- Cloudy ------------#
        # cloudy sky upward lw at surf
        file_nm = 'ulwrf.sfc.gauss.2017.nc'
        ds = xr.open_dataset(os.path.join(base_dir, obs_dir, file_nm), decode_times=False)
        add_datetime_info(ds)
        surf_lw_up = ds.ulwrf.groupby('month').mean('time')

        # cloudy sky downward lw at surf
        file_nm = 'dlwrf.sfc.mon.2017.nc'
        ds = xr.open_dataset(os.path.join(base_dir, obs_dir, file_nm), decode_times=False)
        add_datetime_info(ds)
        surf_lw_down = ds.dlwrf.groupby('month').mean('time')

        # cloudy sky upward sw at sfc
        file_nm = 'uswrf.sfc.gauss.2017.nc'
        ds = xr.open_dataset(os.path.join(base_dir, obs_dir, file_nm), decode_times=False)
        add_datetime_info(ds)
        surf_sw_up = ds.uswrf.groupby('month').mean('time')
        # downward sw flux
        file_nm = 'dswrf.sfc.mon.2017.nc'
        ds = xr.open_dataset(os.path.join(base_dir, obs_dir, file_nm), decode_times=False)
        add_datetime_info(ds)
        surf_sw_down = ds.dswrf.groupby('month').mean('time')

        # --------- Clear ------------ #
        # clear sky downward lw at surf
        file_nm = 'csdlf.sfc.mon.2017.nc'
        ds = xr.open_dataset(os.path.join(base_dir, obs_dir, file_nm), decode_times=False)
        add_datetime_info(ds)
        surf_lw_down_clr = ds.csdlf.groupby('month').mean('time')

        # clear_sky upward lw at surf
        surf_lw_up_clr = surf_lw_cre - (surf_lw_down-surf_lw_down_clr) + surf_lw_up

        # clear sky upward sw at sfc
        file_nm = 'csusf.sfc.gauss.2017.nc'
        ds = xr.open_dataset(os.path.join(base_dir, obs_dir, file_nm), decode_times=False)
        add_datetime_info(ds)
        surf_sw_up_clr = ds.csusf.groupby('month').mean('time')
        # downward sw flux
        file_nm = 'csdsf.sfc.mon.2017.nc'
        ds = xr.open_dataset(os.path.join(base_dir, obs_dir, file_nm), decode_times=False)
        add_datetime_info(ds)
        surf_sw_down_clr = ds.csdsf.groupby('month').mean('time')

        #dt['surf_net_cre'] = dt['surf_lw_cre'] + dt['surf_sw_cre'] 

        calc_and_add_flux_to_dict(surf_cre, ds_name, surf_sw_up, surf_sw_down,
                            surf_lw_up, surf_lw_down, surf_sw_up_clr, 
                            surf_sw_down_clr, surf_lw_up_clr, surf_lw_down_clr)

    # ================== Read CFSR ================== #
    def read_ncep_cfsr():
        obs_dir = 'NCEP/CFSR'
        ds_name = 'CFSR'
        file_nm = 'pgbl01.gdas.2010.grb2.nc'  # 01, 02 in filename means how many hours to average
        ds = xr.open_dataset(os.path.join(base_dir, obs_dir, file_nm), decode_times=False)
       
        # cloudy sky upward lw at surf
        surf_lw_up = ds.ULWRF_L1_FcstAvg
        # cloudy sky downward lw at surf
        surf_lw_down = ds.DLWRF_L1_FcstAvg

        # cloudy sky upward sw at surf
        surf_sw_up = ds.USWRF_L1_FcstAvg
        # downward shortwave radiation flux at surf
        surf_sw_down = ds.DSWRF_L1_FcstAvg
        
        # clear sky upward lw at surf
        surf_lw_up_clr = ds.CSULF_L1_FcstAvg
        # clear sky downward lw at surf
        surf_lw_down_clr = ds.CSDLF_L1_FcstAvg

        # clear sky upward sw at surf
        surf_sw_up_clr = ds.CSUSF_L1_FcstAvg
        # downward shortwave radiation flux at surf
        surf_sw_down_clr = ds.CSDSF_L1_FcstAvg

        calc_and_add_flux_to_dict(surf_cre, ds_name, surf_sw_up, surf_sw_down,
                            surf_lw_up, surf_lw_down, surf_sw_up_clr, 
                            surf_sw_down_clr, surf_lw_up_clr, surf_lw_down_clr)

    # ================== Read JRA55 ================== #
    def read_jra55():
        obs_dir = 'JRA55'
        ds_name = 'JRA55'

        # cloudy sky upward lw
        file_nm = 'fcst_phy2m.212_ulwrf.reg_tl319.201001_201012.liu357967.nc'
        ds = xr.open_dataset(os.path.join(base_dir, obs_dir, file_nm), decode_times=False)
        surf_lw_up = ds.ULWRF_GDS4_SFC_S130 #.mean('initial_time0_hours')

        # cloudy sky downward lw
        file_nm = 'fcst_phy2m.205_dlwrf.reg_tl319.201001_201012.liu357967.nc'
        ds = xr.open_dataset(os.path.join(base_dir, obs_dir, file_nm), decode_times=False)
        surf_lw_down = ds.DLWRF_GDS4_SFC_S130 #.mean('initial_time0_hours')

        # cloudy sky upward sw
        file_nm = 'fcst_phy2m.211_uswrf.reg_tl319.201001_201012.liu357967.nc'
        ds = xr.open_dataset(os.path.join(base_dir, obs_dir, file_nm), decode_times=False)
        surf_sw_up = ds.USWRF_GDS4_SFC_S130
        # downward sw
        file_nm = 'fcst_phy2m.204_dswrf.reg_tl319.201001_201012.liu357967.nc'
        ds = xr.open_dataset(os.path.join(base_dir, obs_dir, file_nm), decode_times=False)
        surf_sw_down = ds.DSWRF_GDS4_SFC_S130

        # clear sky upward lw (the same as cloudy?)
        #file_nm = 'fcst_phy2m.162_csulf.reg_tl319.201001_201012.liu357967.nc' # only toa data
        #file_nm = 'fcst_phy2m.212_ulwrf.reg_tl319.201001_201012.liu357967.nc'
        #ds = xr.open_dataset(os.path.join(base_dir, obs_dir, file_nm), decode_times=False)
        surf_lw_up_clr = surf_lw_up #ds.ULWRF_GDS4_SFC_S130 #.mean('initial_time0_hours')

        # clear sky downward lw
        file_nm = 'fcst_phy2m.163_csdlf.reg_tl319.201001_201012.liu357967.nc'
        ds = xr.open_dataset(os.path.join(base_dir, obs_dir, file_nm), decode_times=False)
        surf_lw_down_clr = ds.CSDLF_GDS4_SFC_S130 #.mean('initial_time0_hours')

        # clear sky upward sw
        file_nm = 'fcst_phy2m.160_csusf.reg_tl319.201001_201012.liu357967.nc'
        ds = xr.open_dataset(os.path.join(base_dir, obs_dir, file_nm), decode_times=False)
        surf_sw_up_clr = ds.CSUSF_GDS4_SFC_S130
        # downward sw
        file_nm = 'fcst_phy2m.161_csdsf.reg_tl319.201001_201012.liu357967.nc'
        ds = xr.open_dataset(os.path.join(base_dir, obs_dir, file_nm), decode_times=False)
        surf_sw_down_clr = ds.CSDSF_GDS4_SFC_S130

        calc_and_add_flux_to_dict(surf_cre, ds_name, surf_sw_up, surf_sw_down,
                    surf_lw_up, surf_lw_down, surf_sw_up_clr, 
                    surf_sw_down_clr, surf_lw_up_clr, surf_lw_down_clr)

    # ================== Read ERA-Interim ================== #
    def read_era_interim():
        obs_dir = 'ecmwf_data'
        ds_name = 'ERA-Interim'
        file_nm = 'toa_sfc_cld_clear_flux_monthly_2017.nc'
        ds = xr.open_dataset(os.path.join(base_dir, obs_dir, file_nm), decode_times=False)
        add_datetime_info(ds)
        # Change J/m^2 to W/m^2
        time_step = 3600.0 * (ds.time[1]-ds.time[0])

        # cloudy sky net lw at surface
        surf_net_lw = ds.str.groupby('month').mean('time') / time_step
        # downward lw flux
        #surf_lw_down = ds.strd.groupby('month').mean('time') / time_step
        # upward lw flux
        #surf_lw_up = surf_lw_down - surf_net_lw

        # cloudy sky net sw at surf
        surf_net_sw = ds.ssr.groupby('month').mean('time') / time_step
        # downward sw flux
        #surf_sw_down = ds.ssrd.groupby('month').mean('time') / time_step
        # upward sw flux
        #surf_sw_up = surf_net_sw - surf_sw_down

        # clear sky net lw at surface
        surf_net_lw_clr = ds.strc.groupby('month').mean('time') / time_step
        # clear sky net sw at surf
        surf_net_sw_clr = ds.ssrc.groupby('month').mean('time') / time_step

        dt = {}
        dt['surf_lw_cre'] = rename_dims_to_standards(surf_net_lw) - \
                            rename_dims_to_standards(surf_net_lw_clr)
        dt['surf_sw_cre'] = rename_dims_to_standards(surf_net_sw) - \
                               rename_dims_to_standards(surf_net_sw_clr)
        dt['surf_net_cre'] = dt['surf_lw_cre'] + dt['surf_sw_cre']
        
        surf_cre[ds_name] = dt

    # Control flow
    surf_cre = {}

    if dataset_name.upper()=='ALL':
        read_ceres_ebaf_ed4()
        read_ceres_ebaf_ed28()
        read_ncep()
        read_ncep_cfsr()
        read_jra55()
        read_era_interim()
    elif dataset_name.upper()=='CERES_EBAF ED4.0':
        read_ceres_ebaf_ed4()
    elif dataset_name.upper()=='CERES_EBAF ED2.8':
        read_ceres_ebaf_ed28()
    elif dataset_name.upper()=='NCEP':
        read_ncep()
    elif dataset_name.upper()=='NCEP/CFSR':
        read_ncep_cfsr()
    elif dataset_name.upper()=='JRA55':
        read_jra55()
    elif dataset_name.upper()=='ERA-INTERIM':
        read_era_interim()
    else:
        print('Available datasets are: CERES_EBAF Ed4.0, CERES_EBAF Ed2.8, NCEP, NCEP/CFSR, JRA55, ERA-Interim')
 
    return surf_cre   

