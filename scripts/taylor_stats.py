from __future__ import print_function
import numpy as np
import xarray as xr
import pandas as pd
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import matplotlib.pyplot as plt
import string
from taylorDiagram import TaylorDiagram
from plot_spatial_figs import interpolate_obs_dict


def fill_ndarr_with_1darr(arr_1d, shp, axis=None):
    arr_len = len(arr_1d)
    # There is potential bug here as there are possible more than
    # one dimensions equals to arr_len
    if axis is None:
        for n, nlen in enumerate(shp):
            if nlen == arr_len:
                axis = n
    if axis is not None:
        newshp = [x for i,x in enumerate(shp) if i!=axis]
        newshp.append(1)
        arr_nd = np.tile(arr_1d, tuple(newshp))
        arr_nd = np.moveaxis(arr_nd, -1, axis)
        return arr_nd
    else:
        print('Error: no dim is suitable.')


def pattern_cor(mod, obs, weights):
    """
    Refer to https://www.ncl.ucar.edu/Document/Functions/Contributed/pattern_cor.shtml
    pattern_cor2 function in "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl"
    """
    mod = np.array(mod)
    obs = np.array(obs)
    weights = np.array(weights)
    dimx = np.shape(mod)
    WGT = fill_ndarr_with_1darr(weights, dimx, axis=0)

    sumWGT   = np.sum(WGT)
    xAvgArea = np.sum(mod * WGT) / sumWGT # weighted area average
    yAvgArea = np.sum(obs * WGT) / sumWGT

    xAnom    = mod - xAvgArea  # anomalies
    yAnom    = obs - yAvgArea

    xyCov    = np.sum(WGT * xAnom * yAnom)
    xAnom2   = np.sum(WGT * xAnom**2)
    yAnom2   = np.sum(WGT * yAnom**2)

    r = xyCov / ( np.sqrt(xAnom2) * np.sqrt(yAnom2) )

    return r


def calc_rmse(mod, obs, weights):
    mod = np.array(mod)
    obs = np.array(obs)
    weights = np.array(weights)
    dimx = np.shape(mod)
    wgt = fill_ndarr_with_1darr(weights, dimx, axis=0)
    #np.sqrt(wgt * (mod-obs)**2 / np.prod(np.shape(mod)))
    rmse = np.sqrt(np.average((mod-obs)**2, axis=(0,1), weights=wgt))

    return rmse

def calc_centered_rmse(mod, obs, weights):
    # https://pcmdi.llnl.gov/staff/taylor/CV/Taylor_diagram_primer.pdf?id=96
    mod = np.array(mod)
    obs = np.array(obs)
    weights = np.array(weights)
    dimx = np.shape(mod)

    WGT = fill_ndarr_with_1darr(weights, dimx, axis=0)
    sumWGT   = np.sum(WGT)
    xAvgArea = np.sum(mod * WGT) / sumWGT # weighted area average
    yAvgArea = np.sum(obs * WGT) / sumWGT

    xAnom    = mod - xAvgArea  # anomalies
    yAnom    = obs - yAvgArea

    #rmse = np.sqrt( np.sum(WGT*(xAnom-yAnom)**2 ) / np.prod(np.shape(mod)))
    rmse = np.sqrt(np.average((xAnom-yAnom)**2, axis=(0,1), weights=WGT))

    return rmse


def calc_stddev(mod, obs, norm=True):
    mod = np.array(mod)
    obs = np.array(obs)

    stddev_mod = np.std(mod)
    stddev_obs = np.std(obs)

    if norm:
        stddev_mod = stddev_mod / stddev_obs
        stddev_obs = 1.0
    
    return stddev_mod, stddev_obs


def get_obs_dict_tm(obs_dict, lats, lons):
    obs_dict1 = interpolate_obs_dict(obs_dict, lats, lons)
    obs_dict_tm = {}
    for key,val in obs_dict1.items():
        try:
            #print(key)
            obs_dict_tm[key] = val.mean('time')
        except:
            obs_dict_tm[key] = val
    return obs_dict_tm


def taylor_stats(dt_dict, obs_dict, varnm, obs_varnm=None, tm=False, norm=True):
    lats = np.array(dt_dict[varnm]['lat'])
    lons = np.array(dt_dict[varnm]['lon'])
    #print(len(lats), len(lons))
    if tm:
        mod = np.array(dt_dict[varnm].mean('time'))
    else:
        mod = np.array(dt_dict[varnm])
    obs_dict_tm = get_obs_dict_tm(obs_dict, lats, lons)
    if obs_varnm is None:
        obs = np.array(obs_dict_tm[varnm])
    else:
        obs = np.array(obs_dict_tm[obs_varnm])
    #print(np.shape(mod), np.shape(obs))
    weights = np.cos(np.deg2rad(lats))

    corr = pattern_cor(mod, obs, weights)
    stddev_mod, stddev_obs = calc_stddev(mod, obs, norm=norm)
    #rmse = calc_rmse(mod, obs, weights)
    #rmse = calc_centered_rmse(mod, obs, weights)
    rmse = np.sqrt(stddev_mod**2 + stddev_obs**2 - 2*stddev_mod*stddev_obs*corr)
    return [corr, rmse, stddev_mod, stddev_obs]


def get_taylor_stats_for_cmip5_isca(cmip_tm_dict, ds_isca_arr, exp_names, obs_dict, varnm1, varnm2, obs_varnm=None, norm=True):
    corr_arr = []
    rmse_arr = []
    stddev_mod_arr = []
    stddev_obs_arr = []
    label_arr = []

    for model_nm, dt_mod in cmip_tm_dict.items():
        stat = taylor_stats(dt_mod, obs_dict, varnm1, obs_varnm=obs_varnm, tm=False, norm=norm)
        # [corr, rmse, stddev_mod, stddev_obs]
        corr_arr.append(stat[0])
        rmse_arr.append(stat[1])
        stddev_mod_arr.append(stat[2])
        stddev_obs_arr.append(stat[3])
        label_arr.append(model_nm)

    for exp_nm, dt_isca in zip(exp_names, ds_isca_arr):
        stat = taylor_stats(dt_isca, obs_dict, varnm2, obs_varnm=obs_varnm, tm=True, norm=norm)
        # [corr, rmse, stddev_mod, stddev_obs]
        # print('exp '+exp_nm+': corr='+str(stat[0])+', rmse='+str(stat[1])+', stddev='+str(stat[2]))
        corr_arr.append(stat[0])
        rmse_arr.append(stat[1])
        stddev_mod_arr.append(stat[2])
        stddev_obs_arr.append(stat[3])
        label_arr.append(exp_nm)

    return [label_arr, corr_arr, rmse_arr, stddev_mod_arr, stddev_obs_arr]


def re_organize_cmip_isca_stat(stdrefs, samples, varnm, stats):
    '''
    stats: [label_arr, corr_arr, rmse_arr, stddev_mod_arr, stddev_obs_arr]
    '''
    stdrefs[varnm] = stats[-1][0]

    item_arr = []
    for label, corr, stddev_mod in zip(stats[0], stats[1], stats[3]):
        item = [stddev_mod, corr, label]
        item_arr.append(item)

    samples[varnm] = item_arr


def write_taylor_stats_for_cmip5_isca(cmip_tm_dict, ds_isca_arr, exp_names, obs_dict, varnm,
         obs_varnm=None, norm=True, out_fn='taylor.txt'):

    with open(out_fn, "w") as fn:
        fn.write('label, corr, rmse, stddev_mod, stddev_obs\n')

        for model_nm, dt_mod in cmip_tm_dict.items():
            stat = taylor_stats(dt_mod, obs_dict, varnm, obs_varnm=obs_varnm, tm=False, norm=norm)
            # [corr, rmse, stddev_mod, stddev_obs]
            #fn.write('%s, %.4f, %.4f, %.4f, %.4f\n'%(model_nm, stat[0], stat[1], stat[2], stat[3]))
            fn.write(model_nm + ', ' + ', '.join([str(x) for x in stat]) + '\n')

        for exp_nm, dt_isca in zip(exp_names, ds_isca_arr):
            stat = taylor_stats(dt_isca, obs_dict, varnm, obs_varnm=obs_varnm, tm=True, norm=norm)
            # [corr, rmse, stddev_mod, stddev_obs]
            # print('exp '+exp_nm+': corr='+str(stat[0])+', rmse='+str(stat[1])+', stddev='+str(stat[2]))
            #fn.write('%s, %.4f, %.4f, %.4f, %.4f\n'%(exp_nm, stat[0], stat[1], stat[2], stat[3]))
            fn.write(exp_nm + ', ' + ', '.join([str(x) for x in stat]) + '\n')

    print(out_fn + ' saved.')


def write_taylor_diagram_stats(ds_isca_arr, exp_names, obs_flux_dict, 
                            out_lwcre_fn='lwcre_taylor.txt',
                            out_swcre_fn='swcre_taylor.txt', 
                            out_netcre_fn='netcre_taylor.txt'):

    group_models = [('BNU', 'BNU-ESM'),
        ('CCCma', 'CanESM2'),
        ('CNRM-CERFACS', 'CNRM-CM5'),
        ('CSIRO-BOM', 'ACCESS1-0'),
        ('FIO', 'FIO-ESM'),
        ('INM', 'inmcm4'), 
        ('IPSL', 'IPSL-CM5A-LR'),
        ('IPSL', 'IPSL-CM5A-MR'),
        ('MIROC', 'MIROC-ESM'),
        ('MOHC', 'HadCM3'),
        ('MPI-M', 'MPI-ESM-LR'),
        ('MPI-M', 'MPI-ESM-MR'),
        ('MRI', 'MRI-CGCM3'),
        ('MRI', 'MRI-ESM1'),
        ('NASA-GISS', 'GISS-E2-H'),
        ('NCAR', 'CCSM4'),
        ('NCC', 'NorESM1-M'),
        ('NIMR-KMA', 'HadGEM2-AO'),
        ('NOAA-GFDL', 'GFDL-CM3'),
        ('NOAA-GFDL', 'GFDL-ESM2M'), 
        ('NSF-DOE-NCAR', 'CESM1-CAM5') ]
   
    variables = ["rlut", "rsut", "rlutcs", "rsutcs",]

    start_year = 1996
    end_year = 2005

    P = os.path.join
    exp_nm = "historical" # 'amip'
    hist_cre = {}
    for (group, model_nm) in group_models:
        var_dict = {}
        for var in variables:
            dt_dir = P('/scratch/ql260', 'cmip5_data', exp_nm, var)
            filename = model_nm+'_'+exp_nm+'_'+var+'_'+str(start_year)+'_'+str(end_year)+'.nc'
            ds = xr.open_dataset(P(dt_dir, filename), decode_times=False)
            var_dict[var] = ds[var]
        var_dict['toa_sw_cre'] = var_dict["rsutcs"] - var_dict["rsut"]

        if 'ccsm4' in model_nm.lower():
            ## The lats of rlutcs and rlut have small differences.
            lwcre = var_dict["rlutcs"].values - var_dict["rlut"].values
            times = var_dict["rlutcs"].time
            lats = var_dict["rlutcs"].lat
            lons = var_dict["rlutcs"].lon
            var_dict['toa_lw_cre'] = xr.DataArray(lwcre, coords=[times,lats,lons], dims=['time','lat','lon'])
        else:
            var_dict['toa_lw_cre'] = var_dict["rlutcs"] - var_dict["rlut"]
        var_dict['toa_net_cre'] = var_dict['toa_sw_cre'] + var_dict['toa_lw_cre']
        #print(var_dict['toa_sw_cre'], var_dict['toa_lw_cre'], var_dict['toa_net_cre'])
        hist_cre[model_nm] = var_dict
    
    hist_cre_tm = {}
    for kk, (model_nm, dt_cre) in enumerate(hist_cre.items()):
        dt_cre_tm = {}
        for key,val in dt_cre.items():
            dt_cre_tm[key] = val.mean('time')
        hist_cre_tm[model_nm] = dt_cre_tm

    stdrefs = {}
    samples = {}
    var_name_arr = []
    title_arr = []

    norm = False

    varnm = 'toa_lw_cre'
    write_taylor_stats_for_cmip5_isca(hist_cre_tm, ds_isca_arr, exp_names, obs_flux_dict, varnm, norm=norm, out_fn=out_lwcre_fn)

    varnm = 'toa_sw_cre'
    write_taylor_stats_for_cmip5_isca(hist_cre_tm, ds_isca_arr, exp_names, obs_flux_dict, varnm, norm=norm, out_fn=out_swcre_fn)

    varnm = 'toa_net_cre'
    write_taylor_stats_for_cmip5_isca(hist_cre_tm, ds_isca_arr, exp_names, obs_flux_dict, varnm, norm=norm, out_fn=out_netcre_fn)


def plot_taylor_diagram(lwcre_fn, swcre_fn, netcre_fn, figname):

    stdrefs = {}
    samples = {}
    var_name_arr = []
    title_arr = []

    norm = False

    varnm = 'toa_lw_cre'
    var_name_arr.append(varnm)
    title_arr.append('LW CRE at TOA')
    df = pd.read_csv(lwcre_fn, header=[0])
    # [label_arr, corr_arr, rmse_arr, stddev_mod_arr, stddev_obs_arr]
    lw_stat = []
    for col in df.columns:
        lw_stat.append(list(df[col].values))
    re_organize_cmip_isca_stat(stdrefs, samples, varnm, lw_stat)
    
    varnm = 'toa_sw_cre'
    var_name_arr.append(varnm)
    title_arr.append('SW CRE at TOA')
    df = pd.read_csv(swcre_fn, header=[0])
    # [label_arr, corr_arr, rmse_arr, stddev_mod_arr, stddev_obs_arr]
    sw_stat = []
    for col in df.columns:
        sw_stat.append(list(df[col].values))
    re_organize_cmip_isca_stat(stdrefs, samples, varnm, sw_stat)

    varnm = 'toa_net_cre'
    var_name_arr.append(varnm)
    title_arr.append('Net CRE at TOA')
    df = pd.read_csv(netcre_fn, header=[0])
    # [label_arr, corr_arr, rmse_arr, stddev_mod_arr, stddev_obs_arr]
    net_stat = []
    for col in df.columns:
        net_stat.append(list(df[col].values))
    re_organize_cmip_isca_stat(stdrefs, samples, varnm, net_stat)

    # Isca use different color with cmip
    N = len(lw_stat[0])
    n_cmip = N - 6
    n_isca = 6
    n_isca_half = int(n_isca/2)

    colors1 = plt.matplotlib.cm.Dark2_r(np.linspace(0, 0.7, n_cmip))
    colors2 = [np.squeeze(plt.matplotlib.cm.Dark2_r(np.linspace(0.8, 0.9, 1)))] * n_isca_half
    colors3 = [np.squeeze(plt.matplotlib.cm.Dark2_r(np.linspace(0.95, 1, 1)))] * n_isca_half
    #colors = colors1 + colors2 + colors3 # Not work for np.array
    colors = np.concatenate((colors1, colors2, colors3), axis=0)

    rects = {}
    for i, var in enumerate(var_name_arr):
        rects[var] = int('22'+str(i+1))

    xbase_arr = [3, 5, 3]
    fig = plt.figure(figsize=(10,8))

    xy_pos_arr = [(np.arccos(0.38), 15.5), (np.arccos(0.53), 26), (np.arccos(0.33), 19)]
    txt_pos_arr = [(np.arccos(0.21), 17), (np.arccos(0.4), 28), (np.arccos(0.2), 21)] # (theta, radius)
    for kk, (varnm, title, xbase, xy_pos, txt_pos) in enumerate(zip(var_name_arr, 
                                title_arr, xbase_arr, xy_pos_arr, txt_pos_arr)):
        if 'net_cre' in varnm:
            srange = (0, 1.6)
        else:
            srange = (0, 1.5)
        dia = TaylorDiagram(stdrefs[varnm], fig=fig, rect=rects[varnm],
                            label='Reference', xbase=xbase, srange=srange)

        # Add samples to Taylor diagram
        for i,(stddev, corrcoef, name) in enumerate(samples[varnm]):
            if i+1 < 10:
                i_ms = 8
            elif i+1 == 21:
                i_ms = 9
            else:
                i_ms = 10
            dia.add_sample(stddev, corrcoef, marker='$%d$' % (i+1), ms=i_ms, ls='',
                        linewidth=1, mfc=colors[i], mec=None, mew=0, label=name)

        # Add RMS contours, and label them
        contours = dia.add_contours(levels=5, colors='0.5') # 5 levels
        dia.ax.clabel(contours, inline=1, fontsize=10, fmt='%.1f')
        # Tricky: ax is the polar ax (used for plots), _ax is the
        # container (used for layout)
        dia._ax.set_title('('+string.ascii_lowercase[kk]+') '+title, loc='left', pad=15)

        # https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.axes.Axes.annotate.html
        dia.ax.annotate('RMSE',
            xy=xy_pos,  # theta, radius
            xytext=txt_pos,
            arrowprops=dict(arrowstyle='-|>'), #facecolor='black', shrink=0.05),
            horizontalalignment='left',
            verticalalignment='top')
    # Add a figure legend and title. For loc option, place x,y tuple inside [ ].
    # Can also use special options here:
    # http://matplotlib.sourceforge.net/users/legend_guide.html

    fig.legend(dia.samplePoints,
            [ p.get_label() for p in dia.samplePoints ],
            numpoints=1, prop=dict(size='small'), loc='center',
            ncol=2, bbox_to_anchor=(0.68, 0.24))

    fig.tight_layout()
    plt.subplots_adjust(wspace=-0.2)

    fig.savefig(figname, bbox_inches='tight', pad_inches=0.05, transparent=False)
