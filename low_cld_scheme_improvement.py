from __future__ import print_function
from __future__ import division
import numpy as np
import xarray as xr
from mpl_toolkits.basemap import Basemap
from scipy import interpolate
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import proplot as plot
from matplotlib.patches import Polygon
from analysis_functions import get_unique_line_labels
from plot_spatial_figs import interpolate_obs_dict


def get_lcc_regions_coords():
    """Return the name and location of low cloud cover regions"""
    regions_dict = {'Peru': {'lat':(-20.0, -10.0), 'lon': (360.0-90.0, 360.0-80.0)},
                'Namibia': {'lat':(-20.0, -10.0), 'lon': (0.0, 10.0)}, 
                'California': {'lat':(25.0, 35.0), 'lon': (360.0-130.0, 360.0-120.0)},
                #'California': {'lat':(20.0, 40.0), 'lon': (360.0-130.0, 360.0-120.0)},
                'Australia': {'lat':(-35.0, -25.0), 'lon': (100.0, 110.0)},
                'Canary': {'lat':(28.0, 38.0), 'lon': (360.0-25.0, 360.0-15.0)},
                #'North Pacific': {'lat':(40.0, 50.0), 'lon': (170.0, 180.0)},
                #'North Atlantic': {'lat':(50.0, 60.0), 'lon': (360.0-45.0, 360.0-35.0)},
                #'China': {'lat':(20.0, 30.0), 'lon': (105.0, 120)}, 
                }
    return regions_dict


def get_poly_dict_from_reg_dict(regions_dict, keys=None):
    if keys is None:
        i_keys = regions_dict.keys()
    else:
        i_keys = keys

    def poly_coord(x, y):
        xx = [x[0], x[1], x[1], x[0]]
        yy = [y[0], y[0], y[1], y[1]]
        return xx, yy

    poly_dict = {}
    for key in i_keys:
        lats, lons = poly_coord(regions_dict[key]['lat'], regions_dict[key]['lon'])
        poly_dict[key] = {'lat':lats, 'lon':lons}
        
    return poly_dict


def get_regional_mean_for_ds(ds, range_dict, var_name):
    """range_dict should be like {'lat':(-20.0, -10.0), 'lon': (0.0, 10.0)} """

    l_lat = np.logical_and(ds.lat >= range_dict['lat'][0], 
                           ds.lat <= range_dict['lat'][1])
    l_lon = np.logical_and(ds.lon >= range_dict['lon'][0], 
                           ds.lon <= range_dict['lon'][1])
    var = ds[var_name].where(l_lat, drop=True).where(l_lon, drop=True)
    
    # calculate regional mean
    coslat = np.cos(np.deg2rad(var.lat))
    lat_dim = var.dims.index('lat')
    ## Time is averaged as well...
    var_mean = np.average(var.mean(('time', 'lon')), axis=lat_dim-1, weights=coslat)

    return var_mean


def get_regional_mean_for_dt(dt, range_dict):
    """range_dict should be like {'lat':(-20.0, -10.0), 'lon': (0.0, 10.0)} """

    l_lat = np.logical_and(dt.lat >= range_dict['lat'][0], 
                           dt.lat <= range_dict['lat'][1])
    l_lon = np.logical_and(dt.lon >= range_dict['lon'][0], 
                           dt.lon <= range_dict['lon'][1])
    var = dt.where(l_lat, drop=True).where(l_lon, drop=True)
    
    # calculate regional mean
    coslat = np.cos(np.deg2rad(var.lat))
    lat_dim = var.dims.index('lat')
    ## Time is averaged as well...
    try:
        var_mean = np.average(var.mean(('time', 'lon')), axis=lat_dim-1, weights=coslat)
    except:
        var_mean = np.average(var.mean(('lon')), axis=lat_dim, weights=coslat)
    return var_mean


def calc_2d_var_change_over_region(ds_arr, regions_dict, var_name, order=1):
    """ Return the regional mean variable change.
        Change is calculated as ds_arr[0] - ds_arr[1] """

    var_mean_region_dict = {}
    for region, range_dict in regions_dict.items():
        var_mean = []
        for ds in ds_arr:
            var_mean.append(get_regional_mean_for_ds(ds, range_dict, var_name))
        if order==1:
            var_mean_region_dict[region] = var_mean[0] - var_mean[1]
        if order==2:
            var_mean_region_dict[region] = var_mean[1] - var_mean[0]

    return var_mean_region_dict


def get_dict_values(dt_dict, keys):
    dt = []
    for key in keys:
        dt.append(dt_dict[key])
    return np.array(dt)


def draw_screen_poly(lats, lons, m, ax, edgecolor='red',
                facecolor=None, alpha=None, fill=False, linewidth=1):
    """Ref: https://stackoverflow.com/questions/12251189/how-to-draw-rectangles-on-a-basemap
    Example:
        poly_list = [{'lat':[a,b,c,d], 'lon':[a,b,c,d]}, {'lat':[a,b,c,d], 'lon':[a,b,c,d]}]
    """
    x, y = m(lons, lats)
    xy = zip(x,y)
    poly = Polygon(list(xy), edgecolor=edgecolor, facecolor=facecolor,
                alpha=alpha, linewidth=linewidth, fill=fill)
    ax.add_patch(poly)


def plot_map(ax, dt, cnlevels=np.arange(-60,65,5), 
        plot_type='contourf', color='gray', cmap='RdBu_r', extend='both',
        slat=-30, nlat=30, lat_line_interval=15, grid_label_fs=8,
        fill_land=False, poly_dict=None):

    lats = dt.lat.values
    lons = dt.lon.values
    lons_2d, lats_2d = np.meshgrid(lons, lats)

    m = Basemap(llcrnrlon=0, urcrnrlon=360, llcrnrlat=slat, urcrnrlat=nlat, ax=ax, resolution='c') # lon_0=180, #(projection='robin', lon_0=0., ax=ax, resolution='c')
    m.drawcoastlines()
    if fill_land:
        m.fillcontinents(color='black',lake_color='aqua')
    m.drawparallels(np.arange(slat, nlat+1, lat_line_interval), labels=[1,0,0,0], linewidth=0, fontsize=grid_label_fs)
    m.drawmeridians(np.arange(-180.,181.,60.), labels=[0,0,0,1], linewidth=0, fontsize=grid_label_fs)
    xi, yi = m(lons_2d, lats_2d)

    if plot_type == 'contourf':
        if cnlevels is None:
            cs = m.contourf(xi, yi, dt, cmap=cmap)
        else:
            cs = m.contourf(xi, yi, dt, cmap=cmap, levels=cnlevels, extend=extend)
    else:
        cs = m.contour(xi, yi, dt, levels=cnlevels, colors=color, linewidths=1)
        ax.clabel(cs, cs.levels[::30], inline=True, fmt="%.0f", fontsize=grid_label_fs)

    # Add polygon

    if poly_dict is not None:
        for key, ipoly in poly_dict.items():
            draw_screen_poly(ipoly['lat'], ipoly['lon'], m, ax=ax)

    return cs


def low_cloud_improvement_evaluation_with_map(ds_arr_low, exp_names_local, obs_flux_dict, figname):
    # plot sw cre
    varnm = 'toa_sw_cre'
    obs_dict1 = interpolate_obs_dict(obs_flux_dict, ds_arr_low[0].lat, ds_arr_low[0].lon)
    obs = obs_dict1[varnm].mean('time')

    swcre_diff1 = ds_arr_low[0][varnm].mean('time') - obs
    swcre_diff2 = ds_arr_low[1][varnm].mean('time') - obs

    regions_dict = get_lcc_regions_coords()
    poly_dict = get_poly_dict_from_reg_dict(regions_dict)

    plot.close()
    fig, axes = plot.subplots([[1,1], [2,2], [3,4]], axwidth=4.5, hratios=[0.8, 0.8, 1.5], share=0)  # 
    
    diff_cnlevels = np.arange(-60,61,5)

    slat = -40
    nlat = 40
    lat_line_interval = 20
    grid_label_fs = 10 #SMALL_SIZE
    for kk, (ax, dt, exp_nm) in enumerate(zip(axes[0:2], [swcre_diff1, swcre_diff2], exp_names_local)):
        cs = plot_map(ax, dt, cnlevels=diff_cnlevels, slat=slat, nlat=nlat, 
                lat_line_interval=lat_line_interval, grid_label_fs=grid_label_fs,
                poly_dict=poly_dict)
        ax.set_title('('+chr(97+kk)+') SW CRE bias ('+exp_nm+' - obs)') #, fontsize=BIGGER_SIZE)    
    fig.colorbar(cs, loc='r', rows=(1,2), ticks=diff_cnlevels[::6], label=r'$Wm^{-2}$', width='1em') #, labelsize=SMALL_SIZE)

    ##### ============ SW bias ============#
    ax = axes[2]
    # plot bars
    sw_cre_bias = []
    row_nms = []
    col_names = ['FD - obs', 'ALL - obs']
    for region, range_dict in regions_dict.items():
        var_mean = []
        for dt, exp_nm in zip([swcre_diff1, swcre_diff2], exp_names):
            var_mean.append(get_regional_mean_for_dt(dt, range_dict))
        sw_cre_bias.append(var_mean)
        row_nms.append(region)

    sw_cre_bias = pd.DataFrame(sw_cre_bias, columns=pd.Index(col_names, name='Bias'),
                                index=pd.Index(row_nms, name=''))
    
    obj = ax.bar(sw_cre_bias, cycle='accent')
    ax.legend(obj, ncol=1)
    xlim = [-0.5, 4.5]
    ax.set_xlim(xlim)

    #ax.set_ylim([np.min(sw_cre_bias).min()*1.1, np.max(sw_cre_bias).max()*1.1])
    if np.min(sw_cre_bias).min() > 0:
        ax.set_ylim([0, np.max(sw_cre_bias).max()*1.1])
    else:
        ax.set_ylim([np.min(sw_cre_bias).min(), np.max(sw_cre_bias).max()*1.1])
    
    ax.format(xlocator=1, ytickminor=False, xtickminor=False)
    ax.set_title('(c) SW CRE bias') #, fontsize=BIGGER_SIZE)
    ax.set_ylabel('CRE Bias (Wm$^{-2}$)')#, fontsize=MEDIUM_SIZE)
    #ax.tick_params(labelsize=SMALL_SIZE-2)
    ax.format(xrotation=-30)

    # =============== plot CRE and cloud amount relationship ===============
    swcre_dict = calc_2d_var_change_over_region(ds_arr_low, regions_dict, 'toa_sw_cre', order=2)
    lcc_dict = calc_2d_var_change_over_region(ds_arr_low, regions_dict, 'low_cld_amt', order=2)

    keys = regions_dict.keys()
    swcre = get_dict_values(swcre_dict, keys)
    lcc = get_dict_values(lcc_dict, keys)

    ax = axes[3]
    lines = []
    for key in keys:
        l = ax.scatter(lcc_dict[key], swcre_dict[key], s=30, label=key, clip_on=False)
        lines.append(l)

    ax.set_xlabel('LCC changes (%)') #, fontsize=MEDIUM_SIZE)
    ax.set_ylabel('SW CRE changes (Wm$^{-2}$)') #, fontsize=MEDIUM_SIZE)
    xlim = [min(lcc), max(lcc)*1.2]
    #xlim = [min(lcc)*1.1, max(lcc)*1.1]
    #xlim = [0, max(lcc)*1.2]
    ax.set_xlim(xlim)
    ax.set_title('(d) Changes of SW CRE and LCC') #, fontsize=BIGGER_SIZE)
    
    new_lines, new_labels = get_unique_line_labels(lines)
    ax.legend(new_lines, new_labels, ncol=1)

    slope, intercept, r_value, p_value, std_err = stats.linregress(lcc, swcre)
    x2 = np.linspace(min(xlim), max(xlim), 100)
    y2 = slope * x2 + intercept
    ax.plot(x2, y2, 'k-', linewidth=1)

    r_str = r'$R$ = ' +"%.3f" % (r_value)
    ax.text(5, -22, r_str, fontsize=10)

    axes.format(grid=False)

    fig.tight_layout()
    fig.savefig(figname, bbox_inches='tight', pad_inches=0.05, transparent=False)


def low_cloud_improvement_evaluation(ds_arr_low, exp_names_local, obs_flux_dict, figname):
    # plot sw cre
    varnm = 'toa_sw_cre'
    obs_dict1 = interpolate_obs_dict(obs_flux_dict, ds_arr_low[0].lat, ds_arr_low[0].lon)
    obs = obs_dict1[varnm].mean('time')

    swcre_diff1 = ds_arr_low[0][varnm].mean('time') - obs
    swcre_diff2 = ds_arr_low[1][varnm].mean('time') - obs

    regions_dict = get_lcc_regions_coords()
    poly_dict = get_poly_dict_from_reg_dict(regions_dict)

    plot.close()
    fig, axes = plot.subplots(ncols=2, aspect=1.2, share=0) #hratios=[1.2, 1.2, 2, 1.5], axwidth=3,

    ##### ============ SW bias ============#
    ax = axes[0]
    # plot bars
    sw_cre_bias = []
    row_nms = []
    col_names = ['FD - obs', 'ALL - obs']
    for region, range_dict in regions_dict.items():
        var_mean = []
        for dt, exp_nm in zip([swcre_diff1, swcre_diff2], exp_names):
            var_mean.append(get_regional_mean_for_dt(dt, range_dict))
        sw_cre_bias.append(var_mean)
        row_nms.append(region)

    sw_cre_bias = pd.DataFrame(sw_cre_bias, columns=pd.Index(col_names, name='Bias'),
                                index=pd.Index(row_nms, name=''))

    obj = ax.bar(sw_cre_bias, cycle='accent')
    ax.legend(obj, ncol=1)
    xlim = [-0.5, 4.5]
    ax.set_xlim(xlim)

    if np.min(sw_cre_bias).min() > 0:
        ax.set_ylim([0, np.max(sw_cre_bias).max()*1.1])
    else:
        ax.set_ylim([np.min(sw_cre_bias).min(), np.max(sw_cre_bias).max()*1.1])
    
    ax.format(xlocator=1, ytickminor=False, xtickminor=False)
    ax.set_title('(a) SW CRE bias', loc='left')
    ax.set_ylabel('CRE Bias (Wm$^{-2}$)')
    ax.format(xrotation=-30)

    # =============== plot CRE and cloud amount relationship ===============
    swcre_dict = calc_2d_var_change_over_region(ds_arr_low, regions_dict, 'toa_sw_cre', order=2)
    lcc_dict = calc_2d_var_change_over_region(ds_arr_low, regions_dict, 'low_cld_amt', order=2)

    keys = regions_dict.keys()
    swcre = get_dict_values(swcre_dict, keys)
    lcc = get_dict_values(lcc_dict, keys)

    ax = axes[1]
    lines = []
    for key in keys:
        l = ax.scatter(lcc_dict[key], swcre_dict[key], s=30, label=key, clip_on=False)
        lines.append(l)

    ax.set_xlabel('Low cloud amount changes (%)')
    ax.set_ylabel('SW CRE changes (Wm$^{-2}$)')
    xlim = [min(lcc), max(lcc)*1.2]
    ax.set_xlim(xlim)
    ax.set_title('(b)', loc='left')
    
    new_lines, new_labels = get_unique_line_labels(lines)
    ax.legend(new_lines, new_labels, ncol=1)

    slope, intercept, r_value, p_value, std_err = stats.linregress(lcc, swcre)
    x2 = np.linspace(min(xlim), max(xlim), 100)
    y2 = slope * x2 + intercept
    ax.plot(x2, y2, 'k-', linewidth=1)

    r_str = r'$R$ = ' +"%.3f" % (r_value)
    ax.text(5, -22, r_str, fontsize=10)

    axes.format(grid=False)

    fig.tight_layout()
    
    fig.savefig(figname, bbox_inches='tight', pad_inches=0.05, transparent=False)
    #fig.savefig(figname.replace('.pdf', '.png'), bbox_inches='tight', pad_inches=0.05, transparent=False)
 