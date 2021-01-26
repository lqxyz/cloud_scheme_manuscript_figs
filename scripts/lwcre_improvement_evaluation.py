"""
Refer to Bony et al (2004), On dynamic and thermal dynamic components of cloud changes, Clim Dyn.
"""
from __future__ import print_function
from __future__ import division
import numpy as np
import xarray as xr
from mpl_toolkits.basemap import Basemap
import sh
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import proplot as plot
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


def plot_omega500_map(ax, ds, coeff=24*3600/100, cnlevels=np.arange(-60,65,5),
        plot_type='contourf', color='gray', cmap='RdBu_r', extend='both',sel_p=500):

    dt = ds.omega.sel(pfull=sel_p).mean('time')*coeff
    lats = dt.lat.values
    lons = dt.lon.values
    lons_2d, lats_2d = np.meshgrid(lons, lats)
    m = Basemap(llcrnrlon=0, urcrnrlon=360, llcrnrlat=-30, urcrnrlat=30, ax=ax, resolution='c') # lon_0=180, #(projection='robin', lon_0=0., ax=ax, resolution='c')
    m.drawcoastlines()
    m.fillcontinents(color='black',lake_color='aqua')
    m.drawparallels(np.arange(-30.,31.,15.),   labels=[1,0,0,0], linewidth=0)
    m.drawmeridians(np.arange(-180.,181.,60.), labels=[0,0,0,1], linewidth=0)
    xi, yi = m(lons_2d, lats_2d)

    if plot_type == 'contourf':
        if cnlevels is None:
            cs = m.contourf(xi, yi, dt, cmap=cmap)
        else:
            cs = m.contourf(xi, yi, dt, cmap=cmap, levels=cnlevels, extend=extend)
    if plot_type == 'contour':
        cs = m.contour(xi, yi, dt, levels=cnlevels, colors=color, linewidths=1)
        ax.clabel(cs, cs.levels[::30], inline=True, fmt="%.0f")
    return cs

def cld_amt_and_lw_cre_composite_analysis_omega500(ds_lev, dt_arrs, labels, i_colors, titles, ylabels, fig_nm,
        nrows=2, ncols=2, bins=np.arange(-100,101,5), xtick_interval_num=4,
        time_axis=0, capsize=1):

    def plot_dt_arr(dt_arr, label, i_color='k'):
        for kk, dt in enumerate(dt_arr):
            if kk==0:
                ax = axes[1]
                ylabel = ylabels[0]
                title = titles[0]
            if kk==1 or kk==2:
                ax = axes[2]
                ylabel = ylabels[1]
                title = titles[1]
            if kk==3:
                ax = axes[3]
                ylabel = ylabels[2]
                title = titles[2]
    
            if time_axis is not None:
                dt_mean = np.nanmean(dt, axis=time_axis)
                dt_std = np.nanstd(dt, axis=time_axis)
            else:
                dt_mean = dt
                dt_std = np.zeros_like(dt) 
 
            if kk==0:
                marker='-'
            if kk==1:
                marker='-o'
            if kk==2:
                marker='-^'
            if kk==3:
                marker='-s'
            l = ax.plot(x_bins, dt_mean, i_color+marker, markersize=4, mfc='w', label=label)
            if kk==0:
                ax.errorbar(x_bins, dt_mean, color=i_color, yerr=dt_std, ecolor=i_color, capsize=capsize)
            if 'density' in ylabel.lower():
                ax.set_title('(b) '+title, loc='left')
            elif '%' in ylabel.lower() : 
                ax.set_title('(c) '+title, loc='left')
            else:
                ax.set_title('(d) '+title, loc='left')

            ax.set_xlim([min(bins), max(bins)])
            ax.set_xticks(bins[::xtick_interval_num])
            ax.set_ylabel(ylabel)
        return l
    

    x_bins = (bins[0:-1]+bins[1:])/2
    plot.close('all')
    fig, axes = plot.subplots([[1,1], [2,3], [4,0]], axwidth=5, hratios=[1.2, 2, 2], share=0)

    ax = axes[0]

    print('plot omega500...')
    cnlevels = np.arange(-60,65,5)
    plot_type = 'contourf'
    cs = plot_omega500_map(ax, ds_lev, cnlevels=cnlevels, plot_type=plot_type)
    if plot_type == 'contourf':
        ax.colorbar(cs, loc='r', ticks=cnlevels[::6], width='1em')
    ax.set_title('(a) $\omega_{500}$ (hPa day$^{-1}$)', loc='left')

    print('plot lines...')
    lines = []
    # plot cloud amount
    for dt_arr, i_color, label in zip(dt_arrs, i_colors, labels):
        l = plot_dt_arr(dt_arr, label, i_color=i_color)
        lines.extend(l)

    axes[1:].format(xlabel="$\omega_{500}$ (hPa day$^{-1}$)",
        xtickminor=False, grid=False)

    legend_elements = []
    for ds_nm, i_color in zip(labels, i_colors):
        #legend_elements.append(Patch(facecolor='C'+str(k), edgecolor='C'+str(k), label=ds_nm))
        legend_elements.append(Line2D([0], [0], linestyle='-', color=i_color, lw=1, label=ds_nm))

    # #legend_elements = []
    legend_elements.append(Line2D([0], [0], linestyle='-', marker='o', color='k', mfc='w', lw=1, label='Low cloud amount'))
    legend_elements.append(Line2D([0], [0], linestyle='-', marker='^', color='k', mfc='w', lw=1, label='High cloud amount'))
    legend_elements.append(Line2D([0], [0], linestyle='-', marker='s', color='k', mfc='w', lw=1, label='LW CRE'))

    axes[-1].legend(legend_elements, ncol=1, loc='r',  bbox_to_anchor=(5, 0.5))

    axes[1:].format(xlim=[-60, 50])

    fig.tight_layout() #w_pad=-10.5, h_pad=-10)
    fig.savefig(fig_nm, bbox_inches='tight', pad_inches=0.05, transparent=False)
    #fig.savefig(fig_nm.replace('.pdf', '.png'), bbox_inches='tight', pad_inches=0.05, transparent=False)
