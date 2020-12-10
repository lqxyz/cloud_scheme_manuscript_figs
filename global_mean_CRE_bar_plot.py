
from __future__ import print_function
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import proplot as plot
from zonal_cre_from_cmip_isca import get_global_mean


def global_mean_CRE_bar_plot(ds_arr, exp_names, obs_toa_cre, figname):
    text_size = 7.5

    obs_nm41 = 'CERES_EBAF Ed4.1'
    obs_dict41 = obs_toa_cre[obs_nm41]

    obs_nm28 = 'ERA-Interim' #'CERES_EBAF Ed2.8'
    obs_dict28 = obs_toa_cre[obs_nm28]

    # ================================================ #
    # plot histogram of CREs
    # ================================================ #
    # recover dict from 0-d numpy array
    cmip5_gm_cre = np.load('cmip5_historical_gm_cre_r128x64.npy', allow_pickle=True).item()

    cmip5_model_nms = sorted(cmip5_gm_cre.keys(), key=str.casefold)
    cmip5_model_nms.remove('MME') 
    #cmip5_model_nms.append('MME') 

    nModels = len(cmip5_gm_cre) - 1
    #x = np.arange(0,  nModels+1, 1)
    nIsca = 3

    x = np.arange(1,  nModels+len(exp_names)+1, 1)

    pos_x1 = 1
    pos_x2 = 8.5
    pos_x3 = 16

    var_names = ['toa_sw_cre', 'toa_lw_cre', 'toa_net_cre',]
    var_titles = ['TOA SW CRE', 'TOA LW CRE', 'TOA total CRE']

    #plot.rc.titleloc = 'uc'
    plot.rc.margin = 0.05

    plot.close('all')
    fig, axes = plot.subplots(nrows=3, ncols=1, aspect=(5, 1), axwidth=6, sharex=False) #, hratios=(0.3, 0.3, 0.3))

    legend_labels = []
    # ======================== SWCRE ======================== #
    var_nm = 'toa_sw_cre'
    var_title = '(a) TOA SW CRE'
    cres = []
    for k, cmip_nm in enumerate(cmip5_model_nms):
        cres.append(cmip5_gm_cre[cmip_nm][var_nm])
        legend_labels.append(str(k+1)+' - '+ cmip_nm)

    for k, isca_nm in enumerate(exp_names):
        cres.append(get_global_mean(ds_arr[k][var_nm].mean('time')))
        legend_labels.append(str(nModels+k+1)+' - '+ isca_nm)

    pos_sw_cres = [-cre-35 for cre in cres]

    ax = axes[0]
    dummies = [axes[0].plot([], [], ls='')[0] for l in legend_labels]

    obj = ax.bar(x, pos_sw_cres, cycle='darkgray', edgecolor='black') 
    # cycle='Reds', colorbar='ul', colorbar_kw={'frameon': False}
    ax.format(xlocator=1, ytickminor=True, yminorlocator=1, xminorlocator=1,
        title=var_title, suptitle='')
    # set different color for Isca exps
    for i in range(nModels, nModels+nIsca):
        obj[i].set_color('navajowhite')
        obj[i].set_edgecolor('black')
    for i in range(nModels+nIsca, nModels+nIsca*2):
        obj[i].set_color('powderblue')
        obj[i].set_edgecolor('black')

    locs = np.arange(0, 26, 5)
    new = [-int(loc+35) for loc in locs]
    ax.set_yticks(locs)
    ax.set_yticklabels(new)
    xlims = [0.5, max(x)+0.5]

    MME = cmip5_gm_cre['MME'][var_nm]
    ax.plot(xlims, [-MME-35, -MME-35], 'C1:')
    
    locs = ax.get_yticks()
    ax.text(pos_x3,  max(locs)-2, 'Multimodel Mean ('+ '%.1f' % MME + ')', 
            color='C1', fontsize=text_size)

    obs_cre = get_global_mean(obs_dict41[var_nm]).mean('time')
    ax.plot(xlims, [-obs_cre-35, -obs_cre-35], 'C2--')
    ax.text(pos_x1,  max(locs)-2, obs_nm41.replace('_','-')+' ('+ '%.1f' % obs_cre + ')',
            color='C2', fontsize=text_size)
    
    obs_cre28 = get_global_mean(obs_dict28[var_nm]).mean('time')
    ax.plot(xlims, [-obs_cre28-35, -obs_cre28-35], 'C3--')
    ax.text(pos_x2,  max(locs)-2, obs_nm28.replace('_','-')+' ('+ '%.1f' % obs_cre28 + ')',
            color='C3', fontsize=text_size)

    ax.set_xlim(xlims)
    ax.set_ylim([0, max(locs)+3])

    # ======================== LWCRE ======================== #
    var_nm = 'toa_lw_cre'
    var_title = '(b) TOA LW CRE'
    cres = []
    for cmip_nm in cmip5_model_nms:
        cres.append(cmip5_gm_cre[cmip_nm][var_nm])
    
    for k, isca_nm in enumerate(exp_names):
        cres.append(get_global_mean(ds_arr[k][var_nm].mean('time')))

    pos_lw_cres = [cre-10 for cre in cres]

    ax = axes[1]
    obj = ax.bar(x, pos_lw_cres, cycle='darkgray', edgecolor='black')
    ax.format(xlocator=1, ytickminor=True, yminorlocator=1, xminorlocator=1,
        title=var_title)
    
    # set different color for Isca exps
    for i in range(nModels, nModels+nIsca):
        obj[i].set_color('navajowhite')
        obj[i].set_edgecolor('black')
    for i in range(nModels+nIsca, nModels+nIsca*2):
        obj[i].set_color('powderblue')
        obj[i].set_edgecolor('black')

    xlims = [0.5, max(x)+0.5]
    ax.set_xlim(xlims)

    locs = np.arange(0, 31, 5)
    new = [int(loc+10) for loc in locs]
    ax.set_yticks(locs)
    ax.set_yticklabels(new)
    
    MME = cmip5_gm_cre['MME'][var_nm]
    ax.plot(xlims, [MME-10, MME-10], 'C1:')
    locs = ax.get_yticks()
    ax.text(pos_x3,  max(locs)-5, 'Multimodel Mean ('+ '%.1f' % MME + ')',
            color='C1', fontsize=text_size)
    obs_cre = get_global_mean(obs_dict41[var_nm]).mean('time')
    ax.plot(xlims, [obs_cre-10, obs_cre-10], 'C2--')
    ax.text(pos_x1,  max(locs)-5, obs_nm41.replace('_','-')+' ('+ '%.1f' % obs_cre + ')',
            color='C2', fontsize=text_size)

    obs_cre28 = get_global_mean(obs_dict28[var_nm]).mean('time')
    ax.plot(xlims, [obs_cre28-10, obs_cre28-10], 'C3--')
    ax.text(pos_x2,  max(locs)-5, obs_nm28.replace('_','-')+' ('+ '%.1f' % obs_cre28 + ')',
            color='C3', fontsize=text_size)
    
    ax.set_ylim([0, max(locs)])

    # ======================== NET CRE ======================== #
    var_nm = 'toa_net_cre'
    var_title = '(c) TOA Net CRE'
    cres = []
    for cmip_nm in cmip5_model_nms:
        cres.append(cmip5_gm_cre[cmip_nm][var_nm])
    
    for k, isca_nm in enumerate(exp_names):
        cres.append(get_global_mean(ds_arr[k][var_nm].mean('time')))

    pos_net_cres = [-cre-10 for cre in cres]

    ax = axes[2]
    obj = ax.bar(x, pos_net_cres, cycle='darkgray', #cycle='Reds', colorbar='ul',
        edgecolor='black', ) #colorbar_kw={'frameon': False}
    ax.format(xlocator=1, ytickminor=True, yminorlocator=1, xminorlocator=1,
        title=var_title)
    # set different color for Isca exps
    for i in range(nModels, nModels+nIsca):
        obj[i].set_color('navajowhite')
        obj[i].set_edgecolor('black')
    for i in range(nModels+nIsca, nModels+nIsca*2):
        obj[i].set_color('powderblue')
        obj[i].set_edgecolor('black')
    locs = np.arange(0, 26, 5)
    new = [-int(loc+10) for loc in locs]

    ax.set_yticks(locs)
    ax.set_yticklabels(new)

    xlims = [0.5, max(x)+0.5]
    MME = cmip5_gm_cre['MME'][var_nm]
    ax.plot(xlims, [-MME-10, -MME-10], 'C1:')
    
    locs = ax.get_yticks()
    ax.text(pos_x3,  max(locs)-3, 'Multimodel Mean ('+ '%.1f' % MME + ')', 
            color='C1', fontsize=text_size)
    obs_cre = get_global_mean(obs_dict41[var_nm]).mean('time')
    ax.plot(xlims, [-obs_cre-10, -obs_cre-10], 'C2--')
    ax.text(pos_x1,  max(locs)-3, obs_nm41.replace('_','-')+' ('+ '%.1f' % obs_cre + ')',
            color='C2', fontsize=text_size)
    obs_cre28 = get_global_mean(obs_dict28[var_nm]).mean('time')
    ax.plot(xlims, [-obs_cre28-10, -obs_cre28-10], 'C3--')
    ax.text(pos_x2,  max(locs)-3, obs_nm28.replace('_','-')+' ('+ '%.1f' % obs_cre28 + ')',
            color='C3', fontsize=text_size)

    ax.set_xlim(xlims)
    ax.set_ylim([0, max(locs)])

    axes.format(grid=False, ylabel=r'Flux (Wm$^{-2}$)')

    # ======================== Add dummy legends ======================== #
    plot.rc.update({"legend.handlelength": 0})
    fig.legend(dummies, legend_labels, ncols=1, label='', frame=False, loc='r')

    plot.rc.reset()
    fig.savefig(figname, bbox_inches='tight', pad_inches=0.1, transparent=False)
