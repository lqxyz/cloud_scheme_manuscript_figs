from __future__ import print_function
from __future__ import division
import numpy as np
from scipy import optimize
import inspect


def cld_frac_wrt_rh_linear(rh, a):
    cf = a * (rh-1) + 1
    cf[cf>1.0] = 1.0
    cf[cf<0.0] = 0.0
    return cf

def cld_frac_spookie(rh, rhc):
    cf = (rh-rhc) / (1-rhc)
    cf[cf>1.0] = 1.0 
    cf[cf<0.0] = 0.0 
    return cf

def fit_cld_profile_paras(trial_func, rh_t, cld_frac_t, fit_rhc=False,
        left_boundary=None, right_boundary=None,
        rhc_min=0.1, rhc_max=1-1e-5, **kwargs):
    
    #print('Trial func is: '+str(trial_func.__name__))
    levels = rh_t.level
    nlevs = len(levels)

    paras = []

    for idx in range(nlevs):
        xdata = rh_t[idx,:,:].values.flatten()/1e2
        ydata = cld_frac_t[idx,:,:].values.flatten()

        num_paras = len(inspect.getargspec(trial_func).args)-1
        if left_boundary is None:
            left_bds = np.squeeze(np.full((1, num_paras), -np.inf))
        else:
            left_bds = left_boundary
        if right_boundary is None:
            right_bds = np.squeeze(np.full((1, num_paras), np.inf))
        else:
            right_bds = right_boundary
    
        if fit_rhc:
            if num_paras>1:
                left_bds[-1] = rhc_min
                right_bds[-1] = rhc_max
                left_bds = tuple(left_bds)
                right_bds = tuple(right_bds)
            else:
                left_bds = rhc_min
                right_bds = rhc_max
        try:
            popt, pcov = optimize.curve_fit(trial_func, xdata, ydata, #p0=p0, 
                        bounds=[left_bds, right_bds], **kwargs)
        except:
            print('No paras are found!')
            popt = np.array([np.nan]*num_paras)
        paras.append(popt)

    return paras
