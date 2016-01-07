import numpy as np
import time
from scipy.optimize import minimize,curve_fit
from astropy.table import Table
import matplotlib.pyplot as plt

def make_fit_setup(function_dictionary,key):
    fit_setup = {}
    fit_setup['func'] = function_dictionary['func'][key]
    fit_setup['bounds'] = function_dictionary['bounds'][key]
    fit_setup['p0'] = function_dictionary['p0'][key]
    fit_setup['inverse'] = function_dictionary['i_func'][key]
    return fit_setup


def get_best_function(data,vbins,zbins,function_dictionary
                      ,question,answer,min_log_fv):
  
    # Find suitable p0 values from fits to the overall data:
    fv_all = np.sort(data[question + '_' + answer + '_weighted_fraction'])
    fv_nonzero = fv_all != 0
    cf = np.linspace(0,1,len(fv_all))
    x,y = [np.log10(fv_all[fv_nonzero]),cf[fv_nonzero]]
    
    x_fit = np.log10(np.linspace(10**(min_log_fv), 1, 100))
    indices = np.searchsorted(x,x_fit)
    y_fit = y[indices.clip(0, len(y)-1)]
    
    chisq_tot = np.zeros(len(function_dictionary['func'].keys()))
    k_tot = np.zeros(len(function_dictionary['func'].keys()))
    c_tot = np.zeros(len(function_dictionary['func'].keys()))
    
    for n,key in enumerate(function_dictionary['func'].keys()):
        # Overall data fitting:
        fit_setup = make_fit_setup(function_dictionary,key)
        func = fit_setup['func']
        p0 = fit_setup['p0']
        bounds = fit_setup['bounds']
        
        res =  minimize(chisq_fun, p0,
                        args=(func,x_fit,y_fit),
                        bounds=bounds,method='SLSQP')
        
        function_dictionary['p0'][key] = res.x
        
        if res.success == False:
            print('Failed to minimise total dataset')
            popt,pcov = curve_fit(func,x_fit,y_fit,maxfev=10**5) # unbounded
            res =  minimize(chisq_fun, popt,
                        args=(func,x_fit,y_fit),
                        bounds=bounds,method='SLSQP')
            if res.success == False:
                print('Still failed to minimise!')    
        
        fit_vbin_results = fit_vbin_function(data,vbins,zbins,fit_setup,
                                             question,answer,min_log_fv,clip=None)

        finite_chisq = np.isfinite(fit_vbin_results['chi2nu'])
        # Deal with chisq nans here.
        chisq = np.sum(fit_vbin_results['chi2nu'][finite_chisq])/(np.sum(finite_chisq))
        k = np.mean(fit_vbin_results['k'])
        c = np.mean(fit_vbin_results['c'])
        
        chisq_tot[n] = chisq
        k_tot[n] = k
        c_tot[n] = c
        
        print('chisq({}) = {}'.format(function_dictionary['label'][key],chisq))
    
    n = np.argmin(chisq_tot)
    keys = [key for key in function_dictionary['func'].keys()]
    key = keys[n]
    fit_setup = make_fit_setup(function_dictionary,key)
       
    return fit_setup 


def get_fit_setup(fit_setup):

    func = fit_setup['func']
    p0 = fit_setup['p0']
    bounds = fit_setup['bounds']
    
    return func, p0, bounds
  
  
def chisq_fun(p, f, x, y):
    return ((f(x, *p) - y)**2).sum()


def fit_vbin_function(data, vbins, zbins, fit_setup,
                      question,answer,min_log_fv,
                      kc_fit_results=None,
                      even_sampling=True,clip=2):
    
    start_time = time.time()
    
    min_fv = 10**(min_log_fv)
    
    redshift = data['REDSHIFT_1']
    fv = question + '_' + answer +'_weighted_fraction'
    
    if kc_fit_results is not None:
        kcfunc, kparams, cparams, lparams,kclabel = kc_fit_results
    
    # Set up the list to write the parameters in to:
    param_data = []
    
    max_z_bins_to_plot = 5
    
    bounds = fit_setup['bounds']
    p0 = fit_setup['p0']
    func = fit_setup['func']
    
    colours = ['b','g','k','r']
    xg = np.linspace(-2,0,100)
    
    # Loop over Voronoi magnitude-size bins
    for v in np.unique(vbins):
        vselect = vbins == v
        data_v = data[vselect]
        zbins_v = zbins[vselect]

        z_bins_unique = np.unique(zbins_v)

        for z in z_bins_unique:
            data_z = data_v[zbins_v == z]
            n = len(data_z)
            
            D = data_z[[fv]]
            D.sort(fv)
            D['cumfrac'] = np.linspace(0, 1, n)
            D = D[D[fv] > min_fv]
            D['log10fv'] = np.log10(D[fv])
            if even_sampling:
                D_fit_log10fv = np.log10(np.linspace(10**(min_log_fv), 1, 100))
                D = D[(D['log10fv'] > min_log_fv)] #& (D['log10fv'] < max_log_fv)]
                indices = np.searchsorted(D['log10fv'], D_fit_log10fv)
                D_fit = D[indices.clip(0, len(D)-1)]
            else:
                D_fit = D[D['log10fv'] > min_log_fv]

            res = minimize(chisq_fun, p0,
                           args=(func,
                                 D_fit['log10fv'].astype(np.float64),
                                 D_fit['cumfrac'].astype(np.float64)),
                           bounds=bounds, method='SLSQP')
            
            p = res.x
            chi2nu = res.fun / (n - len(p))
            
            if res.success == False:
               print('Fit not found for z={},v={}'.format(z,v))
                
            means = [data_z['PETROMAG_MR'].mean(),
                     np.log10(data_z['PETROR50_R_KPC']).mean(),
                     data_z['REDSHIFT_1'].mean()]

            if len(p) < 2:
                p = np.array([p[0], 10])

            param_data.append([v,z] + means + p[:2].tolist() + # Maybe change output table here
                              [chi2nu])
            
    fit_vbin_results = Table(rows=param_data,
                             names=('vbin','zbin', 'Mr',
                                    'R50', 'redshift', 'k', 'c', 'chi2nu'))
    
    print('All bins fitted! {}s in total'.format(time.time()-start_time))
    # remove 'odd' fits.
    if clip != None:
        k_values = fit_vbin_results['k']
        k_mean = np.mean(k_values)
        k_std = np.std(k_values)
        k_range = [k_mean-clip*k_std,k_mean+clip*k_std]
        
        c_values = fit_vbin_results['c']
        c_mean = np.mean(c_values)
        c_std = np.std(c_values)
        c_range = [c_mean-clip*c_std,c_mean+clip*c_std]
        
        select = ((k_values > k_range[0]) & (k_values < k_range[1]) 
                  & (c_values > c_range[0]) & (c_values < c_range[1]))

        fit_vbin_results = fit_vbin_results[select]
    
    return fit_vbin_results


def fit_mrz(d, f_k, f_c, clip=None,plot=True):
    # Fit a linear function of M, R and z to k and c
    
    dout = d.copy()
    dout['kf'] = np.zeros(len(d))
    dout['cf'] = np.zeros(len(d))
    
    kparams = []
    cparams = []
    # Set limits of the functions here.
    kmin = d['k'].min() 
    kmax = d['k'].max() 
    cmin = d['c'].min()
    cmax = d['c'].max()

    # Loop over GZ morphologies
    x = np.array([d[c] for c in ['Mr', 'R50', 'redshift']], np.float64)
    k = d['k'].astype(np.float64)
    c = d['c'].astype(np.float64)

    kp, kc = curve_fit(f_k, x, k, maxfev=100000)
    cp, cc = curve_fit(f_c, x, c, maxfev=100000)
        
    kres = f_k(x, *kp) - k
    knormres = normalise(kres)

    cres = f_c(x, *cp) - c
    cnormres = normalise(cres)
    
    bins = np.linspace(-3,3,15)
    
    if plot == True:
    
        plt.figure(figsize=(15,5))
        plt.subplot(1,2,1)
        plt.hist(knormres,color='r',alpha=0.5,bins=bins)
        plt.xlabel('residual k')
        plt.ylabel('N')
    
        plt.subplot(1,2,2)
        plt.hist(cnormres,color='b',alpha=0.5,bins=bins)
        plt.xlabel('residual c')
        plt.ylabel('N')
        
    if clip != None:
        
        clipped = ((np.absolute(knormres) < clip) & (np.absolute(cnormres) < clip))# 'clip' sigma clipping
        kp, kc = curve_fit(f_k, ((x.T)[clipped]).T, k[clipped], maxfev=100000)
        cp, cc = curve_fit(f_c, ((x.T)[clipped]).T, c[clipped], maxfev=100000)
        
        if plot == True:
            plt.subplot(1,2,1)
            plt.hist(knormres[clipped],color='r',bins=bins)
            plt.xlabel('residual k')
            plt.ylabel('N')
        
            plt.subplot(1,2,2)
            plt.hist(cnormres[clipped],color='b',bins=bins)
            plt.xlabel('residual c')
            plt.ylabel('N')
        
    dout['kf'] = f_k(x, *kp)
    dout['cf'] = f_c(x, *cp)
        
    kparams.append(kp)
    cparams.append(cp)

    return kparams, cparams,  dout, kmin, kmax, cmin, cmax


def normalise(x):
    return (x - x.mean())/x.std()
  
  
def normalise_tot(x,mean,std):
    return (x - mean)/std


def get_term(constant,var,t='linear'):
    
    if t == 'log':
        term = constant*np.log10(var)
    elif t == 'linear':
        term = constant*var
    else:
        term = constant*(10**(var))
    return term
    

def get_func(M_dependence,R_dependence,z_dependence):
    
    def kcfunc(x,A0,AM,AR,Az):
        M_term = get_term(AM,x[0],M_dependence)
        R_term = get_term(AR,x[1],R_dependence)
        z_term = get_term(Az,x[2],z_dependence)
        return A0 + M_term + R_term + z_term
    
    return kcfunc
  
  
def get_kc_functions(fit_vbin_results):

    c_residuals = np.zeros(3**3)
    k_residuals = np.zeros(3**3)
    i = 0
    M_dependences = []
    R_dependences = []
    z_dependences = []

    finite_select = (np.isfinite(fit_vbin_results['k'])) & (np.isfinite(fit_vbin_results['c']))
    fit_vbin_results_finite = fit_vbin_results[finite_select]
    
    for M_dependence in ['log','linear','exp']:
        for R_dependence in ['log','linear','exp']:
            for z_dependence in ['log','linear','exp']:

                '''
    for M_dependence in ['linear']:
        for R_dependence in ['linear']:
            for z_dependence in ['linear']:
                '''
                kcfunc = get_func(M_dependence,R_dependence,z_dependence)
                kparams, cparams,dout, kmin, kmax, cmin, cmax = fit_mrz(fit_vbin_results_finite, kcfunc, kcfunc,clip=None,plot=False)
               
                k_fit_residuals = (dout['kf']-dout['k'])**2
                k_fit_residuals = k_fit_residuals[np.isfinite(k_fit_residuals)]

                c_fit_residuals = (dout['cf']-dout['c'])**2
                c_fit_residuals = c_fit_residuals[np.isfinite(c_fit_residuals)]

                k_residuals[i] = np.mean(k_fit_residuals)
                c_residuals[i] = np.mean(c_fit_residuals)
                i = i+1
            
                M_dependences.append(M_dependence)
                R_dependences.append(R_dependence)
                z_dependences.append(z_dependence)

    k_residuals[np.isfinite(k_residuals) == False] = 10**8
    c_residuals[np.isfinite(c_residuals) == False] = 10**8

    best_k = np.argmin(k_residuals)
    best_c = np.argmin(c_residuals)

    best_M_k = M_dependences[best_k]
    best_R_k = R_dependences[best_k]
    best_z_k = z_dependences[best_k]
    best_M_c = M_dependences[best_c]
    best_R_c = R_dependences[best_c]
    best_z_c = z_dependences[best_c]

    k_func = get_func(best_M_k,best_R_k,best_z_k)
    c_func = get_func(best_M_c,best_R_c,best_z_c)
    
    return k_func,c_func
  
  
def function_inversion(value,func,k,kb,c,cb):
    # for use when function has no mathematical inverse
    xg = np.log10(np.linspace(0.01,1,100))
    low_z_values = func(xg,kb,cb,lb)
    high_z_value = func(value,k,c,l)
    i = (np.abs(low_z_values-high_z_value)).argmin()
    x = xg[i]
    return x
  
  
def debias(data, z_base, k_func,c_func, kparams, cparams,
           question,answer,kmin,kmax,cmin,cmax,fit_setup):
    # Debias the dataset
    
    fv_col = question + '_' + answer + '_weighted_fraction'
    # Each galaxy gets a function fit to its M,R and z parameters, which are scaled
    # to the equivalent M and r functions at low z.
    
    fv = data[fv_col]
    debiased = np.zeros(len(fv))
    fv_nonzero = fv > 0
    log10fv = np.log10(np.asarray(fv[fv_nonzero]))
    func, _, _ = get_fit_setup(fit_setup)
    i_func = fit_setup['inverse']
    bounds = fit_setup['bounds']
    #------
    d  = data[fv_nonzero]
        
    x = np.array([d['PETROMAG_MR'],
                 np.log10(d['PETROR50_R_KPC']),
                 d['REDSHIFT_1']], np.float64)
    xb  = x.copy()
    xb[-1] = z_base
        
    k = k_func(x, *kparams[0])
    c = c_func(x, *cparams[0])
    
    k[k < kmin] = kmin
    k[k > kmax] = kmax
    c[c < cmin] = cmin
    c[c > cmax] = cmax

    #create version of x with all redshifts at z_base
    kb = k_func(xb, *kparams[0])
    cb = c_func(xb, *cparams[0])
        
    kb[kb < kmin] = kmin
    kb[kb > kmax] = kmax
    cb[cb < cmin] = cmin
    cb[cb > cmax] = cmax
        
    cumfrac = func(log10fv, k, c)
    log10fv_debiased = i_func(cumfrac, kb, cb)
        
    fv_debiased = 10**(log10fv_debiased)
    debiased[fv_nonzero] = fv_debiased

    return debiased


def debias_by_fit(data,full_data,vbins,zbins,zbins_coarse,question,
                  answer,function_dictionary,min_log_fv,coarse=False):

    if coarse == True: # can choose whether to coarsely bin here.
        zbins = zbins_coarse.copy()
    
    fit_setup = get_best_function(data,vbins,zbins_coarse,function_dictionary,
                                  question,answer,min_log_fv)
    
    fit_vbin_results = fit_vbin_function(data, vbins, zbins, fit_setup,
                                         question,answer,min_log_fv)
    
    k_func,c_func = get_kc_functions(fit_vbin_results)
    
    kparams, cparams,dout, kmin, kmax, cmin, cmax = fit_mrz(fit_vbin_results,
                                                            k_func,c_func,
                                                            clip=2,plot=False)
    # clip results here.
    
    debiased_fit = debias(full_data,0.03, k_func,c_func, kparams, cparams,
                          question,answer,kmin,kmax,cmin,cmax,fit_setup)

    # Debias ALL of the data 
    
    return debiased_fit,dout,fit_setup,zbins,fit_vbin_results
