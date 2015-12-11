import numpy as np
from astropy.table import Table

def find_nearest(reference,values):
    i = np.zeros(len(values))
    for m,value in enumerate(values):
        i[m] = (np.abs(reference-value)).argmin()
    return i.astype(int)


def sort_data(D):

    D_i = np.arange(len(D))
    order = np.argsort(D)
    D_sorted = D[order]
    D_i_sorted = D_i[order]
    cumfrac = np.linspace(0,1,len(D))
    
    D_table = Table(np.array([D_i_sorted,D_sorted,cumfrac]).T,names=('index','fv','cumfrac'))
    reorder = np.argsort(D_table['index'])
    D_table = D_table[reorder]
    
    for f in np.unique(D_table['fv']):
        f_select = D_table['fv'] == f
        D_table['cumfrac'][f_select] = np.mean(D_table['cumfrac'][f_select])
    
    return D_table


def debias(data,vbins,zbins,question,answer):
    
    fraction_column = question + '_' + answer + '_weighted_fraction'
    data_column = data[fraction_column]
    debiased_column = np.zeros(len(data_column))

    for v in np.unique(vbins):
        select_v = vbins == v
        zbins_v = zbins[select_v]
        
        data_v0 = data_column[(select_v) & (zbins == 1)]
        v0_table = sort_data(data_v0)

        for z in np.unique(zbins_v):
            select_z = zbins == z
    
            data_vz = data_column[(select_v) & (select_z)]
            vz_table = sort_data(data_vz)
    
            debiased_i = find_nearest(v0_table['cumfrac'],vz_table['cumfrac'])
            debiased_fractions = v0_table['fv'][debiased_i]
            
            debiased_column[(select_v) & (select_z)] = debiased_fractions
    
    debiased_column[data_column == 0] = 0 # Don't 'debias up' 0s.
    debiased_column[data_column == 1] = 1 # Don't 'debias down' the 1s.
    
    return debiased_column