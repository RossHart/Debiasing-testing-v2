import numpy as np
from voronoi_2d_binning import voronoi_2d_binning
from sklearn.neighbors import NearestNeighbors
from astropy.table import Table
import matplotlib.pyplot as plt

def voronoi_binning(R50, Mr, n_rect_bins=500, n_per_voronoi_bin=5000,save=False):
    
    rect_bin_val, R50_bin_edges, Mr_bin_edges = np.histogram2d(R50, Mr, n_rect_bins)

    rect_bins_table = Table(data=[R50_bin_edges, Mr_bin_edges],
                            names=['R50_bin_edges', 'Mr_bin_edges'])
    rect_bins_table.meta['nrectbin'] = n_rect_bins # add value for number of 
    # bins to the table. 
    
    # Get bin centres + number of bins:
    R50_bin_centres = 0.5*(R50_bin_edges[:-1] + R50_bin_edges[1:])
    Mr_bin_centres = 0.5*(Mr_bin_edges[:-1] + Mr_bin_edges[1:]) 
    n_R50_bins = len(R50_bin_centres)
    n_Mr_bins = len(Mr_bin_centres)

    # Get ranges:
    R50_bins_min, Mr_bins_min = map(np.min, (R50_bin_centres, Mr_bin_centres))
    R50_bins_max, Mr_bins_max = map(np.max, (R50_bin_centres, Mr_bin_centres))
    R50_bins_range = R50_bins_max - R50_bins_min
    Mr_bins_range = Mr_bins_max - Mr_bins_min
    
    # 'Ravel' out the coordinate bins (.'. length=n_bin*n_bin)
    R50_bin_coords = R50_bin_centres.repeat(n_rect_bins).reshape(n_rect_bins, n_rect_bins).ravel()
    Mr_bin_coords = Mr_bin_centres.repeat(n_rect_bins).reshape(n_rect_bins, n_rect_bins).T.ravel()

    # Only keep bins that contain a galaxy:
    signal = rect_bin_val.ravel() # signal=number of gals.
    ok_bin = (signal > 0).nonzero()[0]
    signal = signal[ok_bin]

    # Normalise x + y to be between 0 and 1:
    x = (R50_bin_coords[ok_bin] - R50_bins_min) / R50_bins_range
    y = (Mr_bin_coords[ok_bin] - Mr_bins_min) / Mr_bins_range

    # Voronoi_2d_binning aims for a target S/N
    noise = np.sqrt(signal)
    targetSN = np.sqrt(n_per_voronoi_bin)

    output = voronoi_2d_binning(x, y, signal, noise, targetSN, plot=0, quiet=1, wvt=True)
    binNum, xNode, yNode, xBar, yBar, sn, nPixels, scale = output

    vbin = np.unique(binNum)
    count = (sn**2).astype(np.int) # N_gals for each voronoi bin.
    R50_vbin_mean = xBar * R50_bins_range + R50_bins_min
    Mr_vbin_mean = yBar * Mr_bins_range + Mr_bins_min
    
    vbins_table = Table(data=[vbin, R50_vbin_mean, Mr_vbin_mean,
                              count, nPixels],
                        names=['vbin', 'R50', 'Mr', 
                               'count_gals', 'count_rect_bins'])
    vbins_table.meta['nrectbin'] = n_rect_bins
    vbins_table.meta['nperbin'] = n_per_voronoi_bin

    # Populate elements of the rectangular grid with
    # the voronoi bin indices and counts
    rect_bin_voronoi_bin = np.zeros(np.product(rect_bin_val.shape), np.int) - 1
    rect_bin_voronoi_bin[ok_bin] = binNum
    rect_bin_count = np.zeros_like(rect_bin_voronoi_bin)
    rect_bin_count[ok_bin] = count
    
    rect_vbins_table = Table(data=[R50_bin_coords, Mr_bin_coords,
                             rect_bin_voronoi_bin],
                             names=['R50', 'Mr', 'vbin'])
    rect_bins_table.meta['nrectbin'] = n_rect_bins
    rect_bins_table.meta['nperbin'] = n_per_voronoi_bin
    
    if save == True:
        rect_bins_table.write(save_directory + 'rect_bins_table.fits', overwrite=True)
        vbins_table.write(save_directory + 'vbins_table.fits', overwrite=True)
        rect_vbins_table.write(save_directory + 'rect_vbins_table.fits', overwrite=True)
        
    plt.hist(vbins_table['count_gals'],histtype='stepfilled',color='b',alpha=0.5,linewidth=0)
    ylims = plt.gca().get_ylim()
    plt.vlines(n_per_voronoi_bin,ylims[0],ylims[1],color='k',linewidth=3,linestyle='dotted')
    plt.ylabel('$N_{bin}$')
    plt.xlabel('$N_{gal}$')
    
    # rect_bins_table: contains all of the bin edges (len=N_bins)
    # rect_vbins_table: has bin centre values + assigned v-bin (len=N_bins**2)
    # vbins_table: for each bin, contains the number of gals, Mr+R50 mean
    # values + the number of rectangular bins it is made up of (len=N_v-bins)
    return rect_bins_table, vbins_table, rect_vbins_table, Mr_bins_min, Mr_bins_range, R50_bins_min, R50_bins_range
  
  
def redshift_binning(data,voronoi_bins,min_gals=100,coarse=False):
    
    redshift = data['REDSHIFT_1']
    z_bins = []

    for N in np.unique(voronoi_bins):
        inbin = voronoi_bins == N
        n_with_morph = np.sum(inbin)
        if coarse == True:
            n_zbins = 4
        else:
            n_zbins = n_with_morph/min_gals
        #n_zbins = 5
        z = redshift[inbin]
        z = np.sort(z)
        bin_edges = np.linspace(0, len(z)-1, n_zbins+1, dtype=np.int)
        z_edges = z[bin_edges]
        z_edges[0] = 0
        z_edges[-1] = 1
        
        z_bins.append(z_edges)
        
    return z_bins
  
  
def voronoi_assignment(data, rect_bins_table, rect_vbins_table,
                       Mr_bins_min, Mr_bins_range, R50_bins_min, R50_bins_range,
                       reassign=False):
    R50_bin_edges = rect_bins_table['R50_bin_edges']
    Mr_bin_edges = rect_bins_table['Mr_bin_edges']
    n_R50_bins = len(R50_bin_edges) - 1
    n_Mr_bins = len(Mr_bin_edges) - 1
    
    R50 = np.log10(data['PETROR50_R_KPC'])
    Mr = data['PETROMAG_MR']
    
    # get the R50 and Mr bin for each galaxy in the sample
    R50_bins = np.digitize(R50, bins=R50_bin_edges).clip(1, n_R50_bins)
    Mr_bins = np.digitize(Mr, bins=Mr_bin_edges).clip(1, n_Mr_bins)

    # convert R50 and Mr bin indices to indices of bins
    # in the combined rectangular grid
    rect_bins = (Mr_bins - 1) + n_Mr_bins * (R50_bins - 1)

    # get the voronoi bin for each galaxy in the sample
    rect_bin_vbins = rect_vbins_table['vbin']
    voronoi_bins = rect_bin_vbins[rect_bins]
    
    if reassign is True: # Find nearest bin if none available
        rect_bins_assigned = rect_vbins_table[rect_vbins_table['vbin'] != -1]
        R50_bin = rect_bins_assigned['R50']
        Mr_bin = rect_bins_assigned['Mr']
        
        x = (R50_bin - R50_bins_min) / R50_bins_range
        y = (Mr_bin - Mr_bins_min) / Mr_bins_range
        
        unassigned = voronoi_bins == -1
        R50u = (R50[unassigned] - R50_bins_min) / R50_bins_range
        Mru = (Mr[unassigned] - Mr_bins_min) / Mr_bins_range
        
        xy = (np.array([R50u,Mru])).T
        xy_ref = (np.array([x,y])).T
        
        nbrs = NearestNeighbors(n_neighbors=1,algorithm='ball_tree').fit(xy_ref,xy)
        d,i = nbrs.kneighbors(xy)
        i = i.squeeze()
        vbins_reassigned = rect_bins_assigned['vbin'][i]
        voronoi_bins[voronoi_bins == -1] = vbins_reassigned
    
    return voronoi_bins
  
  
def redshift_assignment(data,vbins,zbin_ranges):
    
    zbins = np.zeros(len(data))
    
    for v in (np.unique(vbins)):
        z_range = zbin_ranges[v]
        v_data = data[vbins == v]['REDSHIFT_1']
        z_bin = np.digitize(v_data,bins=z_range)
        zbins[vbins == v] = z_bin
        
    return zbins
  
  
def bin_data(data,question,answer,n_vbins=40,signal=100,plot=True):
  
    raw_column = data[question + '_' + answer + '_weighted_fraction']
    fv_nonzero = raw_column > 0 # Select only the non-zero data to add to the 'signal' for each bin.
    
    R50 = data['PETROR50_R_KPC'][fv_nonzero]
    rect_bins_table,vbins_table,rect_vbins_table,Mr_bins_min,Mr_bins_range,R50_bins_min,R50_bins_range = voronoi_binning(np.log10(R50),
                                                                                                                         data['PETROMAG_MR'][fv_nonzero],
                                                                                                                         n_per_voronoi_bin=np.sum(fv_nonzero)/n_vbins)

    vbins = voronoi_assignment(data[fv_nonzero],rect_bins_table,rect_vbins_table,Mr_bins_min,
                               Mr_bins_range, R50_bins_min, R50_bins_range)
    zbin_ranges = redshift_binning(data[fv_nonzero],vbins,min_gals=signal)
    zbin_ranges_coarse = redshift_binning(data[fv_nonzero],vbins,min_gals=None,coarse=True)
    
    vbins = voronoi_assignment(data, rect_bins_table, rect_vbins_table,
                           Mr_bins_min, Mr_bins_range, R50_bins_min, R50_bins_range,
                           reassign=True)
    
    zbins = redshift_assignment(data,vbins,zbin_ranges)
    zbins_coarse = redshift_assignment(data,vbins,zbin_ranges_coarse)
    
    N_v = np.unique(vbins)
    N_z = []
    
    for v in N_v:
        zbins_v = zbins[vbins == v]
        N_z.append(np.max(zbins_v))
        
    print('{} voronoi bins'.format(len(N_v)))
    print('{} redshift bins per voronoi bin'.format(np.mean(N_z)))
    
    if plot == True:
        
        relative_r = [((T-T.min())/(T.max()-T.min())) for T in [vbins_table['R50'],vbins_table['Mr']]]
        relative_r = relative_r[1]# + relative_r[1]
        r_sort = np.argsort(relative_r)

        for N in np.unique(vbins)[r_sort]:
            inbin = vbins == N
            plt.plot(data['PETROR50_R_KPC'][inbin],data['PETROMAG_MR'][inbin], '.')
    
        for N in range(len(vbins_table)):
            x_text_pos = 10**(vbins_table['R50'][N])
            y_text_pos = vbins_table['Mr'][N]
            plt.text(x_text_pos,y_text_pos,'{}'.format(N),
                     color='w',horizontalalignment='center',
                     verticalalignment='center')

        plt.ylabel(r"$M_r$")
        plt.xlabel(r"$R_{50}$ (kpc)")
        plt.xscale('log')
        _ = plt.axis((0.5, 60, -18, -25))
        
        plt.savefig('figures/voronoi_binning/{}_{}.png'.format(question,answer))
        
    return vbins,zbins,zbins_coarse,vbins_table