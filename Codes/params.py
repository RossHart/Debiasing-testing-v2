question = 't01_smooth_or_features'

source_directory = '../../fits/'
numpy_save_directory = 'npy/' # Directory to save the debiased values to.

full_sample = 'full_sample.fits'
volume_limited_sample = 'volume_limited_sample.fits'

question_dictionary = 'questions.pickle'

bins_to_plot = [5,10,15,20] # Only plot this/these voronoi bins.

logistic_bounds = ((0.5, 10), (-10, 10))
exponential_bounds = ((10**(-5),10),(10**(-5),10))
#exponential_bounds = ((0.01,5),(0.01,5),(0.5,1.5))