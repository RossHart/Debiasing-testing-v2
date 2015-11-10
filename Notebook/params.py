question = 't01_smooth_or_features'

source_directory = '../../fits/'
numpy_save_directory = 'npy/' # Directory to save the debiased values to.

full_sample = 'full_sample.fits'
volume_limited_sample = 'volume_limited_sample.fits'

question_dictionary = 'questions.pickle'

bins_to_plot = [5, 10, 15, 20, 25, 30] # Only plot this/these voronoi bins.

#logistic_bounds = ((0.5, 6), (-7.5, 0))
logistic_bounds = ((0.1, 50), (-20.0, 0))
exponential_bounds = ((0.1, 50), (0.01, 3))
inverse_bounds = ((0.01,20),(0.01,5))
