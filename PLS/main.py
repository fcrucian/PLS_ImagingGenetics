import numpy as np
import PLS_utils as utils
import PLS

# Global variables
path_out = "./"  #save path
x_variable = 'Imaging'
y_variable = 'Genetics'
experiment_name = 'PLS_test'

# initialize pdf report
pdf = utils.initialize_pdf(experiment_name)
utils.write(f"PLS between {x_variable} and {y_variable}", pdf)
################################################ INPUT DEFINITION #####################################################

# Random data generation
data = utils.random_data_generation(n_samples=100, n_xfeat=10, n_yfeat=20, n_conf=3)
x_labels = data['x_labels']
y_labels = data['y_labels']
group = data['group']
classes = np.unique(group)

# Input data reprocessing
X, Y = utils.data_preprocessing(data['X'], data['Y'], pdf, standardization=True, deconfounding=False, confounds=data['confounds'])
utils.write(f"X shape {X.shape } \nY shape: {Y.shape}", pdf)

################################################## PLS MODEL #########################################################

# Derive optimal number of PLS components
n_components = PLS.n_comp_CV_variance(X, Y, 60, pdf)  # number of components based on the explained variance
utils.write(f"Number of components based on explained variance: {n_components}", pdf)

## PLS fitting
pls_model = PLS.NIPALS_sklearn(X, Y, n_components)

# PLS latent space
PLS.plot_latent_space(pls_model, x_variable, y_variable, classes, group, pdf, path_out)

# PLS scores
PLS.plot_scores(pls_model, x_variable, y_variable, classes, group, pdf, path_out)

# PLS significance
n_permutations = 1000
PLS.compute_significance(X, Y, pls_model, n_permutations, path_out, pdf)

# Write report
pdf.output(path_out + r'/' + experiment_name + '_report.pdf')
