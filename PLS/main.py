import pandas as pd
import numpy as np
import os
import PLS_utils as pu
import PLS

## VARIABLES
# data paths

# TODO generate random data and dele paths

path="C:/Users/feder/Documents/Devel/ADNI/Digas/Data/"
path_out="C:/Users/feder/Documents/Devel/ADNI/Digas/Results/"
x_path = os.path.join(path, 'features_PLS_2.csv')
y_path = os.path.join(path, 'Genetics/ADNI3inEdipoGenes_SCAT.csv')
conf_path = os.path.join(path, 'confounds_up.xlsx')
x_variable = 'Imaging'
y_variable = 'Genetics'
experiment_name = 'PLS_test'

# initialize pdf report
pdf = pu.initialize_pdf(experiment_name)

## Input definition
X_csv = pd.read(x_path)
Y_csv = pd.read(y_path)

#TODO add code to sort the two CSV in the same order
#TODO add averaging and normalization for sMRI
x_labels = X_csv.columns
y_labels = Y_csv.columns
X_np = X_csv.to_numpy()
Y_np = Y_csv.to_numpy()

confounds = pd.read(conf_path)
group = confounds[['study_group']]
classes = np.unique(group)

## Preprocessing

# standardization
X_std = pu.standardization(np.array(X_np))
Y_std = pu.standardization(np.array(Y_np))
pu.write("Standardization: OK ", pdf)
conf_std = pu.standardization(confounds)

# Age deconfounding from the imaging data
pu.write("Imaging confound data: age ", pdf)
X_deconf = pu.linearRegression(X_std, conf_std)
pu.write("Deconfounding: OK", pdf)

## Preprocessed INPUT
X = np.copy(X_deconf)
Y = np.copy(Y_std)
pu.write("X matrix: " + str(x_variable) + " " + str(X.shape) + "\nY matrix: " + str(y_variable) + " " + str(Y.shape), pdf)

### PLS Model

## Choose number of components
pu.write("CROSS VALIDATION ON PLS COMPONENTS", pdf,height=15)
n_components_variance, pdf = PLS.n_comp_CV_variance(X,Y,60,pdf) # number of components based on the explained variace
n_components = n_components_variance # here you can choose the final number of components
pu.write('Number of components based on explianed variance: ' + str(n_components),pdf)

## PLS fitting
pls_model = PLS.NIPALS_sklearn(X, Y, n_components)
weight_x = pls_model.x_weights_
weight_y = pls_model.y_weights_
loading_x = pls_model.x_loadings_
loading_y = pls_model.y_loadings_

#save PLS weights for external tools plotting
col = []
for i in range(n_components):
    stringa = 'comp ' + str(i)
    col.append(stringa)
df_weight_x = pd.DataFrame(weight_x, columns=col)
df_weight_x.insert(0, 'variables', x_labels)
df_weight_x.to_excel(path_out + '\weight_X.xlsx')
df_weight_y = pd.DataFrame(weight_y, columns=col)
df_weight_y.insert(0, 'variables', y_labels)
df_weight_y.to_excel(path_out + '\weight_Y.xlsx')

# Plot PLS latent space
pdf = PLS.plot_latent_space(X, Y, pls_model, n_components, x_variable, y_variable, classes, group, pdf, path_out)

# Plot PLS scores
pdf = PLS.plot_scores(X, Y, pls_model, n_components, classes, group, x_variable, y_variable, pdf, path_out, bonferroni=True)

## Compute PLS singular values
eigval = np.zeros(n_components)
for i in range(n_components):
    eigval[i] = np.dot((weight_x[:, i]).T, (X.transpose().dot(Y)).dot(weight_y[:, i]))

pu.write("EIGENVALUES NIPALS: " + str(eigval), pdf)
## Compute PLS significance
n_permutations = 1000
p_value, pdf = PLS.compute_significance(X, Y, eigval, n_components, n_permutations, path_out, pdf)
print('p-value = {0}'.format(p_value))

pdf.output(path_out+r'/'+experiment_name+'_report.pdf')
