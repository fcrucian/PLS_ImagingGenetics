import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.linalg import pinv
from scipy import stats
from sklearn.cross_decomposition import PLSCanonical, PLSSVD, CCA

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

from numpy import arange
from pandas import read_csv
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

from matplotlib import gridspec
import matplotlib.patches as mpatches
from statannot import add_stat_annotation

import random

import seaborn as sns

from fpdf import FPDF
from datetime import date
import utils

import scipy

def NIPALS_sklearn(X, Y, n_components):
    """Computes the PLSCanonical
    Input:
    X -> X matrix for the PLS model mxn where m is the number of subjects and n the X feature number
    Y -> Y matrix for the PLS model mxk where m is the number of subjects and k the Y feature number
    n_components -> number of components to be used to compute the PLS (found through the functions n_comp_CV_pred_error or n_comp_CV_variance)
    Output:
    plsca -> Fitted PLS model on X and Y
    """
    plsca = PLSCanonical(n_components, scale=False, algorithm='nipals', max_iter=10000, tol=1e-5)
    plsca.fit(X, Y)
    return plsca

def n_comp_CV_pred_error(X, Y, group, pdf, path):
    """Computes the optimal number of components for the PLSCanonical minimizing the PRESS error
    Input:
    X -> X matrix for the PLS model mxn where m is the number of subjects and n the X feature number
    Y -> Y matrix for the PLS model mxk where m is the number of subjects and k the Y feature number
    group -> Subjects' true labels
    pdf-> pdf variable used along the whole code to store the results
    path-> Path to save the plots (same as the pf path)
    n_components -> number of components to be used to compute the PLS (found through the functions n_comp_CV_pred_error or n_comp_CV_variance)

    Output:
    n_components -> optimal number of components
    pdf -> upadted pdf
    """

    all_components = np.min([Y.shape[1], X.shape[1]])

    rep_results = np.ndarray([2, all_components])

    for k in range(all_components):
        print("comp: " + str(k))

        count = 0

        kfold = StratifiedKFold(2, shuffle=True, random_state=1)

        for train, test in kfold.split(X, group):
            X_train = X[train]
            X_test = X[test]

            Y_train = Y[train]
            Y_test = Y[test]

            # Creating a model for each data batch
            plsca1 = NIPALS_sklearn(X_train, Y_train, n_components=k + 1)
            plsca2 = NIPALS_sklearn(X_test, Y_test, n_components=k + 1)

            # Fitting a model for each data batch
            plsca1.fit(X_train, Y_train)
            plsca2.fit(X_test, Y_test)

            # Quantifying the prediction error on the unseen data batch
            err1 = np.sum((plsca1.predict(X_test) - Y_test) ** 2)
            err2 = np.sum((plsca2.predict(X_train) - Y_train) ** 2)
            # Quantifying the prediction error on the unseen data batch
            rep_results[count, k] = np.mean([err1,err2])
            count = count + 1

    plt.plot(range(1, all_components + 1), np.mean(rep_results, 0))
    plt.xlabel('# components')
    plt.ylabel('PRESS error')
    plt.savefig(path + 'NumberComponents_PRESS.jpg')

    id = np.where(np.mean(rep_results, 0) == np.min(np.mean(rep_results, 0)))
    n_components = id[0][0] + 1

    stringa = "Number of components resulted optimizing PRESS error: " + str(n_components)
    pdf.multi_cell(0, 5, stringa)
    utils.add_image(path + 'NumberComponents_PRESS.jpg', pdf, w=100)

    return n_components, pdf

def n_comp_CV_variance(X, Y, threshold, pdf):
    """Computes the optimal number of components for the PLSCanonical to explain a user defined data variability
    Input:
    X -> X matrix for the PLS model mxn where m is the number of subjects and n the X feature number
    Y -> Y matrix for the PLS model mxk where m is the number of subjects and k the Y feature number
    pdf-> pdf variable used along the whole code to store the results
    threshold -> percentage of data variability to be explained

    Output:
    n_components_perc -> optimal number of components
    pdf -> upadted pdf
    """

    eig_vect_x, eig_val, eig_vect_y = np.linalg.svd(X.transpose().dot(Y))
    eig_val_perc = np.zeros(len(eig_val))

    for i in range(0, len(eig_val)):
        eig_val_perc[i] = eig_val[i] / (np.sum(eig_val)) * 100

    stringa = "Percentage of each component: " + str(eig_val_perc)
    pdf.multi_cell(0, 5, stringa)

    summary = 0.0
    count = 0

    while summary <= threshold:
        summary += eig_val_perc[count]
        count += 1

    n_components_perc = count

    stringa = "Number of components needed to explain at least the "+ str(threshold)+"% of the data: " + str(n_components_perc)
    pdf.multi_cell(0, 5, stringa)

    return n_components_perc, pdf


def compute_significance(X, Y, autovalore, n_components, n_permutations, path, pdf, mask_weights=False):
    """Computes the PLS significance through a permutation test
    Input:
    X -> X matrix for the PLS model mxn where m is the number of subjects and n the X feature number
    Y -> Y matrix for the PLS model mxk where m is the number of subjects and k the Y feature number
    autovalore -> The eigenvalues obtained from the model to be tested
    n_components -> number of components to be used to compute the PLS (found through the functions n_comp_CV_pred_error or n_comp_CV_variance)
    n_permutations -> number of permutations of the Y matrix rows
    pdf-> pdf variable used along the whole code to store the results

    Output:
    p_value -> obtained p_value from the permutation test
    pdf -> upadted pdf
    """
    count = 0.0
    eig_val_perm_tot = []
    for i in range(n_permutations):
        eig_val_perm = np.zeros(n_components)
        #Y_perm = np.take(Y, np.random.permutation(Y.shape[0]), axis=0)
        X_perm = np.take(X, np.random.permutation(X.shape[0]), axis=0)
        plsca = NIPALS_sklearn(X_perm, Y, n_components)
        weight_x = plsca.x_weights_
        weight_y = plsca.y_weights_

        for t in range(n_components):
            eig_val_perm[t] = np.dot((weight_x[:, t]).T, ((X_perm.transpose().dot(Y))).dot(weight_y[:, t]))

        eig_val_perm_tot.append(eig_val_perm)

        if (np.sum(eig_val_perm) > np.sum(autovalore)):
            count += 1.0

    p_value = count / n_permutations

    eig_val_perm_tot = np.array(eig_val_perm_tot)
    p_values = []
    for i in range(autovalore.shape[0]):
        count = 0.0
        for j in range(eig_val_perm_tot.shape[0]):
            if eig_val_perm_tot[j, i] > autovalore[i]:
                count += 1.0
        p = count / n_permutations
        if p <= 0.05:
            print('Component ' + str(i))
            print('p-value: ' + str(p))
        p_values.append(p)

    print("P-values for each component: " + str(p_values))
    stringa = "P-values for each component: " + str(p_values)
    pdf.multi_cell(0, 5, stringa)

    fig, axs = plt.subplots(4,4,figsize=(10, 10))
    axs_rav = axs.ravel()
    eig_val_perm_tot = np.array(eig_val_perm_tot)

    for i in range(n_components):
        axs_rav[i].hist(eig_val_perm_tot[:, i], bins=32)
        axs_rav[i].axvline(x=autovalore[i], color='red')
        axs_rav[i].set_title('Eigenval ' + str(i))

    plt.tight_layout()
    plt.show()
    fig.savefig(path + '/histograms.png', dpi=300)

    #utils.add_image(path + '/histograms.png',pdf,w=120)

    stringa = "SIGNIFICANCE"
    pdf.multi_cell(0, 15, stringa)

    stringa = "p-value = " + str(p_value)
    pdf.multi_cell(0, 5, stringa)

    stringa = "Number of permutations: " + str(n_permutations)
    pdf.multi_cell(0, 5, stringa)

    return p_value, pdf

def plot_scores(X, Y, plsca, n_components, classes, group, x_variable, y_variable, pdf, path,bonferroni=False,mask_weights=False,validation=False):
    """Plot the PLS X and Y latent space projection scores
    Input:
    plsca -> fitted PLS model
    n_components -> number of components to be used to compute the PLS (found through the functions n_comp_CV_pred_error or n_comp_CV_variance)
    classes -> list of labels of the different subjects classes
    group -> Subjects' true labels
    x_variable -> keyword to describe X data
    y_variable -> keyword to describe y data
    pdf-> pdf variable used along the whole code to store the results
    path-> Path to save the plots (same as the pf path)

    Output:
    pdf -> upadted pdf
    """
    x_scores_tot, y_scores_tot = plsca.transform(X, Y)

    for i in range(n_components):

        x_scores = x_scores_tot[:,i]
        y_scores = y_scores_tot[:,i]

        data=np.array([x_scores, group])
        df_x = pd.DataFrame(data=data.T, columns=["x_scores","class"])
        x_x = "class"
        y_x = "x_scores"

        data=np.array([y_scores, group])
        df_y = pd.DataFrame(data=data.T, columns=["y_scores","class"])
        x_y = "class"
        y_y = "y_scores"

        colors_=['#0eb5b2','#ffdc40']
        color_dict=dict(zip(order,colors_))

        x_scores=[]
        y_scores=[]

        for c in classes:
            x_scores.append(x_scores_tot[group == c][:, i])
            y_scores.append(y_scores_tot[group == c][:, i])

            box_pairs_x = []
            p_values_x = []
            for c1 in range(len(order)):
                for c2 in range(c1+1, len(order)):
                    h, p_mann_x = scipy.stats.mannwhitneyu(x_scores[c1],x_scores[c2])
                    utils.write(
                        'MannWhitneyU: ' + str(order[c1]) + ' vs ' + str(order[c2]) + ': p_value_x = ' + str(p_mann_x),pdf)
                    if p_mann_x <= 0.05:
                        p_values_x.append(p_mann_x)
                        box_pairs_x.append((str(order[c1]), str(order[c2])))



            box_pairs_y = []
            p_values_y = []
            for c1 in range(len(order)):
                for c2 in range(c1+1, len(order)):
                    h, p_mann_y = scipy.stats.mannwhitneyu(y_scores[c1],y_scores[c2])
                    utils.write(
                        'MannWhitneyU: ' + str(order[c1]) + ' vs ' + str(order[c2]) + ': p_value_y = ' + str(p_mann_y),pdf)
                    if p_mann_y <= 0.05:
                        p_values_y.append(p_mann_y)
                        box_pairs_y.append((str(order[c1]), str(order[c2])))

        fig1, ax = plt.subplots()
        boxplot = sns.boxplot(data=df_x, x=x_x, y=y_x, order=order,  width=0.5,palette=color_dict,ax=ax)

        if len(p_values_x) != 0:
            test_results = add_stat_annotation(ax, data=df_x, x=x_x, y=y_x, order=order,
                                           box_pairs=box_pairs_x, pvalues=p_values_x,
                                           text_format='star', perform_stat_test=False,
                                           loc='outside', verbose=2, comparisons_correction=None,line_offset=0.05)

        ax.set_ylabel('Projection scores for '+ x_variable)
        ax.set_xlabel('')
        plt.tight_layout()
        plt.show()

        fig = boxplot.get_figure()
        fig.savefig(path + '/' + 'x_scores_component_' + str(i) + '.jpg', dpi=300,bbox_inches='tight')
        stringa = "X scores component: " + str(i)
        pdf.multi_cell(0, 5, stringa)
        utils.add_image(path + '/' + 'x_scores_component_' + str(i) + '.jpg', pdf, w=120)

        fig1, ax = plt.subplots()
        boxplot = sns.boxplot(data=df_y, x=x_y, y=y_y, order=order,  width=0.5,palette=color_dict, ax=ax)

        if len(p_values_y) != 0:
            test_results = add_stat_annotation(ax, data=df_y, x=x_y, y=y_y, order=order,
                                           box_pairs=box_pairs_y, pvalues=p_values_y,
                                           text_format='star',perform_stat_test=False,
                                           loc='outside', verbose=2, comparisons_correction=None)

        ax.set_ylabel('Projection scores for '+ y_variable)
        ax.set_xlabel('')
        plt.tight_layout()
        plt.show()

        fig = boxplot.get_figure()
        fig.savefig(path + '/' + 'y_scores_component_' + str(i) + '.jpg', dpi=300,bbox_inches='tight')
        stringa = "Y scores component: " + str(i)
        pdf.multi_cell(0, 5, stringa)
        utils.add_image(path + '/' + 'y_scores_component_' + str(i) + '.jpg', pdf, w=120)

        plt.show()
    return pdf


def plot_latent_space(X, Y, plsca, n_components, x_variable, y_variable, classes, group, pdf, path, mask_weights=False, validation=False):
    """Plot the PLS latent space projections
    Input:
    X -> X matrix for the PLS model mxn where m is the number of subjects and n the X feature number
    Y -> Y matrix for the PLS model mxk where m is the number of subjects and k the Y feature number
    plsca -> fitted PLS model
    n_components -> number of components to be used to compute the PLS (found through the functions n_comp_CV_pred_error or n_comp_CV_variance)
    x_variable -> keyword to describe X data
    y_variable -> keyword to describe y data
    classes -> list of labels of the different subjects classes
    group -> Subjects' true labels
    pdf-> pdf variable used along the whole code to store the results
    path-> Path to save the plots (same as the pf path)

    Output:
    pdf -> upadted pdf
    """

    utils.write("LATENT SPACE", pdf, height=15)
    colors_pastel = sns.color_palette('pastel')

    weight_x = plsca.x_weights_
    weight_y = plsca.y_weights_

    if mask_weights:
        weight_x[42::, :] = 0

    leg = []
    count = 0

    colors_pastel = sns.color_palette('bright')
    colors_groups = np.zeros([group.shape[0], 3])
    #group[group=='MCI']='PAT'
    #group[group=='EMCI']='PAT'
    #group[group=='LMCI']='PAT'
    #group[group == 'AD'] = 'PAT'

    for j in classes:
        if j == 'AD':
            leg.append(mpatches.Patch(color=colors_pastel[0], label=j))
            colors_groups[group == j, :] = colors_pastel[0]
        elif j == 'MCI':
            leg.append(mpatches.Patch(color=colors_pastel[1], label=j))
            colors_groups[group == j, :] = colors_pastel[1]
        elif j == 'EMCI':
            leg.append(mpatches.Patch(color=colors_pastel[2], label=j))
            colors_groups[group == j, :] = colors_pastel[2]
        elif j == 'LMCI':
            leg.append(mpatches.Patch(color=colors_pastel[4], label=j))
            colors_groups[group == j, :] = colors_pastel[4]
        elif j == 'CN':
#            leg.append(mpatches.Patch(color=colors_pastel[7], label=j))
#            colors_groups[group == j, :] = colors_pastel[7]
            leg.append(mpatches.Patch(color=[1, 0.86, 0.25], label=j))
            colors_groups[group == j, :] = [1, 0.86, 0.25]

        elif j == 'PAT':
#            leg.append(mpatches.Patch(color=colors_pastel[0], label=j))
#            colors_groups[group == j, :] = colors_pastel[0]
            leg.append(mpatches.Patch(color=[0.05, 0.71, 0.70], label=j))
            colors_groups[group == j, :] = [0.05, 0.71, 0.70]

        elif j == 0:
            leg.append(mpatches.Patch(color=colors_pastel[0], label=j))
            colors_groups[group == j, :] = colors_pastel[0]
        elif j == 1:
            leg.append(mpatches.Patch(color=colors_pastel[7], label=j))
            colors_groups[group == j, :] = colors_pastel[7]
#        count+=1
    colors = np.copy(colors_groups)

    x_ticks = []
    y_ticks = []
    for i in range(n_components):
        x_ticks.append('x_comp_'+str(i))
        y_ticks.append('y_comp_'+str(i))
    #cov, pdf = utils.get_covariance(np.dot(X,weight_x).T, np.dot(Y, weight_y).T, pdf)
    #pdf = utils.plot_covariance_matrix(cov, x_ticks, y_ticks, path, pdf)

    for i in range(n_components):

        tx = np.dot(X, weight_x[:, i])
        ty = np.dot(Y, weight_y[:, i])
        x_lim = np.max([tx.max(), np.abs(tx.min()), ty.max(), np.abs(ty.min())])
        y_lim = np.max([ty.max(), np.abs(ty.min())])
        plt.figure(tight_layout=True)
        # plt.subplot(Y.shape[1], 1, i+1)
        plt.scatter(tx, ty, c=colors, alpha=0.4)

        corr = np.corrcoef(tx.astype('float64'),ty.astype('float64').T)[0,1]
        cov = np.cov(tx.astype('float64'),ty.astype('float64').T)[0,1]
        print('Component '+str(i+1) + '\nCovariance: '+str(cov)+'\nCorrcoef: '+str(corr))
        utils.write("Covariance: " + str(cov),pdf)
        utils.write("Correlation: " + str(corr),pdf)

        plt.ylabel(y_variable)
        plt.xlabel(x_variable)
        if validation:
            plt.title('Validation projection on component ' + str(i + 1))
        else:
            plt.title('Projection on component ' + str(i + 1))

        plt.axis('scaled')
        plt.xlim([-x_lim, x_lim])
        plt.ylim([-x_lim, x_lim])

        plt.legend(handles=leg)

        if validation:
            plt.savefig(path + '/' + str(i) + 'latent_PLS_all_subjs_validation.jpg', dpi=300)
            plt.show()
            utils.write("Projection on component: " + str(i), pdf)
            utils.add_image(path + '/' + str(i) + 'latent_PLS_all_subjs_validation.jpg', pdf, w=120)
        else:
            plt.savefig(path + '/' + str(i) + 'latent_PLS_all_subjs.jpg', dpi=300)
            plt.show()
            utils.write("Projection on component: " + str(i), pdf)
            utils.add_image(path + '/' + str(i) + 'latent_PLS_all_subjs.jpg', pdf, w=120)
    return pdf
