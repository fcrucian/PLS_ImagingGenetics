import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.cross_decomposition import PLSCanonical
from statannot import add_stat_annotation

import PLS_utils as utils


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

def n_comp_CV_variance(X, Y, threshold, pdf):
    """Computes the optimal number of components for the PLSCanonical to explain a user defined data variability
    Input:
    X -> X matrix for the PLS model mxn where m is the number of subjects and n the X feature number
    Y -> Y matrix for the PLS model mxk where m is the number of subjects and k the Y feature number
    pdf-> pdf variable used along the whole code to store the results
    threshold -> percentage of data variability to be explained

    Output:
    n_components_perc -> optimal number of components
    """

    eig_vect_x, eig_val, eig_vect_y = np.linalg.svd(X.transpose().dot(Y))
    eig_val_perc = np.zeros(len(eig_val))

    for i in range(0, len(eig_val)):
        eig_val_perc[i] = eig_val[i] / (np.sum(eig_val)) * 100

    utils.write(f"Variance percentage of each component: {eig_val_perc}",pdf)

    summary = 0.0
    count = 0

    while summary <= threshold:
        summary += eig_val_perc[count]
        count += 1

    n_components_perc = count

    string = f"Number of components needed to explain at least the {threshold}% of the data: {n_components_perc}"
    utils.write(string, pdf)

    return n_components_perc


def compute_significance(X, Y, plsca, n_permutations, path, pdf):
    """Computes the PLS significance through a permutation test
    Input:
    X -> X matrix for the PLS model mxn where m is the number of subjects and n the X feature number
    Y -> Y matrix for the PLS model mxk where m is the number of subjects and k the Y feature number
    n_components -> number of components to be used to compute the PLS (found through the functions n_comp_CV_pred_error or n_comp_CV_variance)
    n_permutations -> number of permutations of the Y matrix rows
    pdf-> pdf variable used along the whole code to store the results

    Output:
    p_value -> obtained p_value from the permutation test
    pdf -> upadted pdf
    """

    # PLS singular values
    weight_x = plsca.x_weights_
    weight_y = plsca.y_weights_

    eigval = np.zeros(plsca.n_components)
    for i in range(plsca.n_components):
        eigval[i] = np.dot((weight_x[:, i]).T, (X.transpose().dot(Y)).dot(weight_y[:, i]))

    utils.write(f"\nEIGENVALUES NIPALS: {eigval}", pdf)

    # permutations of X matrix rows
    count = 0.0
    eig_val_perm_tot = []
    for i in range(n_permutations):
        eig_val_perm = np.zeros(plsca.n_components)
        X_perm = np.take(X, np.random.permutation(X.shape[0]), axis=0) #permute X matrix
        plsca_perm = NIPALS_sklearn(X_perm, Y, plsca.n_components) #compute the PLS between the permuted X and the original Y
        weight_x_perm = plsca_perm.x_weights_
        weight_y_perm = plsca_perm.y_weights_

        for t in range(plsca.n_components):
            eig_val_perm[t] = np.dot((weight_x_perm[:, t]).T, ((X_perm.transpose().dot(Y))).dot(weight_y_perm[:, t])) #calculate the permuted eigenvals

        eig_val_perm_tot.append(eig_val_perm)

        if (np.sum(eig_val_perm) > np.sum(eigval)): #check if the sum of the permuted eigenvalues is higher than the sum of the original ones
            count += 1.0

    # global p-value
    p_value = count / n_permutations #obtain the overall p-value
    utils.write(f"\nGLOBAL SIGNIFICANCE "
             f"\np-value = {p_value}"
             f"\nnumber of permutations = {n_permutations}",pdf)

    # calculate p-value for each component
    eig_val_perm_tot = np.array(eig_val_perm_tot)
    p_values = []
    for i in range(eigval.shape[0]):
        count = 0.0
        for j in range(eig_val_perm_tot.shape[0]):
            if eig_val_perm_tot[j, i] > eigval[i]:
                count += 1.0
        p = count / n_permutations
        p_values.append(p)

    utils.write(f"\nPLS COMPONENTS SIGNIFICANCE "
             f"\nP-values for each component: {p_values}"
             f"\nnumber of permutations = {n_permutations}", pdf)

    # plot p-values distributions
    fig, axs = plt.subplots(2,3,figsize=(6, 4))
    axs_rav = axs.ravel()
    eig_val_perm_tot = np.array(eig_val_perm_tot)

    for i in range(plsca.n_components):
        axs_rav[i].hist(eig_val_perm_tot[:, i], bins=32)
        axs_rav[i].axvline(x=eigval[i], color='red')
        axs_rav[i].set_title(f'Eigenval {i}')

    plt.tight_layout()
    plt.show()
    fig.savefig(path + '/p-values_distributions.png', dpi=300)
    utils.add_image(path + '/p-values_distributions.png',pdf,w=120)

def plot_scores(plsca, x_variable, y_variable, classes, group, pdf, path,bonferroni=False):
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

    utils.write("PROJECTION SCORES", pdf, height=15)

    order = ['CN', 'PAT']
    colors_ = ['#0eb5b2', '#ffdc40']
    color_dict = dict(zip(order, colors_))
    leg = [mpatches.Patch(color='#0eb5b2', label='CN'), mpatches.Patch(color='#ffdc40', label='PAT')]

    for i in range(plsca.n_components):

        utils.write(f"Component {i}", pdf)
        x_scores_tot = plsca.x_scores_[:, i]
        y_scores_tot = plsca.y_scores_[:, i]

        data = np.array([x_scores_tot, group],dtype=object)
        df_x = pd.DataFrame(data=data.T, columns=["x_scores","class"])
        x_x = "class"
        y_x = "x_scores"

        data = np.array([y_scores_tot, group],dtype=object)
        df_y = pd.DataFrame(data=data.T, columns=["y_scores","class"])
        x_y = "class"
        y_y = "y_scores"


        x_scores=[]
        y_scores=[]

        for c in classes:
            x_scores.append(x_scores_tot[group == c])
            y_scores.append(y_scores_tot[group == c])

        box_pairs_x = []
        p_values_x = []
        for c1 in range(len(order)):
            for c2 in range(c1+1, len(order)):
                h, p_mann_x = scipy.stats.mannwhitneyu(x_scores[c1],x_scores[c2])
                utils.write(
                    f'MannWhitneyU: {order[c1]} vs {order[c2]}: p_value_x = {p_mann_x}',pdf)
                if p_mann_x <= 0.05:
                    p_values_x.append(p_mann_x)
                    box_pairs_x.append((str(order[c1]), str(order[c2])))

        box_pairs_y = []
        p_values_y = []
        for c1 in range(len(order)):
            for c2 in range(c1+1, len(order)):
                h, p_mann_y = scipy.stats.mannwhitneyu(y_scores[c1],y_scores[c2])
                utils.write(
                    f'MannWhitneyU: {order[c1]} vs {order[c2]}: p_value_y = {p_mann_y}',pdf)
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

        ax.set_ylabel(f'Projection scores for {x_variable}')
        ax.set_xlabel('')
        plt.legend(handles=leg)
        plt.tight_layout()
        plt.show()

        fig = boxplot.get_figure()
        fig.savefig(path + f'/x_scores_component_{i}.jpg', dpi=300,bbox_inches='tight')
        utils.write(f"X scores component: {i}", pdf)
        utils.add_image(path + f'/x_scores_component_{i}.jpg', pdf, w=120)

        fig1, ax = plt.subplots()
        boxplot = sns.boxplot(data=df_y, x=x_y, y=y_y, order=order,  width=0.5,palette=color_dict, ax=ax)

        if len(p_values_y) != 0:
            test_results = add_stat_annotation(ax, data=df_y, x=x_y, y=y_y, order=order,
                                           box_pairs=box_pairs_y, pvalues=p_values_y,
                                           text_format='star',perform_stat_test=False,
                                           loc='outside', verbose=2, comparisons_correction=None)

        ax.set_ylabel(f'Projection scores for {y_variable}')
        ax.set_xlabel('')
        plt.tight_layout()
        plt.legend(handles=leg)
        plt.show()

        fig = boxplot.get_figure()
        fig.savefig(path + f'/y_scores_component_{i}.jpg', dpi=300,bbox_inches='tight')
        utils.write(f"Y scores component: {i}", pdf)
        utils.add_image(path + f'/y_scores_component_{i}.jpg', pdf, w=120)

        plt.show()



def plot_latent_space(plsca, x_variable, y_variable, classes, group, pdf, path):
    """
    Plot the PLS latent space projections
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
    """

    utils.write("LATENT SPACE", pdf, height=15)

    colors_pastel = sns.color_palette('pastel')
    colors_groups = np.zeros([group.shape[0], 3])
    leg = []
    for j in classes:
        if j == 'CN':
            leg.append(mpatches.Patch(color=[1, 0.86, 0.25], label=j))
            colors_groups[group == j, :] = [1, 0.86, 0.25]
        elif j == 'PAT':
            leg.append(mpatches.Patch(color=[0.05, 0.71, 0.70], label=j))
            colors_groups[group == j, :] = [0.05, 0.71, 0.70]

    colors = np.copy(colors_groups)

    for i in range(plsca.n_components):
        utils.write(f"Component {i}", pdf)
        tx = plsca.x_scores_[:, i]
        ty = plsca.y_scores_[:, i]

        x_lim = np.max([tx.max(), np.abs(tx.min()), ty.max(), np.abs(ty.min())])
        y_lim = np.max([ty.max(), np.abs(ty.min())])
        plt.figure(tight_layout=True)
        # plt.subplot(Y.shape[1], 1, i+1)
        plt.scatter(tx, ty, c=colors, alpha=0.4)

        corr = np.corrcoef(tx.astype('float64'),ty.astype('float64').T)[0,1]
        cov = np.cov(tx.astype('float64'),ty.astype('float64').T)[0,1]
        utils.write(f"Covariance: {cov} \nCorrelation: {corr}",pdf)

        plt.ylabel(y_variable)
        plt.xlabel(x_variable)
        plt.title(f"Projection on component {i}")

        plt.axis('scaled')
        plt.xlim([-x_lim, x_lim])
        plt.ylim([-x_lim, x_lim])

        plt.legend(handles=leg)

        plt.savefig(path + f"/component_{i}_latent_space.jpg", dpi=300)
        plt.show()
        utils.write(f"Projection on component: {i}", pdf)
        utils.add_image(path + f"/component_{i}_latent_space.jpg", pdf, w=120)
