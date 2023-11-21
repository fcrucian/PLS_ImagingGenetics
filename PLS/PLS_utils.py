import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import pinv
from fpdf import FPDF
from datetime import date


def initialize_pdf(experiment_name):
    """
    Initialize pdf to generate test report
    Input:
    experiment_name: string, chosed experiment name

    Output:
    pdf: pdf object to store test results
    """

    today = date.today()
    d2 = today.strftime("%B %d, %Y")
    pdf = FPDF(format='A4', unit='mm')
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"{experiment_name} {d2}", ln=1, align="C")

    return pdf


def add_image(image_path, pdf_file, w):
    '''Helper function to add image to pdf object'''
    pdf_file.image(image_path, w=w)


def write(string, pdf, width=150, height=5):
    '''Helper function to add write text to pdf object'''
    pdf.multi_cell(width, height, string)


def random_data_generation(n_samples, n_xfeat, n_yfeat, n_conf=None):
    """
    Generate random data including X, Y and confounds matrices
    Input:
    n_samples: int, number of simulated subjects
    n_xfeat: int, number of simulated features in X
    n_yfeat: int, number of simulated features in Y
    n_conf: int or None, number of simulated confounds variables (if deconfounding not needed input None)

    Output:
    data: dict, generated data
    """
    # generate ID variables
    sample_ids = ['subj_{}'.format(i) for i in range(0,n_samples)]

    # generate the X and Y features names
    feat_x = ['feat_x_{}'.format(i) for i in range(0,n_xfeat)]
    feat_y = ['feat_y_{}'.format(i) for i in range(0,n_yfeat)]

    # generate X and Y dataframes
    X_df = pd.DataFrame(np.random.randn(n_samples, n_xfeat), columns=feat_x)
    Y_df = pd.DataFrame(np.random.randn(n_samples, n_yfeat), columns=feat_y)
    X_df['ID']=sample_ids # add ID column
    Y_df['ID']=sample_ids # add ID column

    # if n_conf is not None, generate the confounds variables dataframe
    if n_conf != None:
        conf_lab = ['conf_{}'.format(i) for i in range(0,n_conf)]
        confounds_df = pd.DataFrame(np.random.randn(n_samples, n_conf), columns=conf_lab)
        confounds_df['ID']=sample_ids
    else:
        conf_lab = None
        confounds_df = None

    # generate the simulted classes for the simulated instances (in this example two classes are simulated)
    group =  np.random.choice(['PAT','CN'],size=n_samples)

    # pupulate the data dictionary
    data = {'X': X_df,
            'Y': Y_df,
            'confounds': confounds_df,
            'x_labels': feat_x,
            'y_labels': feat_y,
            'confounds_labels': conf_lab,
            'group': group}

    return data


def data_preprocessing(X, Y, pdf, standardization=True, deconfounding=False, confounds=None):
    """
    Simulates the preprocessing perfomed in the X and Y matrices including standardization and deconfounding
    Input:
    X -> mxn array, X matrix for the PLS model where m is the number of subjects and n the X feature number
    Y -> mxk array, Y matrix for the PLS model where m is the number of subjects and k the Y feature number
    standardization -> bool, apply mean/variance standardization (default True)
    deconfounding -> bool, apply deconfounding (default False)
    confounds -> mxj array, where m is the number of subjects and k the confounds number (considered only if decounfounding=True, default None)
    pdf-> pdf variable used along the whole code to store the results

    Output:
    X_std, Y_std -> input matrices preproocessed and standardized (if deconfounding = False)
    X_deconf, Y_deconf -> input matrices prerpocessed, standardized and deconfounded (if deconfounding = True)

    """
    # N.B. when using real data it is FUNDAMENTAL to have the same subjects in both X and Y and to sort the two dataframes in the same order

    # Select the common subjects in both X and Y
    common_ids = set(X['ID']).intersection(Y['ID'])

    # Filter X and Y based on the common IDs
    Y_up = Y.loc[Y.ID.isin(common_ids)]
    X_up = X.loc[X.ID.isin(common_ids)]

    # Sort X and Y dataframes
    X_up.sort_values(by='ID', inplace=True)
    Y_up.sort_values(by='ID', inplace=True)

    # The subjects column can be dropped and the dataframe transformed to numpy array
    X_np = X_up.drop('ID', axis=1).to_numpy()
    Y_np = Y_up.drop('ID', axis=1).to_numpy()

    # standardization
    X_std = standardize(X_np)
    Y_std = standardize(Y_np)
    write("Standardization: OK ", pdf)

    # if deconfounding, filter the confounds based on the common IDS, sort the dataframe and drop the ids column
    if deconfounding:
        confounds_up = confounds.loc[confounds.ID.isin(common_ids)]
        confounds.sort_values(by='ID', inplace=True)
        confounds_np = confounds_up.drop('ID', axis=1).to_numpy()
        confounds_std = standardize(confounds_np)

        # Compute linear regression between the X (or Y) and the confouds, and consider the residuals as deconfounded data.
        X_deconf = linearRegression(X_std, confounds_std)
        Y_deconf = linearRegression(Y_std, confounds_std)
        write("Deconfounding: OK", pdf)
        conf_labels = list(confounds_up.columns[:-1])
        write("Confounds: " + str(conf_labels), pdf)
        return X_deconf, Y_deconf

    return X_std, Y_std


def standardize(X):
    '''Standardize the matrix X with n columns'''
    X = np.copy(X)
    if len(X.shape)>1:
        for i in range(X.shape[1]):
            X[:, i] = (X[:, i] - np.mean(X[:, i])) / (np.std(X[:, i]) + 0.000001)  # Z-standardization + a costant = 0.00001 in order to not get a division by zero
    else:
        X = (X - np.mean(X)) / (np.std(X) + 0.000001)  # Z-standardization + a costant = 0.00001 in order to not get a division by zero
    return X


def linearRegression(X, y):
    '''Compute Linear Regression between X and y'''
    if len(y.shape) > 1:
        CONFbeta = np.dot(pinv(y.astype(None)), X)
        X_deconf = X - np.dot(y, CONFbeta)
    else:
        CONFbeta = np.dot(pinv([y.astype(None)]).T, X)
        X_deconf = X - np.dot(y[:, np.newaxis], CONFbeta)
    return X_deconf


def get_covariance(X, Y):
    '''Helper function to return correlation coefficient between X and Y'''
    cov = np.corrcoef(X,Y)
    return cov


def plot_settings():
    '''Set general plot settings'''
    plt.rc('axes', titlesize=18)  # fontsize of the axes title
    plt.rc('axes', labelsize=14)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=11)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=11)  # fontsize of the tick labels
    plt.rc('legend', fontsize=13)  # legend fontsize
    plt.rc('font', size=13)  # controls default text sizes
