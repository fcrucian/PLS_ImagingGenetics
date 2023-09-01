import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.linalg import pinv
from scipy import stats
from sklearn.cross_decomposition import PLSCanonical, CCA

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

from sklearn import linear_model
from numpy import arange
from pandas import read_csv
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import Lasso

from matplotlib import gridspec
import matplotlib.patches as mpatches

import random

from sklearn.model_selection import StratifiedKFold
import seaborn as sns

from fpdf import FPDF
from datetime import date

def initialize_pdf(experiment_name):
    today = date.today()
    d2 = today.strftime("%B %d, %Y")
    pdf = FPDF(format='A4', unit='mm')
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=experiment_name + str(d2), ln=1, align="C")

    return pdf

def add_image(image_path, pdf_file, w):
    pdf_file.image(image_path, w=w)


def write(stringa, pdf, width=150, height=5):
    pdf.multi_cell(width, height, stringa)

def substitute_NaN(data, group):
    states, counts = np.unique(group, return_counts=True)
    data_np = data.astype('float32')

    for state in states:
        idx = np.where(group[:] == state)
        for conf_i in range(data.shape[1]):
            data_tmp = data_np[idx, conf_i].astype('float32')
            idx_nan = np.where(np.isnan(data_tmp))
            not_nan_data = data_tmp[~ np.isnan(data_tmp)]
            median_group = np.median(not_nan_data)
            data_np[idx and np.isnan(data_np)] = median_group

    return data_np


def standardization(X):
    '''normalized the matrix X with n columns'''
    X = np.copy(X)
    if len(X.shape)>1:
        for i in range(X.shape[1]):
            X[:, i] = (X[:, i] - np.mean(X[:, i])) / (np.std(X[:, i]) + 0.000001)  # Z-standardization + a costant = 0.00001 in order to not get a division by zero
    else:
        X = (X - np.mean(X)) / (np.std(X) + 0.000001)  # Z-standardization + a costant = 0.00001 in order to not get a division by zero
    return X


def linearRegression(X, y):
    '''execute Linear Regression'''
    if len(y.shape) > 1:
        CONFbeta = np.dot(pinv(y.astype(None)), X)
        X_deconf = X - np.dot(y, CONFbeta)

    else:
        CONFbeta = np.dot(pinv([y.astype(None)]).T, X)
        X_deconf = X - np.dot(y[:, np.newaxis], CONFbeta)
    return X_deconf

def featuresSelectionLeftRight(X: pd.DataFrame):
    '''join with a mean function the features have Left and Right values'''
    col_names = X.columns
    copy = pd.DataFrame()
    idx_L = []
    idx_R = []
    i = -1

    # get index of columns with L and R
    for name in col_names:
        # print(name)
        i += 1
        if (' R ' in name):
            # print("R : {0}".format(i))
            idx_R.append(i)
        elif (' L' in name):
            # print("L : {0}".format(i))
            idx_L.append(i)
        else:
            continue
    # create new data frame for imaging data
    for j in range(0, X.shape[1]):
        if j in idx_R:
            c_R = X.iloc[:, j]
            c_L = X.iloc[:, j + 1]
            new_c = (c_R + c_L) / 2
            label = col_names[j]
            label = label[:-3]
            # print(label)
            copy[label] = new_c
        elif j in idx_L:
            continue
        else:
            # print(col_names[j])
            copy[col_names[j]] = X.iloc[:, j]

    return copy

def plot_settings():
    # sns.set_style('darkgrid')       # darkgrid, whitegrid, dark, white, ticks

    plt.rc('axes', titlesize=18)  # fontsize of the axes title
    plt.rc('axes', labelsize=14)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=11)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=11)  # fontsize of the tick labels
    plt.rc('legend', fontsize=13)  # legend fontsize
    plt.rc('font', size=13)  # controls default text sizes


def get_class_numbers(group,pdf):
    n_MCI = np.sum(group == 'MCI')
    n_EMCI = np.sum(group == 'EMCI')
    n_LMCI = np.sum(group == 'LMCI')
    n_AD = np.sum(group == 'AD')
    n_CN = np.sum(group == 'CN')
    n_SMC = np.sum(group == 'SMC')
    print("Subjects: \n" + str(n_MCI) + " MCI\n" + str(n_EMCI) + " EMCI\n" + str(n_LMCI) + " LMCI\n" + str(
        n_AD) + "AD\n" + str(n_CN) + " CN\n" + str(n_SMC) + " SMC")

    write("Subjects: \n" + str(n_MCI) + " MCI\n" + str(n_EMCI) + " EMCI\n" + str(n_LMCI) + " LMCI\n" + str(
        n_AD) + "AD\n" + str(n_CN) + " CN\n" + str(n_SMC) + " SMC\n", pdf)
    return pdf
