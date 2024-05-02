# required packages
import numpy as np
import time
import matplotlib.pyplot as plt
import multiprocessing
import pandas as pd
import sys
import random
import scipy.optimize as optimize

from modelling_utils_paramInit import paramInit
from utils import generate_stimulus_timecourse, r_squared
from modelling_utils_fitObjective import objective, model

"""
Author: A. Brands

Description: Cross-validation of the model (Fit-then-average procedure, where parameters are fitter per electrode).

"""

# define root directory
file = open('setDir.txt')
dir = file.readline()
dir = dir[:-1]

##### SPECIFY ELECTRODE TYPE
# electrode_type = 'visuallyResponsive'
electrode_type = 'categorySelective'

# import variables
img_cat     = np.loadtxt(dir+'variables/cond_stim.txt', dtype=str)
temp_cond   = np.loadtxt(dir+'variables/cond_temp.txt', dtype=int)
t           = np.loadtxt(dir+'variables/t.txt', dtype=float)
t_index     = np.arange(len(t)).astype(str).tolist()

# import electrodes
if electrode_type == 'visuallyResponsive':
    electrodes = pd.read_csv(dir+'data_subjects/electrodes_visuallyResponsive_manuallyAssigned.txt', header=0, index_col=0, delimiter=' ')
elif electrode_type == 'categorySelective':
     electrodes = pd.read_csv(dir+'data_subjects/electrodes_categorySelective_0-5.txt', header=0, index_col=0, delimiter=' ')

# # model fit for a single electrode
# electrodes = electrodes[(electrodes.subject == 'sub-p14') & (electrodes.electrode == 'LO01')]
# electrodes.reset_index(drop=True, inplace=True)

# count electrodes
n_electrodes = len(electrodes)
# n_electrodes = 1

# type of trials
trial_type = ['onepulse', 'twopulse_repeat']

# r square
r2_per_fold = ['r2_1', 'r2_2', 'r2_3', 'r2_4', 'r2_5', 'r2_6', 'r2_7', 'r2_8', 'r2_9', 'r2_10', 'r2_11', 'r2_12']
r2_per_test = ['r2_1', 'r2_2', 'r2_3', 'r2_4', 'r2_5', 'r2_6']

# model hyperparameters
max_nfev        = 500
sample_rate     = 512
nFolds          = 12

# set model parameters
model_type = 'DN'
# model_type = 'TTC17'
# model_type = 'TTC19'
if model_type not in ['DN', 'TTC17', 'TTC19']:
    sys.exit('\n Model does not exist, choose one of the following ["DN", "TTC17", "TTC19"] \n')

# scaling
# scaling = 'none'
# scaling = 'S'
scaling = 'P'

print(30*'-')
print(30*'-')
print('Model ', model_type, ' with scaling: ', scaling)
print(30*'-')
print(30*'-')

# retrieve initial values of parameters
params_names, x0, lb, ub = paramInit(model_type, scaling)
print('Parameters that will be fitted: ', params_names)

def model_fit(i):

    # defines electrode and subject
    subject         = electrodes.loc[i, 'subject']
    electrode_name  = electrodes.loc[i, 'electrode']
    electrode_idx   = int(electrodes.loc[i, 'electrode_idx'])

    # # print progress
    # print('Performing cross-validation for ', subject, ': ', electrode_name, '...')

    # initiate dataframe for performance per fold
    r_sq_pd = pd.DataFrame()
    r_sq_pd.loc[0, 'subject'] = subject
    r_sq_pd.loc[0, 'electrode_idx'] = int(electrode_idx)
    r_sq_pd.loc[0, 'electrode'] = electrode_name
    r_sq_pd.loc[0, 'r2'] = 0
    for k in range(nFolds):
        r_sq_pd.loc[0, 'r2_' + str(k + 1)] = 0

    # initiate dataframe for saving r2 per test sample for each fold
    r_sq_fold_pd = pd.DataFrame()
    r_sq_fold_pd['fold'] = np.arange(1, nFolds+1)
    r_sq_fold_pd['r2'] = np.zeros(nFolds)
    for k in range(len(r2_per_test)):
        r_sq_fold_pd[r2_per_test[k]] = np.zeros(nFolds)

    # initiate dataframe for parameter values of full fit
    param_pd = pd.DataFrame()
    param_pd.loc[0, 'subject'] = subject
    param_pd.loc[0, 'electrode_idx'] = int(electrode_idx)
    param_pd.loc[0, 'electrode'] = electrode_name
    param_pd.loc[0, 'r2'] = 0
    for k in range(len(params_names)):
        param_pd.loc[0, params_names[k]] = 0
    
    # import data to be fitted
    y = pd.read_csv(dir+'modelFit/' + electrode_type + '/' + subject + '_' + electrode_name + '/data.txt', sep=' ', header=0)
    # print(y)

    # generate stimulus time courses for each sample
    stim = y.copy()
    for j in range(len(stim)):
        stim.loc[j, t_index] = generate_stimulus_timecourse(stim.loc[j, 'trial_type'], int(stim.loc[j, 'temp_cond']), dir)
    
    # retrieve indices per test sample per image category
    y_idx_per_cat = []
    for j in range(len(img_cat)):
        temp = y[y.img_cat == img_cat[j]].index.tolist()
        y_idx_per_cat.append(temp)
    
    # train full model to get optimal parameters
    X_full = np.array(stim.loc[:, t_index])
    y_full = np.array(y.loc[:, t_index])

    # fit model
    info = y.loc[:, ['trial_type', 'temp_cond', 'img_cat']]
    np.seterr(divide='ignore', invalid='ignore') # suppress printing division errors
    res = optimize.least_squares(objective, x0, args=(model_type, scaling, X_full, y_full, sample_rate, dir, t, info), max_nfev=max_nfev, bounds=(lb, ub))

    # print progress
    print(30*'-')
    print('Fitted params for electrode: ' + electrode_name + ' (' + subject + '):')
    print('(model: ' + model_type.capitalize() + ' with scaling ' + scaling + ')\n')
    for k in range(len(params_names)):
        print(params_names[k] + ': ' + str(res.x[k]))
    print(30*'-')

    # plot model fit for first six image categories
    fig, axs = plt.subplots(1, 6, figsize=(15, 3))
    fig.suptitle('Model ' + model_type + ' with scaling: ' + scaling)
    min_value = np.amin(np.array(y.loc[:, t_index]))
    max_value = np.amax(np.array(y.loc[:, t_index]))
    for j in range(6): # plots one fitted sample per image category (onepulse, ISI of 533 ms)
        
        # select sample
        idx_plot = y_idx_per_cat[j][-1]

        # stimulus timecourse
        stim_plot = X_full[idx_plot, :]
        axs[j].plot(stim_plot, color='blue', alpha=0.2, label='Stimulus')

        # model prediction
        prediction = model(model_type, scaling, stim_plot, sample_rate, res.x, dir, y.loc[idx_plot, 'trial_type'], y.loc[idx_plot, 'temp_cond'], y.loc[idx_plot, 'img_cat'])

        # plot prediction
        axs[j].plot(prediction, 'r', label='Model')

        # compute coefficient of variation
        r_2 = r_squared(y_full[idx_plot, :], prediction)

        # adjust axes
        if j != 0:
            axs[j].set_yticks([])
        else:
            axs[j].set_ylabel('Broadband power', fontsize=8)
        axs[j].set_xlabel('Model timesteps', fontsize=8)
        axs[j].plot(y_full[idx_plot, :], 'k', label='Neural data')
        axs[j].set_title(y.loc[idx_plot, 'trial_type'] + ', ' + str(temp_cond[int(y.loc[idx_plot, 'temp_cond'])]) + 'ms ,\n' + y.loc[idx_plot, 'img_cat'] + r' ($R^{2}$: ' + str(np.round(r_2, 2)) + ')', fontsize=8)

        # data
        axs[j].set_ylim(min_value, max_value+0.05*max_value)

    axs[0].legend(fontsize=6)
    plt.tight_layout()
    plt.savefig(dir+'mkFigure/modelFit/' + electrode_type + '/' + subject + '_' + electrode_name + '_modelFit_' + model_type + '_' + scaling)
    plt.close()

    # print
    print('Full model fitted, figure generated! ')

    # fit full model
    param_pd.loc[0, params_names] = res.x

    # nfold cv
    r_sq_temp = np.zeros((nFolds)) 
    param_pd_temp = np.zeros((nFolds, len(params_names)))
    for fold_idx in range(nFolds): # cross-validate leave-one-out approach

        # get test indices (one sample per image category) and create train data (pseudo-random held-out data)
        test_idx = []
        for j in range(len(img_cat)):
            test_idx.append(random.sample(y_idx_per_cat[j], 1)[0])          # choose random test sample
            y_idx_per_cat[j].remove(test_idx[j])                            # remove from test indices

        # TEST indices
        train_idx = np.arange(len(stim)).tolist()
        train_idx.remove(test_idx[0])
        train_idx.remove(test_idx[1])
        train_idx.remove(test_idx[2])
        train_idx.remove(test_idx[3])
        train_idx.remove(test_idx[4])
        train_idx.remove(test_idx[5])

        # select training and test data
        X_train = np.array(stim.loc[train_idx, t_index])
        y_train = np.array(y.iloc[train_idx, 2:2+len(t_index)])

        X_test = np.array(stim.loc[test_idx, t_index])
        y_test = np.array(y.iloc[test_idx, 2:2+len(t_index)])

        # fit model
        info = y.loc[train_idx, ['trial_type', 'temp_cond', 'img_cat']]
        info.reset_index(inplace=True, drop=True)
        np.seterr(divide='ignore', invalid='ignore') # inhibit printing division errors
        res = optimize.least_squares(objective, x0, args=(model_type, scaling, X_train, y_train, sample_rate, dir, t, info), max_nfev=max_nfev, bounds=(lb, ub))

        # retrieve parameters
        popt = res.x

        # print progress
        print(30*'-')
        print('Fitted params for electrode: ' + electrode_name + ' (' + subject + '):')
        print('(model: ' + model_type.capitalize() + ' with scaling ' + scaling + ')\n')
        for k in range(len(params_names)):
            print(params_names[k] + ': ' + str(popt[k]))
        param_pd_temp[fold_idx, :] = popt
        print(30*'-')

        # test on held-out set
        r_sq_per_test = np.zeros(len(X_test))
        for k in range(len(X_test)):

            # cross-validate
            pred_temp = model(model_type, scaling, X_test[k], sample_rate, res.x, dir, stim.loc[test_idx[k], 'trial_type'], int(stim.loc[test_idx[k], 'temp_cond']), stim.loc[test_idx[k], 'img_cat'])

            # compute explained variance
            temp = r_squared(y_test[k], pred_temp)
            r_sq_per_test[k] = np.round(temp, 3)
            r_sq_fold_pd.loc[fold_idx, r2_per_fold[k]] = np.round(temp, 3)            

        # save cv r2
        r_sq_temp[fold_idx] = np.round(np.mean(r_sq_per_test), 4)

        # print progress
        print('Fold ' + str(fold_idx+1) + ': R2 for (', subject + ',', electrode_name, ') held-out data is', np.round(r_sq_temp[fold_idx], 2), '.')

        # update fold
        r_sq_pd.loc[0, r2_per_fold[fold_idx]] = np.round(np.mean(r_sq_per_test), 4)
        r_sq_fold_pd.loc[fold_idx, 'r2'] = np.round(np.mean(r_sq_per_test), 4)

    # average prediction for held-out set
    r_sq_mean = np.round(np.mean(r_sq_temp), 4)
    r_sq_pd.loc[:, 'r2'] = r_sq_mean
    param_pd.loc[:, 'r2'] = r_sq_mean
    
    r_sq_pd.to_csv(dir+'modelFit/' + electrode_type + '/' + subject + '_' + electrode_name + '/r_sq_' + model_type + '_' + scaling + '.txt', sep=' ', header=True, index=False)
    r_sq_fold_pd.to_csv(dir+'modelFit/' + electrode_type + '/' + subject + '_' + electrode_name + '/r_sq_per_fold_' + model_type + '_' + scaling + '.txt', sep=' ', header=True, index=False)
    param_pd.to_csv(dir+'modelFit/' + electrode_type + '/' + subject + '_' + electrode_name + '/param_' + model_type + '_' + scaling + '.txt', sep=' ', header=True, index=False)

    # print progress
    print('\n')
    print(60*'#')
    print('Done! R2 for (', subject + ',', electrode_name, ') test data is', np.round(np.mean(r_sq_temp), 2), '.')
    print(60*'#', '\n')

if __name__ == '__main__':
    pool = multiprocessing.Pool()
    start_time = time.perf_counter()
    processes = [pool.apply_async(model_fit, args=(x,)) for x in range(n_electrodes)]
    result = [p.get() for p in processes]
    finish_time = time.perf_counter()
    print(f"Program finished in {finish_time-start_time} seconds\n")
    print(30*'-')
    print(30*'-')
    print('Model ', model_type, ' with scaling: ', scaling)
    print(30*'-')
    print(30*'-')
