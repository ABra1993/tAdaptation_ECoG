# required packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import seaborn as sns

from utils import import_info, select_events, select_electrodes, import_epochs, recovery_perISI
from utils import generate_stimulus_timecourse, r_squared

from utils import generate_stimulus_timecourse, import_info, import_epochs, select_events
from modelling_utils_paramInit import paramInit
from modelling_utils_fitObjective import model

"""

Author: A. Brands

"""

############################################################################################## ADAPT CODE HERE
##############################################################################################################
##############################################################################################################
##############################################################################################################

# define root directory
file = open('setDir.txt')
dir = file.readline().strip('\n')
print(dir)

##############################################################################################################
##############################################################################################################
##############################################################################################################
##############################################################################################################

# subject info
# subject             = 'sub-p14'
# electrode_name      = 'LT02'

subject             = 'sub-p12'
electrode_name      = 'G01'

# models
model_type = 'DN'

# scaling
scaling = ['none', 'S']
scaling_color = ['red', 'crimson']

sample_rate = 512

# create stimulus timecourse
stim = generate_stimulus_timecourse('twopulse_repeat', 5, dir)

# import timepoints of on- and offset of stimulus for one and twopulse trials
t                         = np.loadtxt(dir+'variables/t.txt', dtype=float)
timepoints_onepulse       = np.loadtxt(dir+'variables/timepoints_onepulse.txt', dtype=int)
time_window               = np.loadtxt(dir+'variables/time_window.txt', dtype=int)
tempCond                  = np.loadtxt(dir+'variables/cond_temp.txt', dtype=float)
label_tempCond            = np.array(np.array(tempCond, dtype=int), dtype=str)

# get img. classes
stim_cat                  = np.loadtxt(dir+'variables/cond_stim.txt', dtype=str)

# plot cross validation
fig, axs = plt.subplots(2, 5, figsize=(16, 5))

# seperate axes
sns.despine(offset=10)

# set fontsizes
fontsize_tick           = 15
fontsize_legend         = 15
fontsize_label          = 15
fontsize_title          = 18

lw = 2

# import info
_, events, channels, _ = import_info(subject, dir)

# import excluded trials
excluded_epochs = pd.read_csv(dir+'data_subjects/' + subject + '/excluded_epochs.txt', sep=' ', index_col=0, header=0, dtype=int)

# extract broadband data
electrode_idx = select_electrodes(channels, electrode_name)
epochs_b = import_epochs(subject, electrode_idx, dir)
index_epochs_b = [j for j in range(len(events)) if excluded_epochs.iloc[electrode_idx, j] == 1]
epochs_b.iloc[:, index_epochs_b] = np.nan

# select electrode(s) and events
event_idx = select_events(events, 'both', 'twopulse_repeat', dir)

for iS in range(len(scaling)):

    # retrieve parameters
    params_names, _, _, _ = paramInit(model_type, scaling[iS])

    # retrieve model parameters for current electrode
    temp = pd.read_csv(dir+'modelFit/visuallyResponsive/' + subject + '_' + electrode_name + '/param_' + model_type + '_' + scaling[iS] + '.txt', header=0, delimiter=' ', index_col=0)
    temp.reset_index(inplace=True,drop=True)
    params_current = list(temp.loc[0, params_names])

    for j in range(len(stim_cat)-1):

        print('Model: ', scaling[iS], ', cat.: ', stim_cat[j])

        # plot stimulus
        axs[iS, j].plot(t, stim, color='powderblue', label='Stimulus', lw=lw)

        # plot timecourse
        data = np.nanmean(epochs_b[event_idx[j][5]], 1)
        axs[iS, j].plot(t, data, color='black', label='Neural data', lw=lw)

        # plot model timecourse
        if scaling[iS] == 'none':
            pred = model(model_type, scaling, stim, sample_rate, params_current, dir)
            # pred = model_DN(stim, sample_rate, params_current)
        elif scaling[iS] == 'S':
            pred = model(model_type, scaling, stim, sample_rate, params_current, dir, 'twopulse_repeat', 5, stim_cat[j])
            # _, pred = model_csDN(stim, , sample_rate, params_current, dir)
        axs[iS, j].plot(t, pred, color=scaling_color[iS], label=model_type + ' model', lw=lw)

        # compute coefficient of variation
        r_2 = r_squared(data, pred)

        # adjust axis
        axs[iS, j].set_ylim(-0.5, 12)
        axs[iS, j].tick_params(axis='both', which='major', labelsize=fontsize_tick)
        # if j == 0:
            # axs[i, j].set_ylabel('Change in broadband power', fontsize=fontsize_label)
            # axs[i, j].legend(fontsize=fontsize_legend)
        # if i == (len(models))-1:
        #     axs[i, j].set_xlabel('Time (s)', fontsize=fontsize_label)
        # if i == 0:
        #     axs[i, j].set_title(stim_cat[j] + '\n' + r' $R^{2}$: ' + str(np.round(r_2, 2)), fontsize=fontsize_title)
        # else:
        axs[iS, j].set_title(r' $R^{2}$: ' + str(np.round(r_2, 2)), fontsize=fontsize_title)


# save figure
plt.tight_layout()
plt.savefig(dir+'/mkFigure/SuppFig9.svg', format='svg')
plt.savefig(dir+'/mkFigure/SuppFig9') 
# plt.show()

