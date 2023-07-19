# required packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import lines
import statistics
from sklearn.utils import resample
from scipy.optimize import curve_fit
from scipy.ndimage.filters import gaussian_filter1d
import seaborn as sns

# import functions and scripts
from utils import generate_stimulus_timecourse, import_info, import_epochs, select_events, select_events_repetitionTrials, d_prime_perImgCat, estimate_first_pulse
from modelling_utils_paramInit import paramInit
from modelling_utils_fitObjective import model_csDN, model_DN, OF_ISI_recovery_log


"""

Author: A. Brands

"""

############################################################################################## ADAPT CODE HERE
##############################################################################################################
##############################################################################################################
##############################################################################################################

# define root directory
# dir = '/home/amber/OneDrive/code/nAdaptation_ECoG_git/'
dir = '/Users/a.m.brandsuva.nl/Library/CloudStorage/OneDrive-UvA/code/nAdaptation_ECoG_git/'

##############################################################################################################
##############################################################################################################
##############################################################################################################
##############################################################################################################


# import timepoints of on- and offset of stimulus for one and twopulse trials
t                         = np.loadtxt(dir+'variables/t.txt', dtype=float)
timepoints_onepulse       = np.loadtxt(dir+'variables/timepoints_onepulse.txt', dtype=int)
timepoints_twopulse       = np.loadtxt(dir+'variables/timepoints_twopulse.txt', dtype=int)
time_window               = np.loadtxt(dir+'variables/time_window.txt', dtype=int)
tempCond                  = np.loadtxt(dir+'variables/cond_temp.txt', dtype=float)
label_tempCond            = np.array(np.array(tempCond, dtype=int), dtype=str)

# get img. classes
stim_cat = np.loadtxt(dir+'variables/cond_stim.txt', dtype=str)

# determine confidence interval (error)
CI                  = 68
CI_low              = 50 - (0.5*CI)
CI_high             = 50 + (0.5*CI)
B_repetitions       = 1000

# retrieve parameters
model = 'csDN'
params_names, _, _, _ = paramInit(model)
sample_rate = 512

# parameters included in plot
param_names = ['tau1', 'tau2', 'n', 'sigma']
param_labels = [r'$\tau_{1}$', r'$\tau_{2}$', r'$n$', r'$\sigma$']

# labels
# visual areas (labels)
VA = ['V1-V3', 'VOTC', 'LOTC']
VA_n = np.zeros(len(VA))  # number of electrodes
colors_VA = [[233, 167, 0], [48, 64, 141], [187, 38, 102]]

# electrode coordinates
electrodes_visuallyResponsive = pd.read_csv(dir+'subject_data/electrodes_visuallyResponsive_manuallyAssigned.txt', header=0, index_col=0, delimiter=' ')
n_electrodes = len(electrodes_visuallyResponsive)

# extract electrode indices per visual area (i.e. V1-V3, LOTC, VOTC)
VA_name_idx_temp = {}
for i in range(n_electrodes):
    VA_name_current = electrodes_visuallyResponsive.loc[i, 'varea']
    if VA_name_current not in VA_name_idx_temp:
        VA_name_idx_temp[VA_name_current] = [i]
    else:
        VA_name_idx_temp[VA_name_current].append(i)
VA_name_idx = {}
VA_name_idx = {k: VA_name_idx_temp[k] for k in VA}
print(VA_name_idx, '\n')

# create stimulus timecourse
cond = [6]
trials = ['onepulse', 'twopulse_repeat']
stim_onepulse = np.zeros((len(tempCond), len(t))) 
stim_twopulse = np.zeros((len(tempCond), len(t))) 
for i in range(len(tempCond)):
    stim_onepulse[i, :] = generate_stimulus_timecourse(trials[0], i+1, dir)
    stim_twopulse[i, :] = generate_stimulus_timecourse(trials[1], i+1, dir)

    # initiate figure
fig = plt.figure(figsize=(18, 10))
gs = fig.add_gridspec(16, 22)
ax = dict()

ax['broadband_area_single']         = fig.add_subplot(gs[0:4, 0:6])
ax['broadband_area_E_single']       = fig.add_subplot(gs[0:4, 8:12])
ax['broadband_area_V_single']       = fig.add_subplot(gs[0:4, 13:17])
ax['broadband_area_L_single']       = fig.add_subplot(gs[0:4, 18:22])

ax['broadband_area_repeat']         = fig.add_subplot(gs[5:9, 0:6])
ax['broadband_area_E_repeat']       = fig.add_subplot(gs[5:9, 8:12])
ax['broadband_area_V_repeat']       = fig.add_subplot(gs[5:9, 13:17])
ax['broadband_area_L_repeat']       = fig.add_subplot(gs[5:9, 18:22])

ax['tau1']                          = fig.add_subplot(gs[12:16, 0:4])
ax['tau2']                          = fig.add_subplot(gs[12:16, 6:10])
ax['n']                             = fig.add_subplot(gs[12:16, 12:16])
ax['sigma']                         = fig.add_subplot(gs[12:16, 18:22])
ax_param                            = [ax['tau1'], ax['tau2'], ax['n'], ax['sigma']]

ax_broadband                        = [ax['broadband_area_single'], ax['broadband_area_repeat']]
ax_broadband_single                 = [ax['broadband_area_E_single'], ax['broadband_area_V_single'], ax['broadband_area_L_single']]
ax_broadband_repeat                 = [ax['broadband_area_E_repeat'], ax['broadband_area_V_repeat'], ax['broadband_area_L_repeat']]

# fontsizes
fontsize_tick                       = 20
fontsize_legend                     = 20
fontsize_label                      = 20
fontsize_title                      = 22

# linestyles
linestyle = ['solid', 'dashed']
alpha = 0.3
lw = 2

# seperate axes
sns.despine(offset=5)

############################################################################
############################################################### PLOT RESULTS - SINGLE

# initiate dataframes to store data
broadband_area = []
broadband_area_bootstrap = []

broadband_area_pred = []
broadband_area_nom_pred = []
broadband_area_denom_pred = []

broadband_area_pred_bootstrap = []
broadband_area_nom_pred_bootstrap = []
broadband_area_denom_pred_bootstrap = []

current_subject = ''
count_VA = 0
for key, value in VA_name_idx.items():

    # count number of electrodes
    n_electrodes = len(value)
    VA_n[count_VA] = n_electrodes

    # initiat dataframes
    broadband_area_current = np.zeros((n_electrodes, len(t)))
    broadband_area_bootstrap_current = np.zeros((B_repetitions, len(t)))

    # MODEL
    broadband_area_pred_current = np.zeros((n_electrodes, len(t)))
    broadband_area_nom_pred_current = np.zeros((n_electrodes, len(t))) 
    broadband_area_denom_pred_current = np.zeros((n_electrodes, len(t)))

    broadband_area_bootstrap_pred_current = np.zeros((B_repetitions, len(t)))
    broadband_area_nom_bootstrap_pred_current = np.zeros((B_repetitions, len(t))) 
    broadband_area_denom_bootstrap_pred_current = np.zeros((B_repetitions, len(t)))

    param_values_current = np.zeros((n_electrodes, len(param_labels)))

    # iterate over electrodes
    for i in range(n_electrodes):
    # for i in range(4):

        # retrieve info current electrode
        subject = electrodes_visuallyResponsive.subject[value[i]]
        electrode_name = electrodes_visuallyResponsive.electrode[value[i]]
        electrode_idx = int(electrodes_visuallyResponsive.electrode_idx[value[i]])

        # retrieve model parameters for current electrode
        temp = pd.read_csv(dir+'modelFit/visuallyResponsive/' + subject + '_' + electrode_name + '/param_' + model + '.txt', header=0, delimiter=' ', index_col=0)
        temp.reset_index(inplace=True,drop=True)
        params_current = list(temp.loc[0, params_names])
        for j in range(len(param_labels)):
            param_values_current[i, j] = temp.loc[0, params_names[j]]

        # print progress
        print(30*'-')
        print(key)
        print(30*'-')
        print('Computing trials for ' + subject + ', electrode ' + electrode_name + ' (' + str(i+1) + '/' + str(n_electrodes) + ')')

        if current_subject != subject:

            # update current subject
            current_subject = subject

            # import info
            _, events, channels, _ = import_info(subject, dir)

            # event index
            event_idx = select_events(events, 'TEMP', 'onepulse', dir)

            # import excluded trials
            excluded_epochs = pd.read_csv(dir+'subject_data/' + subject + '/excluded_epochs.txt', sep=' ', header=0, dtype=int)

        # extract data
        epochs_b = import_epochs(subject, electrode_idx, dir)
        index_epochs_b = [j for j in range(len(events)) if excluded_epochs.iloc[electrode_idx, j+1] == 1]
        epochs_b.iloc[:, index_epochs_b] = np.nan

        # select twouplse
        # NEURAL DATA
        broadband_area_current[i, :] = np.nanmean(epochs_b[event_idx[5]], axis=1)

        # MODEL
        data_temp = np.zeros((len(stim_cat), len(t)))
        data_nom_temp = np.zeros((len(stim_cat), len(t)))
        data_denom_temp = np.zeros((len(stim_cat), len(t)))
        for k in range(len(stim_cat)):
            _, data_temp[k, :], data_nom_temp[k, :], data_denom_temp[k, :] = model_csDN(stim_onepulse[5, :], trials[0], 5, stim_cat[k], sample_rate, params_current, dir, denom=True)
        broadband_area_pred_current[i, :] = np.mean(data_temp, 0)
        broadband_area_nom_pred_current[i, :] = np.mean(data_nom_temp, 0)
        broadband_area_denom_pred_current[i, :] = np.mean(data_denom_temp, 0)

        # correct for sigma (for illustration)
        sigma = float(temp.loc[:, 'sigma'])
        n = float(temp.loc[:, 'n'])
        broadband_area_denom_pred_current[i, :] = (np.mean(data_denom_temp, 0) - sigma**n)

    # plot parameters
    for i in range(len(param_labels)):

        # compute median and CI
        data_temp = param_values_current[:, i]
        n_samples = len(data_temp)
        medians = np.zeros(B_repetitions)
        for l in range(B_repetitions):
            boot = resample(data_temp, replace=True, n_samples=n_samples)
            medians[l]  = statistics.median(boot)
        avg_temp = statistics.median(medians)
        CI_temp = np.nanpercentile(medians, [CI_low, CI_high])

        # plot
        ax_param[i].plot([count_VA, count_VA], [CI_temp[0], CI_temp[1]], color=np.array(colors_VA[count_VA])/255, zorder=-1)
        ax_param[i].scatter(count_VA, avg_temp, color=np.array(colors_VA[count_VA])/255, edgecolor='white', s=150)
        ax_param[i].set_xlim(-0.5, 2.5)

    # perform bootstrap over broadband timecourse
    for i in range(B_repetitions):

        # draw random sample
        idx_temp = np.arange(n_electrodes)
        n_samples = len(idx_temp)
        boot = resample(idx_temp, replace=True, n_samples=n_samples)
        # boot = [0, 1, 2, 3]

        # retrieve broadband
        # NEURAL DATA
        data_temp = np.zeros((len(boot), len(t)))
        for l in range(len(boot)):
            data_temp[l, :] = broadband_area_current[boot[l], :]
        broadband_area_bootstrap_current[i, :] = gaussian_filter1d(np.mean(data_temp, 0), 10)
        
        # MODEL
        model_temp = np.zeros((len(boot), len(t)))
        model_nom_temp = np.zeros((len(boot), len(t)))
        model_denom_temp = np.zeros((len(boot), len(t)))
        for l in range(len(boot)):
            model_temp[l, :] = broadband_area_pred_current[boot[l], :]
            model_nom_temp[l, :] = broadband_area_nom_pred_current[boot[l], :]
            model_denom_temp[l, :] = broadband_area_denom_pred_current[boot[l], :]
        broadband_area_bootstrap_pred_current[i, :] = np.mean(model_temp, 0)
        broadband_area_nom_bootstrap_pred_current[i, :] = np.mean(model_nom_temp, 0)
        broadband_area_denom_bootstrap_pred_current[i, :] = np.mean(model_denom_temp, 0)

        # # retrieve broadband
        # # NEURAL DATA
        # broadband_area_bootstrap_current[i, :] = np.nanmean(broadband_area_current[boot, :], 0)
        
        # # MODEL
        # broadband_area_bootstrap_pred_current[i, :] = np.nanmean(broadband_area_pred_current[boot, :], 0)
        # broadband_area_nom_bootstrap_pred_current[i, :] = np.nanmean(broadband_area_nom_pred_current[boot, :], 0)
        # broadband_area_denom_bootstrap_pred_current[i, :] = np.nanmean(broadband_area_denom_pred_current[boot, :], 0)

    # append dataframes
    # NEURAL DATA
    broadband_area.append(broadband_area_current)
    broadband_area_bootstrap.append(broadband_area_bootstrap_current)

    # MODEL
    broadband_area_pred.append(broadband_area_pred_current)
    broadband_area_nom_pred.append(broadband_area_nom_pred_current)
    broadband_area_denom_pred.append(broadband_area_denom_pred_current)

    broadband_area_pred_bootstrap.append(broadband_area_bootstrap_pred_current)
    broadband_area_nom_pred_bootstrap.append(broadband_area_nom_bootstrap_pred_current)
    broadband_area_denom_pred_bootstrap.append(broadband_area_denom_bootstrap_pred_current)
    
    # increment count
    count_VA+=1

# plot stimulus timecourse
ax_broadband[0].axvspan(t[timepoints_onepulse[cond[0]-1, 0]], t[timepoints_onepulse[cond[0]-1, 1]], facecolor='grey', alpha=0.2, label='stimulus')

for i in range(len(VA)):

    # plot stimulus timecourse
    ax_broadband_single[i].axvspan(t[timepoints_onepulse[cond[0]-1, 0]], t[timepoints_onepulse[cond[0]-1, 1]], facecolor='grey', alpha=0.2, label='stimulus')

    # NEURAL DATA
    # data_temp = np.mean(broadband_area_bootstrap[i], axis=0)
    # ax_broadband[0].plot(t, data_temp/max(data_temp), color=np.array(colors_VA[i])/255, alpha=0.2)

    # MODEL - ALL
    model_temp = np.mean(broadband_area_pred_bootstrap[i], axis=0)
    ax_broadband[0].plot(t, model_temp/max(model_temp), color=np.array(colors_VA[i])/255)

    # MODEL - NOM
    nom_temp = np.mean(broadband_area_nom_pred[i], 0)
    ax_broadband_single[i].plot(t, nom_temp/max(nom_temp), color=np.array(colors_VA[i])/255, linestyle=linestyle[0], alpha=alpha)

    # MODEL - DENOM
    denom_temp = np.mean(broadband_area_denom_pred[i], 0)
    ax_broadband_single[i].plot(t, denom_temp/max(denom_temp), color=np.array(colors_VA[i])/255, linestyle=linestyle[1], lw=lw)


############################################################################
############################################################### PLOT RESULTS - REPEATED

# initiate dataframes to store data
broadband_area = []
broadband_area_bootstrap = []

broadband_area_pred = []
broadband_area_nom_pred = []
broadband_area_denom_pred = []

broadband_area_pred_bootstrap = []
broadband_area_nom_pred_bootstrap = []
broadband_area_denom_pred_bootstrap = []

current_subject = ''
count_VA = 0
for key, value in VA_name_idx.items():

    # count number of electrodes
    n_electrodes = len(value)
    VA_n[count_VA] = n_electrodes

    # initiat dataframes
    broadband_area_current = np.zeros((n_electrodes, len(t)))
    broadband_area_bootstrap_current = np.zeros((B_repetitions, len(t)))

    # MODEL
    broadband_area_pred_current = np.zeros((n_electrodes, len(t)))
    broadband_area_nom_pred_current = np.zeros((n_electrodes, len(t))) 
    broadband_area_denom_pred_current = np.zeros((n_electrodes, len(t)))

    broadband_area_bootstrap_pred_current = np.zeros((B_repetitions, len(t)))
    broadband_area_nom_bootstrap_pred_current = np.zeros((B_repetitions, len(t))) 
    broadband_area_denom_bootstrap_pred_current = np.zeros((B_repetitions, len(t)))

    # iterate over electrodes
    for i in range(n_electrodes):
    # for i in range(4):

        # retrieve info current electrode
        subject = electrodes_visuallyResponsive.subject[value[i]]
        electrode_name = electrodes_visuallyResponsive.electrode[value[i]]
        electrode_idx = int(electrodes_visuallyResponsive.electrode_idx[value[i]])

        # retrieve model parameters for current electrode
        temp = pd.read_csv(dir+'modelFit/visuallyResponsive/' + subject + '_' + electrode_name + '/param_' + model + '.txt', header=0, delimiter=' ', index_col=0)
        temp.reset_index(inplace=True,drop=True)
        params_current = list(temp.loc[0, params_names])

        # print progress
        print(30*'-')
        print(key)
        print(30*'-')
        print('Computing trials for ' + subject + ', electrode ' + electrode_name + ' (' + str(i+1) + '/' + str(n_electrodes) + ')')

        if current_subject != subject:

            # update current subject
            current_subject = subject

            # import info
            _, events, channels, _ = import_info(subject, dir)

            # event index
            event_idx = select_events(events, 'TEMP', 'twopulse_repeat', dir)

            # import excluded trials
            excluded_epochs = pd.read_csv(dir+'subject_data/' + subject + '/excluded_epochs.txt', sep=' ', index_col=0, header=0, dtype=int)

        # extract data
        epochs_b = import_epochs(subject, electrode_idx, dir)
        index_epochs_b = [j for j in range(len(events)) if excluded_epochs.iloc[electrode_idx, j] == 1]
        epochs_b.iloc[:, index_epochs_b] = np.nan

        # select twouplse
        # NEURAL DATA
        broadband_area_current[i, :] = np.nanmean(epochs_b[event_idx[cond[0]-1]], axis=1)

        # MODEL
        data_temp = np.zeros((len(stim_cat), len(t)))
        data_nom_temp = np.zeros((len(stim_cat), len(t)))
        data_denom_temp = np.zeros((len(stim_cat), len(t)))
        for k in range(len(stim_cat)):
            _, data_temp[k, :], data_nom_temp[k, :], data_denom_temp[k, :] = model_csDN(stim_twopulse[5, :], trials[1], 5, stim_cat[k], sample_rate, params_current, dir, denom=True)
        broadband_area_pred_current[i, :] = np.mean(data_temp, 0)
        broadband_area_nom_pred_current[i, :] = np.mean(data_nom_temp, 0)
        broadband_area_denom_pred_current[i, :] = np.mean(data_denom_temp, 0)

        # correct for sigma (for illustration)
        sigma = float(temp.loc[:, 'sigma'])
        n = float(temp.loc[:, 'n'])            
        broadband_area_denom_pred_current[i, :] = (np.mean(data_denom_temp, 0) - sigma**n)

    # perform bootstrap over broadband timecourse
    for i in range(B_repetitions):

        # draw random sample
        idx_temp = np.arange(n_electrodes)
        n_samples = len(idx_temp)
        boot = resample(idx_temp, replace=True, n_samples=n_samples)
        # boot = [0, 1, 2, 3]

        # retrieve broadband
        # NEURAL DATA
        data_temp = np.zeros((len(boot), len(t)))
        for l in range(len(boot)):
            data_temp[l, :] = broadband_area_current[boot[l], :]
        broadband_area_bootstrap_current[i, :] = gaussian_filter1d(np.mean(data_temp, 0), 10)
        
        # MODEL
        model_temp = np.zeros((len(boot), len(t)))
        model_nom_temp = np.zeros((len(boot), len(t)))
        model_denom_temp = np.zeros((len(boot), len(t)))
        for l in range(len(boot)):
            model_temp[l, :] = broadband_area_pred_current[boot[l], :]
            model_nom_temp[l, :] = broadband_area_nom_pred_current[boot[l], :]
            model_denom_temp[l, :] = broadband_area_denom_pred_current[boot[l], :]
        broadband_area_bootstrap_pred_current[i, :] = np.mean(model_temp, 0)
        broadband_area_nom_bootstrap_pred_current[i, :] = np.mean(model_nom_temp, 0)
        broadband_area_denom_bootstrap_pred_current[i, :] = np.mean(model_denom_temp, 0)

    # append dataframes
    # NEURAL DATA
    broadband_area.append(broadband_area_current)
    broadband_area_bootstrap.append(broadband_area_bootstrap_current)

    # MODEL
    broadband_area_pred.append(broadband_area_pred_current)
    broadband_area_nom_pred.append(broadband_area_nom_pred_current)
    broadband_area_denom_pred.append(broadband_area_denom_pred_current)

    broadband_area_pred_bootstrap.append(broadband_area_bootstrap_pred_current)
    broadband_area_nom_pred_bootstrap.append(broadband_area_nom_bootstrap_pred_current)
    broadband_area_denom_pred_bootstrap.append(broadband_area_denom_bootstrap_pred_current)
    
    # increment count
    count_VA+=1

# plot stimulus timecourse
ax_broadband[1].axvspan(t[timepoints_twopulse[cond[0]-1, 0]], t[timepoints_twopulse[cond[0]-1, 1]], facecolor='grey', alpha=0.2, label='stimulus')
ax_broadband[1].axvspan(t[timepoints_twopulse[cond[0]-1, 2]], t[timepoints_twopulse[cond[0]-1, 3]], facecolor='grey', alpha=0.2, label='stimulus')

for i in range(len(VA)):

    # plot stimulus timecourse
    ax_broadband_repeat[i].axvspan(t[timepoints_twopulse[cond[0]-1, 0]], t[timepoints_twopulse[cond[0]-1, 1]], facecolor='grey', alpha=0.2, label='stimulus')
    ax_broadband_repeat[i].axvspan(t[timepoints_twopulse[cond[0]-1, 2]], t[timepoints_twopulse[cond[0]-1, 3]], facecolor='grey', alpha=0.2, label='stimulus')

    # # NEURAL DATA
    # data_temp = np.mean(broadband_area_bootstrap[i], axis=0)
    # ax_broadband[0].plot(t, data_temp/max(data_temp), color=np.array(colors_VA[i])/255, alpha=0.2)

    # MODEL - ALL
    model_temp = np.mean(broadband_area_pred_bootstrap[i], axis=0)
    ax_broadband[1].plot(t, model_temp/max(model_temp), color=np.array(colors_VA[i])/255)

    # MODEL - NOM
    nom_temp = np.mean(broadband_area_nom_pred[i], 0)
    ax_broadband_repeat[i].plot(t, nom_temp/max(nom_temp), color=np.array(colors_VA[i])/255, linestyle=linestyle[0], alpha=alpha)

    # MODEL - DENOM
    denom_temp = np.mean(broadband_area_denom_pred[i], 0)
    ax_broadband_repeat[i].plot(t, denom_temp/max(denom_temp), color=np.array(colors_VA[i])/255, linestyle=linestyle[1], lw=lw)


for i in range(len(ax_broadband)):
    ax_broadband[i].spines['top'].set_visible(False)
    ax_broadband[i].spines['right'].set_visible(False)
    ax_broadband[i].axhline(0, color='grey', lw=0.5, alpha=0.5)
    ax_broadband[i].tick_params(axis='both', which='major', labelsize=fontsize_tick)
    # ax_broadband[i].set_ylabel('Change in power\n(x-fold)', fontsize=fontsize_label)
    # ax_broadband[i].set_xlabel('Time (ms)', fontsize=fontsize_label)

for i in range(len(ax_broadband_single)):
    ax_broadband_single[i].spines['top'].set_visible(False)
    ax_broadband_single[i].spines['right'].set_visible(False)
    ax_broadband_single[i].axhline(0, color='grey', lw=0.5, alpha=0.5)
    ax_broadband_single[i].tick_params(axis='both', which='major', labelsize=fontsize_tick)
    # ax_broadband_single[i].set_xlabel('Time (ms)')
    # if i == 0:
    #     ax_broadband_single[i].set_ylabel('Model prediction', fontsize=fontsize_label)
    # else:
    #     ax_broadband_single[i].set_xticks([])

for i in range(len(ax_broadband_repeat)):
    ax_broadband_repeat[i].spines['top'].set_visible(False)
    ax_broadband_repeat[i].spines['right'].set_visible(False)
    ax_broadband_repeat[i].axhline(0, color='grey', lw=0.5, alpha=0.5)
    ax_broadband_repeat[i].tick_params(axis='both', which='major', labelsize=fontsize_tick)
    # ax_broadband_repeat[i].set_xlabel('Time (ms)', fontsize=fontsize_label)
    # if i == 0:
    #     ax_broadband_repeat[i].set_ylabel('Model prediction', fontsize=fontsize_label)
    # else:
    #     ax_broadband_repeat[i].set_xticks([])

for i in range(len(ax_param)):
    ax_param[i].set_title(param_labels[i], fontsize=fontsize_title)
    ax_param[i].tick_params(axis='both', which='major', labelsize=fontsize_tick)
    ax_param[i].set_xticks(np.arange(len(VA)))
    ax_param[i].set_xticklabels(VA, fontsize=fontsize_tick, rotation=45)

# save figure    
fig.align_ylabels()
# save figure
plt.tight_layout()
plt.savefig(dir+'mkFigure/Fig7.svg', format='svg', bbox_inches='tight')
plt.savefig(dir+'mkFigure/Fig7', dpi=300, bbox_inches='tight')
# plt.show()

