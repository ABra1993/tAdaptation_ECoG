# required packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statistics
from sklearn.utils import resample
from scipy.ndimage import gaussian_filter1d
import seaborn as sns

# import functions and scripts
from utils import generate_stimulus_timecourse, import_info, import_epochs, select_events, select_events_durationTrials, d_prime_perImgCat
from modelling_utils_paramInit import paramInit
# from models.Models_csDN import Models_csDN
# from models.Models_DN import Models_DN
from modelling_utils_fitObjective import model_csDN, model_DN

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

# specifiy the trial types
img_type = 'all'
# img_type = 'preferred'
# img_type = 'nonpreferred'

##############################################################################################################
##############################################################################################################
##############################################################################################################
##############################################################################################################

# assign condition
if img_type == 'all':
    preference = 0
elif img_type == 'preferred':
    preference = 1
elif img_type == 'nonpreferred':
    preference = 2

# import timepoints of on- and offset of stimulus for one and twopulse trials
t                         = np.loadtxt(dir+'variables/t.txt', dtype=float)
timepoints_onepulse       = np.loadtxt(dir+'variables/timepoints_onepulse.txt', dtype=int)
time_window               = np.loadtxt(dir+'variables/time_window.txt', dtype=int)
tempCond                  = np.loadtxt(dir+'variables/cond_temp.txt', dtype=float)
label_tempCond            = np.array(np.array(tempCond, dtype=int), dtype=str)

# get img. classes
stim_cat                  = np.loadtxt(dir+'variables/cond_stim.txt', dtype=str)

# create stimulus timecourse
stim_onepulse = np.zeros((len(tempCond), len(t))) 
for i in range(len(tempCond)):
    stim_onepulse[i, :] = generate_stimulus_timecourse('onepulse', i, dir)

# define model
# model = 'DN'
model = 'csDN'

# retrieve parameters
params_names, _, _, _ = paramInit(model)
sample_rate = 512

# determine confidence interval for plotting
CI                  = 68
CI_low              = 50 - (0.5*CI)
CI_high             = 50 + (0.5*CI)
B_repetitions       = 1000

# info visual areas
VA                  = ['V1-V3', 'VOTC', 'LOTC']
colors_VA           = [[233, 167, 0], [48, 64, 141], [187, 38, 102]]
VA_n                = np.zeros(len(VA))

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

############################################################################
############################################################## DATA ANALYSIS

# initiate dataframes to store extracted data
broadband = []
broadband_pred = []

broadband_bootstrap = []
broadband_pred_bootstrap = []

# summary metrics
ttp = []
ttp_avg = np.zeros(len(VA)) # over medians of samples
ttp_CI = np.zeros((len(VA), 2))

ttp_pred = []
ttp_avg_pred = np.zeros(len(VA)) # over medians of samples
ttp_CI_pred = np.zeros((len(VA), 2)) 

fwhm = []
fwhm_avg = np.zeros((len(VA), len(tempCond))) # over medians of samples
fwhm_CI = np.zeros((len(VA), len(tempCond), 2))

fwhm_pred = []
fwhm_avg_pred = np.zeros((len(VA), len(tempCond))) # over medians of samples
fwhm_CI_pred = np.zeros((len(VA), len(tempCond), 2))

# save medians for statistical testing
ttp_medians = np.zeros((len(VA), B_repetitions))
ttp_pred_medians = np.zeros((len(VA), B_repetitions))

fwhm_medians = np.zeros((len(VA), len(tempCond), B_repetitions))
fwhm_pred_medians = np.zeros((len(VA), len(tempCond), B_repetitions))

# for statistical testing
current_subject = ''
count_VA = 0
for key, value in VA_name_idx.items():

    # count number of electrodes
    n_electrodes = len(value)
    VA_n[count_VA] = n_electrodes

    # initiat dataframes
    broadband_current = np.zeros((n_electrodes, len(tempCond), len(t)))
    broadband_pred_current = np.zeros((n_electrodes, len(tempCond), len(t)))

    broadband_bootstrap_current = np.zeros((B_repetitions, len(tempCond), len(t)))
    broadband_pred_bootstrap_current = np.zeros((B_repetitions, len(tempCond), len(t)))

    ttp_current = np.zeros((B_repetitions))
    ttp_pred_current = np.zeros((B_repetitions))

    fwhm_current = np.zeros(((B_repetitions), len(tempCond)))
    fwhm_pred_current = np.zeros(((B_repetitions), len(tempCond)))

    # iterate over electrodes
    for i in range(n_electrodes):

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

            # import excluded trials
            excluded_epochs = pd.read_csv(dir+'subject_data/' + subject + '/excluded_epochs.txt', sep=' ', index_col=0, header=0, dtype=int)
        
        # determine preferred image category
        epochs_b = import_epochs(subject, electrode_idx, dir)
        index_epochs_b = [j for j in range(len(events)) if excluded_epochs.iloc[electrode_idx, j] == 1]
        epochs_b.iloc[:, index_epochs_b] = np.nan

        # extract data
        if (img_type == 'preferred') | (img_type == 'nonpreferred'):

            # determine category selectivity
            event_idx_onepulse = select_events(events, 'STIM', 'onepulse', dir)
            d_prime_temp = d_prime_perImgCat(epochs_b, event_idx_onepulse, stim_cat).tolist() # double-check for/confirm img. category selection

            preferred_cat_index = np.argmax(d_prime_temp[0:-1]) # P
            preferred_cat = stim_cat[preferred_cat_index]

            npreferred_cat_index = np.argmin(d_prime_temp[0:-1]) # NP
            npreferred_cat = stim_cat[npreferred_cat_index]

            print('Preferred img. cat.: ', preferred_cat)
            cat = [preferred_cat, npreferred_cat]

            # select events
            event_idx = select_events_durationTrials(events, tempCond, preference, cat)
        else:

            # select events
            event_idx = select_events_durationTrials(events, tempCond, preference)

        # retrieve broadband data and compute metrics
        for j in range(len(tempCond)):

            # select data
            broadband_current[i, j, :] = np.nanmean(epochs_b[event_idx[j]], axis=1)

            # predict data with model
            if model == 'csDN':
                if preference == 0:     # all
                    temp = np.zeros((len(stim_cat), len(t)))
                    for l in range(len(stim_cat)):
                        _, temp[l, :] = model_csDN(stim_onepulse[j, :], 'onepulse', j, stim_cat[l], sample_rate, params_current, dir)
                    broadband_pred_current[i, j, :] = np.mean(temp, 0)
                elif preference == 1:   # preferred
                    _, broadband_pred_current[i, j, :] = model_csDN(stim_onepulse[j, :], 'onepulse', j, cat[0], sample_rate, params_current, dir)
                elif preference == 2:   # nonpreferred
                    temp = np.zeros((len(stim_cat)-1, len(t)))
                    num = 0
                    for l in range(len(stim_cat)):
                        if l != preferred_cat_index:
                            _, temp[num, :] = model_csDN(stim_onepulse[j, :], 'onepulse', j, stim_cat[l], sample_rate, params_current, dir)
                            num+=1
                    broadband_pred_current[i, j, :] = np.mean(temp, 0)
            elif model == 'DN':
                broadband_pred_current[i, j, :] = model_DN(stim_onepulse[j, :], sample_rate, params_current)
        
    # perform bootstrap over broadband timecourse
    for i in range(B_repetitions):

        # draw random sample
        idx_temp = np.arange(n_electrodes)
        n_samples = len(idx_temp)
        boot = resample(idx_temp, replace=True, n_samples=n_samples)

        for j in range(len(tempCond)):

            # retrieve samples
            data_mean = np.zeros((len(boot), len(t)))
            model_mean = np.zeros((len(boot), len(t)))
            for l in range(len(boot)):
                data_mean[l, :] = broadband_current[boot[l], j, :]
                model_mean[l, :] = broadband_pred_current[boot[l], j, :]
            
            # NEURAL DATA
            data_mean = np.mean(data_mean, 0)
            broadband_bootstrap_current[i, j, :] = data_mean
            
            # MODEL
            model_mean = np.mean(model_mean, 0)
            broadband_pred_bootstrap_current[i, j, :] = model_mean

            # NEURAL DATA - determine half-max value
            min_val = np.min(data_mean[timepoints_onepulse[0, 0]:timepoints_onepulse[0, 0]+time_window])

            max_val = np.max(data_mean[timepoints_onepulse[0, 0]:timepoints_onepulse[0, 0]+time_window])
            max_idx = np.argmax(data_mean[timepoints_onepulse[0, 0]:timepoints_onepulse[0, 0]+time_window])

            hm_value = (min_val + max_val)/2 # half-max value
            hm_above = data_mean > hm_value

            first_above = np.argmax(hm_above[timepoints_onepulse[0, 0]:timepoints_onepulse[0, 0]+max_idx] == True)
            fwhm1 = timepoints_onepulse[0, 0] + first_above

            first_above = np.argmax(hm_above[timepoints_onepulse[0, 0]+max_idx:-1] == False)
            fwhm2 = timepoints_onepulse[0, 0] + max_idx + first_above

            fwhm_current[i, j] = list(t)[fwhm2] - list(t)[fwhm1]
            fwhm_medians[count_VA, j, i] = list(t)[fwhm2] - list(t)[fwhm1]

            # MODEL -  half-width at half maximum 
            min_val = np.min(model_mean[timepoints_onepulse[0, 0]:timepoints_onepulse[0, 0]+time_window])

            max_val = np.max(model_mean[timepoints_onepulse[0, 0]:timepoints_onepulse[0, 0]+time_window])
            max_idx = np.argmax(model_mean[timepoints_onepulse[0, 0]:timepoints_onepulse[0, 0]+time_window])

            hm_value = (min_val + max_val)/2 # half-max value
            hm_above = model_mean > hm_value

            first_above = np.argmax(hm_above[timepoints_onepulse[0, 0]:timepoints_onepulse[0, 0]+max_idx] == True)
            fwhm1 = timepoints_onepulse[0, 0] + first_above

            first_above = np.argmax(hm_above[timepoints_onepulse[0, 0]+max_idx:-1] == False)
            fwhm2 = timepoints_onepulse[0, 0] + max_idx + first_above

            fwhm_pred_current[i, j] = list(t)[fwhm2] - list(t)[fwhm1]
            fwhm_pred_medians[count_VA, j, i] = list(t)[fwhm2] - list(t)[fwhm1]

            # time-to-peak
            if j == 5: # time-to-peak (ttp) and ratio-sustained/transient

                # NEURAL DATA
                temp = np.argmax(data_mean[timepoints_onepulse[j, 0]:timepoints_onepulse[j, 1]])
                ttp_current[i] = list(t)[timepoints_onepulse[j, 0]+temp]
                ttp_medians[count_VA, i] = list(t)[timepoints_onepulse[j, 0]+temp]

                # MODEL
                temp = np.argmax(model_mean[timepoints_onepulse[j, 0]:timepoints_onepulse[j, 1]])
                ttp_pred_current[i] = list(t)[timepoints_onepulse[j, 0]+temp]
                ttp_pred_medians[count_VA, i] = list(t)[timepoints_onepulse[j, 0]+temp]

        # normalize responses to maximum (over all temporal conditions)
        broadband_bootstrap_current[i, :, :] = broadband_bootstrap_current[i, :, :]/np.amax(broadband_bootstrap_current[i, :, :])    
        broadband_pred_bootstrap_current[i, :, :] = broadband_pred_bootstrap_current[i, :, :]/np.amax(broadband_pred_bootstrap_current[i, :, :]) 
    
    # append dataframes
    broadband.append(broadband_current)
    broadband_bootstrap.append(broadband_bootstrap_current)

    broadband_pred.append(broadband_pred_current)
    broadband_pred_bootstrap.append(broadband_pred_bootstrap_current)

    ttp.append(ttp_current)
    ttp_pred.append(ttp_pred_current)

    fwhm.append(fwhm_current)
    fwhm_pred.append(fwhm_pred_current)

    # NEURAL DATA - time-to-peak
    data_temp = ttp[count_VA]
    ttp_avg[count_VA] = statistics.median(data_temp) # true median
    ttp_CI[count_VA, :] = np.nanpercentile(data_temp, [CI_low, CI_high])

    # MODEL - time-to-peak
    model_temp = ttp_pred[count_VA]
    n_samples = len(model_temp)
    ttp_avg_pred[count_VA] = statistics.median(model_temp)
    ttp_CI_pred[count_VA, :] = np.nanpercentile(model_temp, [CI_low, CI_high])

    # NEURAL DATA - full-width at half maximum
    for j in range(len(tempCond)):
        data_temp = fwhm[count_VA][:, j]
        fwhm_avg[count_VA, j] = statistics.median(data_temp)
        fwhm_CI[count_VA, j, :] = np.nanpercentile(data_temp, [CI_low, CI_high])

    # MODEL - full-width at half maximum
    for j in range(len(tempCond)):
        model_temp = fwhm_pred[count_VA][:, j]
        fwhm_avg_pred[count_VA, j] = statistics.median(model_temp)
        fwhm_CI_pred[count_VA, j, :] = np.nanpercentile(model_temp, [CI_low, CI_high])

    # increment count
    count_VA+=1

############################################################################
################################################################## VISUALIZE

# initiate figure
fig = plt.figure(figsize=(18, 10))
gs = fig.add_gridspec(19, 22)
ax = dict()

# set fontsizes
fontsize_tick           = 20
fontsize_legend         = 20
fontsize_label          = 20

alpha = 1 # transparency
marker_pred             = '^'
marker_ttp              = 180
marker_fwhm             = 80

# plot specs/adjustments
start                   = 50
end                     = 600
sep                     = 100

# set ticks
add = np.zeros((len(VA), len(tempCond)))
start_add = [0.016, 0.017, 0.018]
add[:, 0] = start_add
for i in range(len(VA)):
    for j in range(1, len(tempCond)):
        add[i, j] = add[i, j - 1]*2

# initiate subplots
ax['broadband'] = fig.add_subplot(gs[0:3, 0:22])
ax['broadband_model'] = fig.add_subplot(gs[4:7, 0:22])

ax['ttp'] = fig.add_subplot(gs[12:19, 0:7])  
ax['fwhm'] = fig.add_subplot(gs[12:15, 9:22])
ax['fwhm_pred'] = fig.add_subplot(gs[16:19, 9:22])
ax_metrics = [ax['ttp'], ax['fwhm'], ax['fwhm_pred']]

# seperate axes
sns.despine(offset=10)

# plot stimulus timecourse and time courses of neural data & model
t_zero          = np.argwhere(t > 0)[0][0]
t_twohundred    = np.argwhere(t > 0.5)[0][0]

x_label_single = ['0', '500']

xtick_idx = []
for i in range(len(tempCond)):

    # append x-tick
    xtick_idx = xtick_idx + ([i*(end+sep) + t_zero, i*(end+sep) + t_twohundred])

    # plot stimulus timecourse
    end_stim = timepoints_onepulse[i, 1]
    for j in range(len(VA)):
        if (j == 0) & (i == 0):
            ax['broadband'].axvspan(i*(end+sep) - start + timepoints_onepulse[i, 0], i*(
                end+sep) + end_stim, facecolor='grey', alpha=0.2, label='stimulus')
            ax['broadband_model'].axvspan(i*(end+sep) - start + timepoints_onepulse[i, 0], i*(
                end+sep) + end_stim, facecolor='grey', alpha=0.2)
        elif (j == 0):
            ax['broadband'].axvspan(i*(end+sep) - start + timepoints_onepulse[i, 0], i*(
                end+sep) + end_stim, facecolor='grey', alpha=0.2)
            ax['broadband_model'].axvspan(i*(end+sep) - start + timepoints_onepulse[i, 0], i*(
                end+sep) + end_stim, facecolor='grey', alpha=0.2)

        # plot mean values (panel A)
        data_temp = gaussian_filter1d(np.mean(broadband_bootstrap[j][:, i, :], axis=0), 10)
        model_temp = gaussian_filter1d(np.mean(broadband_pred_bootstrap[j][:, i, :], axis=0), 10)

        if i == 0:
            ax['broadband'].plot(np.arange(end - start)+i*(end+sep), data_temp[start:end], color=np.array(colors_VA[j])/255, label=VA[j])
            ax['broadband_model'].plot(np.arange(end - start)+i*(end+sep), model_temp[start:end], color=np.array(colors_VA[j])/255)
        else:
            ax['broadband'].plot(np.arange(end - start)+i*(end+sep), data_temp[start:end], color=np.array(colors_VA[j])/255)
            ax['broadband_model'].plot(np.arange(end - start)+i*(end+sep), model_temp[start:end], color=np.array(colors_VA[j])/255)

# plot time-to-peak (panel B) - neural data
error_min = ttp_CI[:, 0]
error_max = ttp_CI[:, 1]
ax['ttp'].scatter(np.arange(len(VA)), ttp_avg, color=np.array(colors_VA)/255, s=marker_ttp)
ax['ttp'].plot([np.arange(len(VA)), np.arange(len(VA))], [error_min, error_max], color='k', zorder=1)

# plot time-to-peak (panel B) - neural data
error_min = ttp_CI_pred[:, 0]
error_max = ttp_CI_pred[:, 1]
ax['ttp'].scatter(np.arange(len(VA))+0.2, ttp_avg_pred, color=np.array(colors_VA)/255, s=marker_ttp, marker=marker_pred)
ax['ttp'].plot([np.arange(len(VA))+0.2, np.arange(len(VA))+0.2], [error_min, error_max], color='black', zorder=1)

#  plot bargraph (slope log-linear fit)
for j in range(len(tempCond)):
    error_min = fwhm_CI[:, j, 0]
    error_max = fwhm_CI[:, j, 1]
    ax['fwhm'].scatter(np.arange(len(VA))+5*j, fwhm_avg[:, j], color=np.array(colors_VA)/255, s=marker_fwhm)
    ax['fwhm'].plot([np.arange(len(VA))+5*j, np.arange(len(VA))+5*j], [error_min, error_max], color='k', zorder=1)

for j in range(len(tempCond)):
    error_min = fwhm_CI_pred[:, j, 0]
    error_max = fwhm_CI_pred[:, j, 1]
    ax['fwhm_pred'].scatter(np.arange(len(VA))+5*j, fwhm_avg_pred[:, j], color=np.array(colors_VA)/255, s=marker_fwhm, alpha=alpha, marker=marker_pred)
    ax['fwhm_pred'].plot([np.arange(len(VA))+5*j, np.arange(len(VA))+5*j], [error_min, error_max], color='black', zorder=1)

# add legend and xticks
ax['broadband'].legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left", borderaxespad=0, ncol=4, frameon=False, fontsize=fontsize_legend)

# adjust axes
ax['broadband'].spines['top'].set_visible(False)
ax['broadband'].spines['right'].set_visible(False)
ax['broadband'].tick_params(axis='x', labelsize=fontsize_label,  labelrotation=45)
ax['broadband'].tick_params(axis='both', which='major',          labelsize=fontsize_tick)
ax['broadband'].set_yticks([0, 0.5, 1])
ax['broadband'].axhline(0, color='grey', lw=0.5, alpha=0.5)
ax['broadband'].set_xticks(xtick_idx)
ax['broadband'].set_xticklabels([])
# ax['broadband'].set_ylabel('Neural data', fontsize=fontsize_label)

ax['broadband_model'].spines['top'].set_visible(False)
ax['broadband_model'].spines['right'].set_visible(False)
ax['broadband_model'].tick_params(axis='both', which='major', labelsize=fontsize_tick)
# ax['broadband_model'].tick_params(axis='x', labelsize=fontsize_label, labelrotation=45)
ax['broadband_model'].set_xlabel('Time (ms)', fontsize=fontsize_label)
ax['broadband_model'].set_yticks([0, 0.5, 1])
ax['broadband_model'].axhline(0, color='grey', lw=0.5, alpha=0.5)
ax['broadband_model'].set_xticks(xtick_idx)
ax['broadband_model'].set_xticklabels(np.tile(x_label_single, 6))
# ax['broadband_model'].set_ylabel('DN model', fontsize=fontsize_label)

# seperate axes
sns.despine(offset=10)

for i in range(len(ax_metrics)):

    ax_metrics[i].spines['top'].set_visible(False)
    ax_metrics[i].spines['right'].set_visible(False)
    ax_metrics[i].tick_params(axis='both', which='major', labelsize=fontsize_tick)

    if i == 0:
        ax_metrics[i].set_xticks(np.arange(len(VA))+0.1)
        ax_metrics[i].set_xlim(-0.5, 2.5)
        ax_metrics[i].set_xticklabels([' ', ' ', ' '], fontsize=fontsize_label)
        if preference == 0:
            ax_metrics[i].set_ylim(0.05, 0.25)
        elif preference == 1:
            ax_metrics[i].set_ylim(0.05, 0.3)
        elif preference == 2:
            ax_metrics[i].set_ylim(0.05, 0.3)
        # ax_metrics[i].set_ylabel('Time-to-peak (s)', fontsize=fontsize_label)

    elif i == 1:
        ax_metrics[i].set_xticks([1, 6, 11, 16, 21, 26])
        ax_metrics[i].set_xticklabels([' ', ' ', ' ', ' ', ' ', ' '])
        if preference == 0:
            ax_metrics[i].set_ylim(-0.05, 0.6)
        elif preference == 1:
            ax_metrics[i].set_ylim(-0.05, 0.7)
        elif preference == 2:
            ax_metrics[i].set_ylim(-0.05, 0.8)

        # ax_metrics[i].set_ylabel('Fwhm (s)', fontsize=fontsize_label)

    elif i == 2:
        ax_metrics[i].set_xticks([1, 6, 11, 16, 21, 26])
        ax_metrics[i].set_xticklabels(label_tempCond, fontsize=fontsize_label, rotation=45)
        if preference == 0:
            ax_metrics[i].set_ylim(-0.05, 0.6)
        elif preference == 1:
            ax_metrics[i].set_ylim(0.0, 0.8)
        elif preference == 2:
            ax_metrics[i].set_ylim(-0.05, 0.8)

        # ax_metrics[i].set_ylabel('Fwhm (s)', fontsize=fontsize_label)
        # ax_metrics[i].set_xlabel('Stimulus duration (ms)', fontsize=fontsize_label)

# save figure
fig.align_ylabels()
plt.tight_layout()
plt.savefig(dir+'mkFigure/Fig4_' + img_type + '.svg', format='svg')
plt.savefig(dir+'mkFigure/Fig4_' + img_type, dpi=300)

# plt.show()

############################################################################
######################################################## STATISTICAL TESTING

alpha = 0.05
Bonferroni = 6

metrics = ['Time-to-peak', 'Full-width half max']
for i in range(2):

    print(30*'--')
    print(metrics[i])

    if i == 0: # time-to-peak

        print('#'*30)
        print('NEURAL DATA')

        # early vs. ventral
        sample1 = ttp_medians[0, :]
        sample2 = ttp_medians[1, :]
        param_diffs = sample1 - sample2

        p = np.min([len(param_diffs[param_diffs < 0]), len(param_diffs[param_diffs > 0])])/B_repetitions
        if p < alpha:
            print('V1-3 vs. VOTC: ', p, ' SIGNIFICANT')
        else:
            print('V1-3 vs. VOTC: ', p)

        # early vs. LO
        sample1 = ttp_medians[0, :]
        sample2 = ttp_medians[2, :]
        param_diffs = sample1 - sample2

        p = np.min([len(param_diffs[param_diffs < 0]), len(param_diffs[param_diffs > 0])])/B_repetitions
        if p < alpha:
            print('V1-3 vs. LOTC: ', p, ' SIGNIFICANT')
        else:
            print('V1-3 vs. LOTC: ', p)

        # ventral vs. LO
        sample1 = ttp_medians[1, :]
        sample2 = ttp_medians[2, :]
        param_diffs = sample1 - sample2

        p = np.min([len(param_diffs[param_diffs < 0]), len(param_diffs[param_diffs > 0])])/B_repetitions
        if p < alpha:
            print('VOTC vs. LOTC: ', p, ' SIGNIFICANT')
        else:
            print('VOTC vs. LOTC: ', p)

        print('#'*30)
        print('MODEL')

        # early vs. ventral
        sample1 = ttp_pred_medians[0, :]
        sample2 = ttp_pred_medians[1, :]
        param_diffs = sample1 - sample2

        p = np.min([len(param_diffs[param_diffs < 0]), len(param_diffs[param_diffs > 0])])/B_repetitions
        if p < alpha:
            print('V1-3 vs. VOTC: ', p, ' SIGNIFICANT')
        else:
            print('V1-3 vs. VOTC: ', p)

        # early vs. LO
        sample1 = ttp_pred_medians[0, :]
        sample2 = ttp_pred_medians[2, :]
        param_diffs = sample1 - sample2

        p = np.min([len(param_diffs[param_diffs < 0]), len(param_diffs[param_diffs > 0])])/B_repetitions
        if p < alpha:
            print('V1-3 vs. LOTC: ', p, ' SIGNIFICANT')
        else:
            print('V1-3 vs. LOTC: ', p)

        # ventral vs. LO
        sample1 = ttp_pred_medians[1, :]
        sample2 = ttp_pred_medians[2, :]
        param_diffs = sample1 - sample2

        p = np.min([len(param_diffs[param_diffs < 0]), len(param_diffs[param_diffs > 0])])/B_repetitions
        if p < alpha:
            print('VOTC vs. LOTC: ', p, ' SIGNIFICANT')
        else:
            print('VOTC vs. LOTC: ', p)

    elif i == 1: # full-width at half-maximum

        print('#'*30)
        print('NEURAL DATA')

        for j in range(len(tempCond)):

            print('\ntempCond: ', j+1)

            # early vs. ventral
            sample1 = fwhm_medians[0, j, :]
            sample2 = fwhm_medians[1, j, :]
            param_diffs = sample1 - sample2

            p = np.min([len(param_diffs[param_diffs < 0]), len(param_diffs[param_diffs > 0])])/B_repetitions
            if p < alpha/Bonferroni:
                print('V1-3 vs. VOTC: ', p, ' SIGNIFICANT')
            else:
                print('V1-3 vs. VOTC: ', p)

            # early vs. LO
            sample1 = fwhm_medians[0, j, :]
            sample2 = fwhm_medians[2, j, :]
            param_diffs = sample1 - sample2

            p = np.min([len(param_diffs[param_diffs < 0]), len(param_diffs[param_diffs > 0])])/B_repetitions
            if p < alpha/Bonferroni:
                print('V1-3 vs. LOTC: ', p, ' SIGNIFICANT')
            else:
                print('V1-3 vs. LOTC: ', p)

            # ventral vs. LO
            sample1 = fwhm_medians[1, j, :]
            sample2 = fwhm_medians[2, j, :]
            param_diffs = sample1 - sample2

            p = np.min([len(param_diffs[param_diffs < 0]), len(param_diffs[param_diffs > 0])])/B_repetitions
            if p < alpha/Bonferroni:
                print('VOTC vs. LOTC: ', p, ' SIGNIFICANT')
            else:
                print('VOTC vs. LOTC: ', p)

        print('#'*30)
        print('MODEL')

        for j in range(len(tempCond)):

            print('\ntempCond: ', j+1)

            # early vs. ventral
            sample1 = fwhm_pred_medians[0, j, :]
            sample2 = fwhm_pred_medians[1, j, :]
            param_diffs = sample1 - sample2

            p = np.min([len(param_diffs[param_diffs < 0]), len(param_diffs[param_diffs > 0])])/B_repetitions
            if p < alpha/Bonferroni:
                print('V1-3 vs. VOTC: ', p, ' SIGNIFICANT')
            else:
                print('V1-3 vs. VOTC: ', p)

            # early vs. LO
            sample1 = fwhm_pred_medians[0, j, :]
            sample2 = fwhm_pred_medians[2, j, :]
            param_diffs = sample1 - sample2

            p = np.min([len(param_diffs[param_diffs < 0]), len(param_diffs[param_diffs > 0])])/B_repetitions
            if p < alpha/Bonferroni:
                print('V1-3 vs. LOTC: ', p, ' SIGNIFICANT')
            else:
                print('V1-3 vs. LOTC: ', p)

            # ventral vs. LO
            sample1 = fwhm_pred_medians[1, j, :]
            sample2 = fwhm_pred_medians[2, j, :]
            param_diffs = sample1 - sample2

            p = np.min([len(param_diffs[param_diffs < 0]), len(param_diffs[param_diffs > 0])])/B_repetitions
            if p < alpha/Bonferroni:
                print('VOTC vs. LOTC: ', p, ' SIGNIFICANT')
            else:
                print('VOTC vs. LOTC: ', p)

# save figure
plt.tight_layout()
plt.savefig(dir+'mkFigure/Fig4_' + img_type + '.svg', format='svg', bbox_inches='tight')
plt.savefig(dir+'mkFigure/Fig4_' + img_type, dpi=300, bbox_inches='tight')
# plt.show()



