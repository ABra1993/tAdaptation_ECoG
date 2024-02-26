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
from modelling_utils_fitObjective import model, objective, OF_ISI_recovery_log


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

# import info responsive electrodes showing category-selectivity
threshold_d_prime = 0.5
# threshold_d_prime = 0.75
# threshold_d_prime = 1.0

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

# define model
model_type = 'DN'
    
# scaling
scaling = 'P'

# retrieve parameters
params_names, _, _, _ = paramInit(model_type, scaling)
sample_rate = 512

# labels
VA_labels = ['V1-V3', 'VOTC', 'LOTC']
cat_labels = ['Preferred', 'Nonpreferred']

# create stimulus timecourse
cond = [6]
trials = ['onepulse', 'twopulse_repeat']
stim_twopulse = np.zeros((len(tempCond), len(t))) 
for i in range(len(tempCond)):
    stim_twopulse[i, :] = generate_stimulus_timecourse(trials[1], i, dir)

# initiate figure
fig = plt.figure(figsize=(16, 3))
gs = fig.add_gridspec(4, 15)
ax = dict()

# initiate plots
ax['broadband_cat_sel_repeat']              = fig.add_subplot(gs[0:4, 0:5])
ax['broadband_cat_sel_pref_repeat']         = fig.add_subplot(gs[0:4, 6:10])
ax['broadband_cat_sel_npref_repeat']        = fig.add_subplot(gs[0:4, 11:15])

ax_broadband = [ax['broadband_cat_sel_repeat']]
ax_broadband_cat_sel_repeat = [ax['broadband_cat_sel_pref_repeat'], ax['broadband_cat_sel_npref_repeat']]

# seperate axes
sns.despine(offset=10)

# fontsizes
fontsize_tick                       = 20
fontsize_legend                     = 20
fontsize_label                      = 20
fontsize_title                      = 22

for i in range(len(ax_broadband)):
    ax_broadband[i].tick_params(axis='both', which='major', labelsize=fontsize_tick)
    ax_broadband[i].spines['top'].set_visible(False)
    ax_broadband[i].spines['right'].set_visible(False)
    ax_broadband[i].axhline(0, color='grey', lw=0.5, alpha=0.5)
    # ax_broadband[i].set_ylabel('Change in power\n(x-fold)')
    # ax_broadband[i].set_xlabel('Time (ms)')

for i in range(len(ax_broadband_cat_sel_repeat)):
    ax_broadband_cat_sel_repeat[i].tick_params(axis='both', which='major', labelsize=fontsize_tick)
    ax_broadband_cat_sel_repeat[i].spines['top'].set_visible(False)
    ax_broadband_cat_sel_repeat[i].spines['right'].set_visible(False)
    ax_broadband_cat_sel_repeat[i].axhline(0, color='grey', lw=0.5, alpha=0.5)
    # ax_broadband_cat_sel_repeat[i].set_xlabel('Time (ms)')
    # ax_broadband_cat_sel_repeat[i].set_title(cat_labels[i], fontsize=fontsize_label)
    # if i == 0:
        # ax_broadband_cat_sel_repeat[i].set_ylabel('Model prediction')

# linestyles
linestyle = ['solid', 'dashed']
alpha = 0.3
lw = 2

# import image classes
subtrials = ['preferred', 'non-preferred']
color_cat = ['dodgerblue', 'crimson']

# import info responsive electrodes showing category-selectivity
responsive_electrodes = pd.read_csv(dir+'subject_data/electrodes_categorySelective_' + str(threshold_d_prime).replace('.', '-') + '.txt', header=0, index_col=0, delimiter=' ')
responsive_electrodes = responsive_electrodes[responsive_electrodes.preferred_cat != 'SCRAMBLED']
responsive_electrodes.reset_index(drop=True, inplace=True)
n_electrodes = len(responsive_electrodes)

############################################################################
############################################################### PLOT RESULTS - REPEAT

# initiate dataframes to store data
broadband_cat_sel = np.zeros((n_electrodes, len(subtrials), len(t)))
broadband_cat_sel_bootstrap = np.zeros((B_repetitions, len(subtrials), len(t)))

broadband_cat_sel_pred = np.zeros((n_electrodes, len(subtrials), len(t)))
broadband_cat_sel_nom_pred = np.zeros((n_electrodes, len(subtrials), len(t)))
broadband_cat_sel_denom_pred = np.zeros((n_electrodes, len(subtrials), len(t)))

broadband_cat_sel_pred_bootstrap = np.zeros((B_repetitions, len(subtrials), len(t)))
broadband_cat_sel_nom_pred_bootstrap = np.zeros((B_repetitions, len(subtrials), len(t)))
broadband_cat_sel_denom_pred_bootstrap = np.zeros((B_repetitions, len(subtrials), len(t)))

# iterate over electrodes
current_subject = ''
for i in range(n_electrodes):
# for i in range(1):

    # retrieve info current electrode
    subject = responsive_electrodes.subject[i]
    electrode_name = responsive_electrodes.electrode[i]
    electrode_idx = int(responsive_electrodes.electrode_idx[i])

    # retrieve model parameters for current electrode
    temp = pd.read_csv(dir+'modelFit/categorySelective/' + subject + '_' + electrode_name + '/param_' + model_type + '_' + scaling + '.txt', header=0, delimiter=' ', index_col=0)
    temp.reset_index(inplace=True,drop=True)
    params_current = list(temp.loc[0, params_names])

    # print progress
    print('Computing trials for ' + subject + ', electrode ' + electrode_name + ' (' + str(i+1) + '/' + str(n_electrodes) + ')')

    if current_subject != subject:

        # update current subject
        current_subject = subject

        # import info
        _, events, channels, _ = import_info(subject, dir)

        # import excluded trials
        excluded_epochs = pd.read_csv(dir+'subject_data/' + subject + '/excluded_epochs.txt', sep=' ', index_col=0, header=0, dtype=int)

    # extract data
    epochs_b = import_epochs(subject, electrode_idx, dir)
    index_epochs_b = [j for j in range(len(events)) if excluded_epochs.iloc[electrode_idx, j] == 1]
    epochs_b.iloc[:, index_epochs_b] = np.nan

    # determine category selectivity (i.e. localizer)
    event_idx_onepulse = select_events(events, 'STIM', 'onepulse', dir)
    d_prime_temp = d_prime_perImgCat(epochs_b, event_idx_onepulse, stim_cat) # double-check for/confirm img. category selection

    preferred_cat_index = np.argmax(d_prime_temp[0:-1]) # P (with exception of scrambled)
    preferred_cat = stim_cat[preferred_cat_index]

    npreferred_cat_index = np.argmin(d_prime_temp[0:-1]) # NP
    npreferred_cat = stim_cat[npreferred_cat_index]
    
    print('Preferred img. cat.: ', preferred_cat)
    cat = [preferred_cat, npreferred_cat]

    # select twopulse trials
    # select twopulse trials
    event_idx_preferred = select_events_repetitionTrials(events, tempCond, 1, cat)  
    event_idx_nonpreferred = select_events_repetitionTrials(events, tempCond, 2, cat)  
    event_idx = [event_idx_preferred[2], event_idx_nonpreferred[2]]  

    for j in range(len(subtrials)):

        # select twouplse
        # NEURAL DATA
        broadband_cat_sel[i, j, :] = np.nanmean(epochs_b[event_idx[j][5]], axis=1)

        # MODEL
        # for 5 remaining img classes
        if j == 0:
            broadband_cat_sel_pred[i, j, :], broadband_cat_sel_nom_pred[i, j, :], broadband_cat_sel_denom_pred[i, j, :] = model(model_type, scaling, stim_twopulse[5, :], sample_rate, params_current, dir, 'twopulse', 5, cat[j], denom=True)
        elif j == 1:
            data_temp = np.zeros((len(stim_cat)-1, len(t)))
            data_nom_temp = np.zeros((len(stim_cat)-1, len(t)))
            data_denom_temp = np.zeros((len(stim_cat)-1, len(t)))
            num = 0
            for l in range(len(stim_cat)):
                if stim_cat[l] != cat[0]:
                    data_temp[num, :], data_nom_temp[num, :], data_denom_temp[num, :] = model(model_type, scaling, stim_twopulse[5, :], sample_rate, params_current, dir, 'twopulse', 5, cat[j], denom=True)
                    num+=1
            broadband_cat_sel_pred[i, j, :] = np.mean(data_temp, 0)
            broadband_cat_sel_nom_pred[i, j, :] = np.mean(data_nom_temp, 0)
            broadband_cat_sel_denom_pred[i, j, :] = np.mean(data_denom_temp, 0)

# perform bootstrap over broadband timecourse
for i in range(B_repetitions):

    # draw random sample
    idx_temp = np.arange(n_electrodes)
    n_samples = len(idx_temp)
    boot = resample(idx_temp, replace=True, n_samples=n_samples)

    for j in range(len(subtrials)):

        # retrieve broadband
        # NEURAL DATA
        data_temp = np.zeros((len(boot), len(t)))
        for l in range(len(boot)):
            data_temp[l, :] = broadband_cat_sel[boot[l], j, :]
        broadband_cat_sel_bootstrap[i, j, :] = gaussian_filter1d(np.mean(data_temp, 0), 10)
        
        # MODEL
        model_temp = np.zeros((len(boot), len(t)))
        model_nom_temp = np.zeros((len(boot), len(t)))
        model_denom_temp = np.zeros((len(boot), len(t)))
        for l in range(len(boot)):
            model_temp[l, :] = broadband_cat_sel_pred[boot[l], j, :]
            model_nom_temp[l, :] = broadband_cat_sel_nom_pred[boot[l], j, :]
            model_denom_temp[l, :] = broadband_cat_sel_denom_pred[boot[l], j, :]
        broadband_cat_sel_pred_bootstrap[i, j, :] = gaussian_filter1d(np.mean(model_temp, 0), 10)
        broadband_cat_sel_nom_pred_bootstrap[i, j, :] = gaussian_filter1d(np.mean(model_nom_temp, 0), 10)
        broadband_cat_sel_denom_pred_bootstrap[i, j, :] = gaussian_filter1d(np.mean(model_denom_temp, 0), 10)

# plot stimulus timecourse
ax_broadband[0].axvspan(t[timepoints_twopulse[cond[0]-1, 0]], t[timepoints_twopulse[cond[0]-1, 1]], facecolor='grey', alpha=0.2, label='stimulus')
ax_broadband[0].axvspan(t[timepoints_twopulse[cond[0]-1, 2]], t[timepoints_twopulse[cond[0]-1, 3]], facecolor='grey', alpha=0.2, label='stimulus')

for i in range(len(subtrials)):

    # plot stimulus timecourse
    ax_broadband_cat_sel_repeat[i].axvspan(t[timepoints_twopulse[cond[0]-1, 0]], t[timepoints_twopulse[cond[0]-1, 1]], facecolor='grey', alpha=0.2, label='stimulus')
    ax_broadband_cat_sel_repeat[i].axvspan(t[timepoints_twopulse[cond[0]-1, 2]], t[timepoints_twopulse[cond[0]-1, 3]], facecolor='grey', alpha=0.2, label='stimulus')

    # NEURAL DATA
    # data_temp = np.mean(broadband_cat_sel_bootstrap[:, i, :], 0)
    # ax_broadband[1].plot(t, data_temp/max(data_temp), color=color_cat[i], alpha=0.2)

    # MODEL - ALL
    model_temp = np.mean(broadband_cat_sel_pred_bootstrap[:, i, :], axis=0)
    ax_broadband[0].plot(t, model_temp/max(model_temp), color=color_cat[i])

    # MODEL - NOM
    nom_temp = np.mean(broadband_cat_sel_nom_pred_bootstrap[:, i, :], axis=0)
    ax_broadband_cat_sel_repeat[i].plot(t, nom_temp/max(nom_temp), color=color_cat[i], linestyle=linestyle[0], alpha=alpha)

    # MODEL - DENOM
    denom_temp = np.mean(broadband_cat_sel_denom_pred_bootstrap[:, i, :], axis=0)
    ax_broadband_cat_sel_repeat[i].plot(t, denom_temp/max(denom_temp), color=color_cat[i], linestyle=linestyle[1], lw=lw)

# save figure
plt.tight_layout()
plt.savefig(dir+'mkFigure/Fig9_P.svg', format='svg', bbox_inches='tight')
plt.savefig(dir+'mkFigure/Fig9_P', dpi=300, bbox_inches='tight')
# plt.show()
