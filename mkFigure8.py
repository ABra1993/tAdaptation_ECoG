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

# define trials
color = ['dodgerblue', 'crimson']
subtrials = ['twopulse_repeat_pref', 'twopulse_repeat_npref']
labels = ['Preferred', 'Non-preferred']

# get img. classes
stim_cat = np.loadtxt(dir+'variables/cond_stim.txt', dtype=str)

# determine confidence interval (error)
CI                  = 68
CI_low              = 50 - (0.5*CI)
CI_high             = 50 + (0.5*CI)
B_repetitions       = 1000

# import info responsive electrodes showing category-selectivity
responsive_electrodes = pd.read_csv(dir+'subject_data/electrodes_categorySelective_' + str(threshold_d_prime).replace('.', '-') + '.txt', header=0, index_col=0, delimiter=' ')
responsive_electrodes = responsive_electrodes[responsive_electrodes.preferred_cat != 'SCRAMBLED']
responsive_electrodes.reset_index(drop=True, inplace=True)
n_electrodes = len(responsive_electrodes)

# define model
# model = 'DN'
model = 'csDN'

# retrieve parameters
params_names, _, _, _ = paramInit(model)
sample_rate = 512

# create stimulus timecourse
stim_onepulse = generate_stimulus_timecourse('onepulse', 4, dir)
stim_twopulse = np.zeros((len(tempCond), len(t))) 
for i in range(len(tempCond)):
    stim_twopulse[i, :] = generate_stimulus_timecourse('twopulse_repeat', i, dir)

# fit curve for recovery of adaptation initial parameter values
p0 = [1, 0]

# timescales to fit and plot curves
t1_plot = np.linspace(min(tempCond), max(tempCond), 1000)

############################################################################
################################################################### ANALYSIS

# initiat dataframes
broadband = np.zeros((n_electrodes, len(subtrials), len(tempCond), len(t)))
broadband_pulse1 = np.zeros((n_electrodes, len(subtrials), len(t)))
broadband_pulse2 = np.zeros((n_electrodes, len(subtrials), len(tempCond), len(t)))

broadband_pred = np.zeros((n_electrodes, len(subtrials), len(tempCond), len(t)))
broadband_pulse1_pred = np.zeros((n_electrodes, len(subtrials), len(t)))
broadband_pulse2_pred = np.zeros((n_electrodes, len(subtrials), len(tempCond), len(t)))

broadband_bootstrap = np.zeros((B_repetitions, len(subtrials), len(tempCond), len(t)))
broadband_pulse1_bootstrap = np.zeros((B_repetitions, len(subtrials), len(t)))
broadband_pulse2_bootstrap = np.zeros((B_repetitions, len(subtrials), len(tempCond), len(t)))

broadband_bootstrap_pred = np.zeros((B_repetitions, len(subtrials), len(tempCond), len(t)))
broadband_pulse1_bootstrap_pred = np.zeros((B_repetitions, len(subtrials), len(t)))        
broadband_pulse2_bootstrap_pred = np.zeros((B_repetitions, len(subtrials), len(tempCond), len(t)))


ISI_recovery = np.zeros((B_repetitions, len(subtrials), len(tempCond)))
ISI_recovery_avg = np.zeros((len(subtrials), len(tempCond)))
ISI_recovery_CI = np.zeros((len(subtrials), len(tempCond), 2))
ISI_recovery_log = np.zeros((B_repetitions, len(subtrials), len(t1_plot)))

ISI_recovery_pred = np.zeros((B_repetitions, len(subtrials), len(tempCond)))
ISI_recovery_pred_avg = np.zeros((len(subtrials), len(tempCond)))
ISI_recovery_pred_CI = np.zeros((len(subtrials), len(tempCond), 2))
ISI_recovery_log_pred = np.zeros((B_repetitions, len(subtrials), len(t1_plot)))


adaptation = np.zeros((B_repetitions, len(subtrials)))
adaptation_avg = np.zeros((len(subtrials)))
adaptation_CI = np.zeros((len(subtrials), 2))

adaptation_pred = np.zeros((B_repetitions, len(subtrials)))
adaptation_pred_avg =  np.zeros(len(subtrials))
adaptation_pred_CI = np.zeros((len(subtrials), 2))


intercept = np.zeros((B_repetitions, len(subtrials)))
intercept_avg =  np.zeros((len(subtrials)))
intercept_CI = np.zeros((len(subtrials), 2))

intercept_pred = np.zeros((B_repetitions, len(subtrials)))
intercept_pred_avg =  np.zeros((len(subtrials)))
intercept_pred_CI = np.zeros((len(subtrials), 2))

# save medians for statistical testing
adaptation_medians = np.zeros((len(subtrials), B_repetitions))
adaptation_pred_medians = np.zeros((len(subtrials), B_repetitions))

intercept_medians = np.zeros((len(subtrials), B_repetitions))
intercept_pred_medians = np.zeros((len(subtrials), B_repetitions))

# iterate over electrode and retrieve cross-validated performance for cbDN (fitted for all image classes seperately)
current_subject = ''    
i = 0
for i in range(n_electrodes):
# for i in range(1):

    # retrieve info current electrode
    subject = responsive_electrodes.subject[i]
    electrode_name = responsive_electrodes.electrode[i]
    electrode_idx = int(responsive_electrodes.electrode_idx[i])

    # retrieve model parameters for current electrode
    temp = pd.read_csv(dir+'modelFit/categorySelective/' + subject + '_' + electrode_name + '/param_' + model + '.txt', header=0, delimiter=' ', index_col=0)
    temp.reset_index(inplace=True, drop=True)
    params_current = list(temp.loc[0, params_names])

    # print progress
    print('Computing trials for ' + subject + ', electrode ' + electrode_name + ' (' + str(i+1) + '/' + str(n_electrodes) + ')')
    print('Preferred img. cat.: ', responsive_electrodes.loc[i, 'preferred_cat'])

    if current_subject != subject:

        # update current subject
        current_subject == subject

        # import info
        _, events, channels, _ = import_info(subject, dir)

        # select excluded epochs
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
    event_idx_preferred = select_events_repetitionTrials(events, tempCond, 1, cat)  
    event_idx_nonpreferred = select_events_repetitionTrials(events, tempCond, 2, cat)  
    event_idx = [event_idx_preferred, event_idx_nonpreferred]          

    # data
    for j in range(len(subtrials)):

        # get onepulse trials
        # NEURAL DATA
        data_first_pulse = estimate_first_pulse(t, epochs_b, event_idx[j], timepoints_twopulse)
        broadband_pulse1[i, j, :] = data_first_pulse

        # for 5 remaining img classes
        if j == 0:
            _, broadband_pulse1_pred[i, j, :] = model_csDN(stim_onepulse, 'onepulse', 3, cat[j], sample_rate, params_current, dir)  
        elif j == 1:
            temp = np.zeros((len(stim_cat)-1, len(t))) # all image categories expect preferred
            num = 0
            for l in range(len(stim_cat)):
                if stim_cat[l] != cat[0]:
                    _, temp[num, :] = model_csDN(stim_onepulse, 'onepulse', 3, stim_cat[l], sample_rate, params_current, dir) 
                    num+=1
            broadband_pulse1_pred[i, j, :] = np.mean(temp, 0)

        # select twouplse
        for k in range(len(tempCond)):

            # NEURAL DATA
            broadband[i, j, k, :] = np.nanmean(epochs_b[event_idx[j][2][k]], axis=1)

            # MODEL
            if j == 0:
                _, broadband_pred[i, j, k, :] = model_csDN(stim_twopulse[k, :], 'twopulse_repeat', k, cat[j], sample_rate, params_current, dir)  
            elif j == 1:
                temp = np.zeros((len(stim_cat)-1, len(t))) # all image categories expect preferred
                num = 0
                for l in range(len(stim_cat)):
                    if stim_cat[l] != cat[0]:
                        _, temp[num, :] = model_csDN(stim_twopulse[k, :], 'twopulse_repeat', k, stim_cat[l], sample_rate, params_current, dir) 
                        num+=1
                broadband_pred[i, j, k, :] = np.mean(temp, 0)

            # compute isolated second pulse
            broadband_pulse2[i, j, k, :] = broadband[i, j, k, :] - broadband_pulse1[i, j, :]
            broadband_pulse2_pred[i, j, k, :] = broadband_pred[i, j, k, :] - broadband_pulse1_pred[i, j, :]

# perform bootstrap over broadband timecourse
for i in range(B_repetitions):

    # draw random sample
    idx_temp = np.arange(n_electrodes)
    n_samples = len(idx_temp)
    boot = resample(idx_temp, replace=True, n_samples=n_samples)

    for j in range(len(subtrials)):

        # compute first pulse
        data_mean = np.zeros((len(boot), len(t)))
        model_mean = np.zeros((len(boot), len(t)))
        for l in range(len(boot)):
            data_mean[l, :] = broadband_pulse1[boot[l], j, :]
            model_mean[l, :] = broadband_pulse1_pred[boot[l], j, :]
        broadband_pulse1_bootstrap[i, j, :] = np.nanmean(data_mean, 0)
        broadband_pulse1_bootstrap_pred[i, j, :] = np.nanmean(model_mean, 0)

        # compute ISI recovery
        for k in range(len(tempCond)):

            # compute degree of recovery
            start_firstpulse        = timepoints_onepulse[0, 0]
            start_second_pulse      = timepoints_twopulse[k, 2]

            # retrieve broadband
            # NEURAL DATA
            data = np.zeros((len(boot), len(t)))
            data_pulse2 = np.zeros((len(boot), len(t)))
            for l in range(len(boot)):
                data[l, :] = broadband[boot[l], j, k, :]
                data_pulse2[l, :] = broadband_pulse2[boot[l], j, k, :]
            broadband_bootstrap[i, j, k, :] = np.nanmean(data, 0)
            broadband_pulse2_bootstrap[i, j, k, :] = np.nanmean(data_pulse2, 0)

            # compute degree of recovery
            AUC1 = np.trapz(broadband_pulse1_bootstrap[i, j, :][start_firstpulse: start_firstpulse+time_window])
            AUC2 = np.trapz(broadband_pulse2_bootstrap[i, j, k, :][start_second_pulse:start_second_pulse+time_window])

            ISI_recovery[i, j, k] = AUC2/AUC1

            # MODEL
            pred = np.zeros((len(boot), len(t)))
            pred_pulse2 = np.zeros((len(boot), len(t)))
            for l in range(len(boot)):
                pred[l, :] = broadband_pred[boot[l], j, k, :]
                pred_pulse2[l, :] = broadband_pulse2_pred[boot[l], j, k, :]
            broadband_bootstrap_pred[i, j, k, :] = np.nanmean(pred, 0)
            broadband_pulse2_bootstrap_pred[i, j, k, :] = np.nanmean(pred_pulse2, 0)
        
            # compute degree of recovery
            AUC1 = np.trapz(broadband_pulse1_bootstrap_pred[i, j, :][start_firstpulse: start_firstpulse+time_window])
            AUC2 = np.trapz(broadband_pulse2_bootstrap_pred[i, j, k, :][start_second_pulse:start_second_pulse+time_window])

            ISI_recovery_pred[i, j, k] = AUC2/AUC1              

        # NEURAL DATA
        popt, _ = curve_fit(OF_ISI_recovery_log, tempCond/1000, ISI_recovery[i, j, :], p0, maxfev=100000) #, bounds=((0, 0), (np.inf, np.inf)))
        ISI_recovery_log[i, j, :] = OF_ISI_recovery_log(t1_plot/1000, *popt)
        intercept[i, j] =  popt[0]
        intercept_medians[j, i] = popt[0]
        adaptation[i, j] = np.mean(ISI_recovery[i, j, :])
        adaptation_medians[j, :] = np.mean(ISI_recovery[i, j, :])

        # MODEL
        popt, _ = curve_fit(OF_ISI_recovery_log, tempCond/1000, ISI_recovery_pred[i, j, :], p0, maxfev=100000) #, bounds=((0, 0), (np.inf, np.inf)))
        ISI_recovery_log_pred[i, j, :] = OF_ISI_recovery_log(t1_plot/1000, *popt)
        intercept_pred[i, j] =  popt[0]
        intercept_pred_medians[j, i] = popt[0]
        adaptation_pred[i, j] = np.mean(ISI_recovery_pred[i, j, :])
        adaptation_pred_medians[j, :] = np.mean(ISI_recovery_pred[i, j, :])

# compute spread
for j in range(len(subtrials)):

    # NEURAL DATA
    adaptation_avg[j] = np.mean(adaptation[:, j])
    adaptation_CI[j, :] = np.nanpercentile(adaptation[:, j], [CI_low, CI_high])

    intercept_avg[j] = np.mean(intercept[:, j])
    intercept_CI[j, :] = np.nanpercentile(intercept[:, j], [CI_low, CI_high])
    
    for i in range(len(tempCond)):
        ISI_recovery_avg[j, i] = np.mean(ISI_recovery[:, j, i])
        ISI_recovery_CI[j, i, :] = np.nanpercentile(ISI_recovery[:, j, i], [CI_low, CI_high])

    # MODEL
    adaptation_pred_avg[j] = np.mean(adaptation_pred[:, j])
    adaptation_pred_CI[j, :] = np.nanpercentile(adaptation_pred[:, j], [CI_low, CI_high])

    intercept_pred_avg[j] = np.mean(intercept_pred[:, j])
    intercept_pred_CI[j, :] = np.nanpercentile(intercept_pred[:, j], [CI_low, CI_high])
    
    for i in range(len(tempCond)):
        ISI_recovery_pred_avg[j, i] = np.mean(ISI_recovery_pred[:, j, i])
        ISI_recovery_pred_CI[j, i, :] = np.nanpercentile(ISI_recovery_pred[:, j, i], [CI_low, CI_high])

############################################################################
################################################################## VISUALIZE

# initiate figure
fig = plt.figure(figsize=(18, 16))
gs = fig.add_gridspec(24, 20)
ax = dict()

# initiate subplots
ax['broadband'] = fig.add_subplot(gs[0:3, 0:20])
ax['broadband_pred'] = fig.add_subplot(gs[4:7, 0:20])
ax_broadband = [ax['broadband'], ax['broadband_pred']]

ax['broadband_isolation_pref'] = fig.add_subplot(gs[10:14, 0:4])
ax['broadband_isolation_pref_pred'] = fig.add_subplot(gs[10:14, 5:9])

ax['broadband_isolation_npref'] = fig.add_subplot(gs[10:14, 11:15])
ax['broadband_isolation_npref_pred'] = fig.add_subplot(gs[10:14, 16:20])

ax_broadband_isolation = [ax['broadband_isolation_pref'], ax['broadband_isolation_npref']]
ax_broadband_isolation_pred = [ax['broadband_isolation_pref_pred'], ax['broadband_isolation_npref_pred']]

ax['ISI_recovery'] = fig.add_subplot(gs[19:24, 0:6])  
ax['ISI_recovery_pred'] = fig.add_subplot(gs[19:224, 7:13])  

# ax['adaptation'] = fig.add_subplot(gs[24:28, 0:9])
ax['intercept'] = fig.add_subplot(gs[19:24, 15:20])

# seperate axes
sns.despine(offset=10)

# plot specs/adjustments
start = 50
end = 700
sep = 100

# fontsizes
fontsize_tick =        20
fontsize_legend =      20
fontsize_label =       20
# fontsize_title =       22

# set ticks
add = np.zeros((len(subtrials), len(tempCond)))
start_add = [0.016, 0.017]
add[:, 0] = start_add
for i in range(len(subtrials)):
    for j in range(1, len(tempCond)):
        add[i, j] = add[i, j - 1]*2

# initiate legend data holders
line = []
marker = []
marker_pred = []

# plot styles
alpha = np.linspace(0.2, 1, len(tempCond))
linestyle = ['solid', 'solid']
lw = 2

# metrics scatter points
s = 120

# y limits
y_lim_in_isolation = [[-0.2, 1.1], [-0.2, 1.1]]
y_lim_recovery = [20, 120]
y_lim_metrics = [-0.5, 1.7]

# compute timepoint of the start of both first and second pulse
start_1 = timepoints_twopulse[0, 0]

# adjust axes
ax['broadband'].spines['top'].set_visible(False)
ax['broadband'].spines['right'].set_visible(False)
ax['broadband'].tick_params(axis='both', which='major', labelsize=fontsize_tick)
# ax['broadband'].set_ylabel('Neural data', fontsize=fontsize_label)
ax['broadband'].axhline(0, color='grey', lw=0.5, alpha=0.5)
ax['broadband'].set_ylim(-0.2, 1.1)

ax['broadband_pred'].spines['top'].set_visible(False)
ax['broadband_pred'].spines['right'].set_visible(False)
ax['broadband_pred'].tick_params(axis='both', which='major',          labelsize=fontsize_tick)
# ax['broadband_pred'].set_xlabel('ISI (ms)',                           fontsize=fontsize_label)
# ax['broadband_pred'].set_ylabel('csDN model',         fontsize=fontsize_label)
ax['broadband_pred'].axhline(0, color='grey', lw=0.5, alpha=0.5)
ax['broadband_pred'].set_ylim(-0.2, 1.1)

for i in range(len(ax_broadband_isolation)):
    ax_broadband_isolation[i].spines['top'].set_visible(False)
    ax_broadband_isolation[i].spines['right'].set_visible(False)
    ax_broadband_isolation[i].tick_params(axis='both', which='major', labelsize=fontsize_tick)
    # ax_broadband_isolation[i].set_ylabel('Change in power\n(x-fold)', fontsize=fontsize_label)
    ax_broadband_isolation[i].axhline(0, color='grey', lw=0.5, alpha=0.5)
    # ax_broadband_isolation[i].set_xlabel('Time (ms)', fontsize=fontsize_label)
    # ax_broadband_isolation[i].set_title('Neural data', fontsize=fontsize_title)
    ax_broadband_isolation[i].set_ylim(y_lim_in_isolation[i])

for i in range(len(ax_broadband_isolation_pred)):
    ax_broadband_isolation_pred[i].spines['top'].set_visible(False)
    ax_broadband_isolation_pred[i].spines['right'].set_visible(False)
    ax_broadband_isolation_pred[i].tick_params(axis='both', which='major', labelsize=fontsize_tick)
    ax_broadband_isolation_pred[i].axhline(0, color='grey', lw=0.5, alpha=0.5)
    # ax_broadband_isolation_pred[i].set_xlabel('Time (ms)', fontsize=fontsize_label)
    # ax_broadband_isolation_pred[i].set_title('csDN model', fontsize=fontsize_title)
    ax_broadband_isolation_pred[i].set_ylim(y_lim_in_isolation[i])
    ax_broadband_isolation_pred[i].set_yticklabels([])

# ax['ISI_recovery'].set_xlabel('ISI (ms)', fontsize=fontsize_label)
# ax['ISI_recovery'].set_ylabel('Recovery (%)', fontsize=fontsize_label)
ax['ISI_recovery'].tick_params(axis='both', which='major', labelsize=fontsize_tick)
ax['ISI_recovery'].set_ylim(y_lim_recovery)
ax['ISI_recovery'].axhline(100, color='grey', linestyle='dotted')
ax['ISI_recovery'].set_xticks(add[1, :])
ax['ISI_recovery'].set_xticklabels(label_tempCond, rotation=45)
ax['ISI_recovery'].spines['top'].set_visible(False)
ax['ISI_recovery'].spines['right'].set_visible(False)

# ax['ISI_recovery_pred'].set_xlabel('ISI (ms)', fontsize=fontsize_label)
# ax['ISI_recovery_pred'].set_ylabel('Recovery (%)', fontsize=fontsize_label)
ax['ISI_recovery_pred'].spines['top'].set_visible(False)
ax['ISI_recovery_pred'].spines['right'].set_visible(False)
ax['ISI_recovery_pred'].tick_params(axis='both', which='major', labelsize=fontsize_tick)
ax['ISI_recovery_pred'].set_ylim(y_lim_recovery)
ax['ISI_recovery_pred'].axhline(100, color='grey', linestyle='dotted')
ax['ISI_recovery_pred'].set_xticks(add[1, :])
ax['ISI_recovery_pred'].set_yticks([])
ax['ISI_recovery_pred'].set_xticklabels(label_tempCond, rotation=45)

# ax['adaptation'].spines['top'].set_visible(False)
# ax['adaptation'].spines['right'].set_visible(False)
# ax['adaptation'].set_xticks(np.arange(len(subtrials))+0.1)
# ax['adaptation'].set_xlim(y_lim_metrics)
# ax['adaptation'].tick_params(axis='both', which='major', labelsize=fontsize_tick)
# ax['adaptation'].tick_params(axis='x', which='major', labelsize=fontsize_tick, rotation=45)
# ax['adaptation'].set_xticklabels(labels, fontsize=fontsize_label)
# ax['adaptation'].set_ylabel('Avg. recovery (%)', fontsize=fontsize_label)

ax['intercept'].spines['top'].set_visible(False)
ax['intercept'].spines['right'].set_visible(False)
ax['intercept'].set_xticks(np.arange(len(subtrials))+0.1)
ax['intercept'].set_xlim(y_lim_metrics)
ax['intercept'].tick_params(axis='both', which='major', labelsize=fontsize_tick)
ax['intercept'].set_xticklabels(labels, fontsize=fontsize_label, rotation=45)
# ax['intercept'].set_ylabel('Long-term recovery (%)', fontsize=fontsize_label)

# plot first pulse in isolation
max_data = [0, 0]
max_model = [0, 0]
for i in range(len(subtrials)):

    # retrieve data
    data_mean = np.zeros((len(stim_cat), len(t)))
    model_mean = np.zeros((len(stim_cat), len(t)))
    for k in range(len(stim_cat)):
        data_mean[k, :] = np.mean(broadband_pulse1_bootstrap[:, i, :], 0)
        model_mean[k, :] = np.mean(broadband_pulse1_bootstrap_pred[:, i, :], 0)
    data_mean = np.mean(data_mean, 0)
    model_mean = np.mean(model_mean, 0)

    # NEURAL DATA
    max_data[i] = max(data_mean[start_1 - start: start_1 - start + time_window])
    data_temp = gaussian_filter1d(data_mean[start_1 - start: start_1 - start + time_window]/max(data_mean[start_1 - start: start_1 - start + time_window]), sigma=10)
    ax_broadband_isolation[i].plot(np.arange(time_window), data_temp, color='black')

    # MODEL
    max_model[i] = max(model_mean[start_1 - start: start_1 - start + time_window])
    model_temp = gaussian_filter1d(model_mean[start_1 - start: start_1 - start + time_window]/max(model_mean[start_1 - start: start_1 - start + time_window]), sigma=10)
    ax_broadband_isolation_pred[i].plot(np.arange(time_window), model_temp, color='black')

# plot rest of figure
t_zero          = np.argwhere(t > 0)[0][0]
t_twohundred    = np.argwhere(t > 0.5)[0][0]

x_label_single = ['0', '500']

xtick_idx = []
for i in range(len(tempCond)):

    # append x-tick
    xtick_idx = xtick_idx + ([i*(end+sep) + t_zero, i*(end+sep) + t_twohundred])

    # compute timepoint of the start of both first and second pulse
    start_2 = timepoints_twopulse[i, 2]

    for j in range(len(subtrials)):

        # plot stimulus timecourse
        if (i == 0) & (j == 0):
            ax_broadband[j].axvspan(i*(end+sep) - start + timepoints_twopulse[i, 0], i*(
                end+sep) - start + timepoints_twopulse[i, 1], facecolor='grey', alpha=0.2, label='stimulus')
            ax_broadband[j].axvspan(i*(end+sep) - start + timepoints_twopulse[i, 2], i*(
                end+sep) - start + timepoints_twopulse[i, 3], facecolor='grey', alpha=0.2)
        else:
            ax_broadband[j].axvspan(i*(end+sep) - start + timepoints_twopulse[i, 0], i*(
                end+sep) - start + timepoints_twopulse[i, 1], facecolor='grey', alpha=0.2)
            ax_broadband[j].axvspan(i*(end+sep) - start + timepoints_twopulse[i, 2], i*(
                end+sep) - start + timepoints_twopulse[i, 3], facecolor='grey', alpha=0.2)
    
        # retrieve data
        data_mean = gaussian_filter1d(np.mean(broadband_bootstrap[:, j, i, :], 0), 10)
        model_mean = gaussian_filter1d(np.mean(broadband_bootstrap_pred[:, j, i, :], 0), 10)

        # plot model and data
        if i == 0:
            ax_broadband[0].plot(np.arange(end - start)+i*(end+sep), data_mean[start:end]/max(data_mean[start:end]), color=color[j], label=labels[j].lower(), lw=lw)
            ax_broadband[1].plot(np.arange(end - start)+i*(end+sep), model_mean[start:end]/max(model_mean[start:end]), color=color[j], label=labels[j].lower(), lw=lw)
        else:
            ax_broadband[0].plot(np.arange(end - start)+i*(end+sep), data_mean[start:end]/max(data_mean[start:end]), color=color[j], lw=lw)
            ax_broadband[1].plot(np.arange(end - start)+i*(end+sep), model_mean[start:end]/max(model_mean[start:end]), color=color[j], lw=lw)

        # plot stimulus in isolation
        # NEURAL DATA
        data_mean = gaussian_filter1d(np.mean(broadband_pulse2_bootstrap[:, j, i, :], 0)/max_data[j], 10)
        ax_broadband_isolation[j].plot(np.arange(time_window), data_mean[start_2 - start: start_2 - start + time_window], color=color[j], alpha=alpha[i])

        # MODEL
        model_mean = gaussian_filter1d(np.mean(broadband_pulse2_bootstrap_pred[:, j, i, :], 0)/max_model[j], 10)
        ax_broadband_isolation_pred[j].plot(np.arange(time_window), model_mean[start_2 - start: start_2 - start + time_window], color=color[j], alpha=alpha[i])

        # NEURAL DATA
        data_temp =  ISI_recovery_avg[j, i]*100
        if i == 0:
            marker_temp = ax['ISI_recovery'].scatter(add[j, i], data_temp, color=color[j], edgecolor='white', s=150)
            marker.append(marker_temp)
        else:
            ax['ISI_recovery'].scatter(add[j, i], data_temp, color=color[j], edgecolor='white', s=150)

        # MODEL
        model_temp = ISI_recovery_pred_avg[j, i]*100
        if i == 0:
            marker_pred_temp = ax['ISI_recovery_pred'].scatter(add[j, i], model_temp, color=color[j], edgecolor='white', marker='^', s=150)
            marker_pred.append(marker_pred_temp)
        else:
            marker_pred_temp = ax['ISI_recovery_pred'].scatter(add[j, i], model_temp, color=color[j], edgecolor='white', marker='^', s=150)

        # plot CI, NEURAL DATA
        error_min = ISI_recovery_CI[j, i, 0]*100
        error_max = ISI_recovery_CI[j, i, 1]*100
        ax['ISI_recovery'].plot([add[j, i], add[j, i]], [error_min, error_max], color='black', zorder=1)

        # plot CI, MODEL
        error_min = ISI_recovery_pred_CI[j, i, 0]*100
        error_max = ISI_recovery_pred_CI[j, i, 1]*100
        ax['ISI_recovery_pred'].plot([add[j, i], add[j, i]], [error_min, error_max], color='black', zorder=1)

# # plot bargraph (slope log-linear fit)
# error_min_temp = adaptation_CI[:, 0]*100
# error_max_temp = adaptation_CI[:, 1]*100
# ax['adaptation'].scatter(np.arange(len(subtrials)), adaptation_avg*100, color=color, s=s)
# ax['adaptation'].plot([np.arange(len(subtrials)), np.arange(len(subtrials))], [error_min_temp, error_max_temp], color='k', zorder=1)

# error_min_temp = adaptation_pred_CI[:, 0]*100
# error_max_temp = adaptation_pred_CI[:, 1]*100
# ax['adaptation'].scatter(np.arange(len(subtrials))+0.2, adaptation_pred_avg*100, color=color, s=s, marker='^')
# ax['adaptation'].plot([np.arange(len(subtrials))+0.2, np.arange(len(subtrials))+0.2], [error_min_temp, error_max_temp], color='black', zorder=1)

# plot bargraph (slope log-linear fit)
error_min_temp = intercept_CI[:, 0]*100
error_max_temp = intercept_CI[:, 1]*100
ax['intercept'].scatter(np.arange(len(subtrials)), intercept_avg*100, color=color, s=s)
ax['intercept'].plot([np.arange(len(subtrials)), np.arange(len(subtrials))], [error_min_temp, error_max_temp], color='k', zorder=1)

error_min_temp = intercept_pred_CI[:, 0]*100
error_max_temp = intercept_pred_CI[:, 1]*100
ax['intercept'].scatter(np.arange(len(subtrials))+0.2, intercept_pred_avg*100, color=color, s=s, marker='^')
ax['intercept'].plot([np.arange(len(subtrials))+0.2, np.arange(len(subtrials))+0.2], [error_min_temp, error_max_temp], color='black', zorder=1)

# plot ISI recovery curve
for i in range(len(subtrials)):

    # retrieve data
    data_temp =  np.mean(ISI_recovery_log[:, i, :], 0)*100
    model_temp = np.mean(ISI_recovery_log_pred[:, i, :], 0)*100

    # plot data
    line_temp, = ax['ISI_recovery'].plot(t1_plot/1000, data_temp, color=color[i], zorder=-5, linestyle=linestyle[i])
    ax['ISI_recovery_pred'].plot(t1_plot/1000, model_temp, color=color[i], zorder=-5, linestyle=linestyle[i])
    line.append(line_temp)

# add legend
vertical_line1 = lines.Line2D([], [], color='black', marker='|', linestyle='None', markersize=30)
vertical_line2 = lines.Line2D([], [], color='black', marker='|', linestyle='None', markersize=30) 
vertical_line1_pred = lines.Line2D([], [], color='black', marker='|', linestyle='None', markersize=30)
vertical_line2_pred = lines.Line2D([], [], color='black', marker='|', linestyle='None', markersize=30)    
ax['ISI_recovery'].legend([(vertical_line1, marker[0], line[0]), (vertical_line2 , marker[1], line[1])], [labels[0].lower(), labels[1].lower()], loc='upper left', frameon=False, fontsize=fontsize_legend)
ax['ISI_recovery_pred'].legend([(vertical_line1_pred, marker_pred[0], line[0]), (vertical_line2_pred, marker_pred[1], line[1])], [labels[0].lower(), labels[1].lower()], loc='upper left', frameon=False, fontsize=fontsize_legend)

# add legend and xticks
ax['broadband'].legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left", borderaxespad=0, ncol=4, frameon=False, fontsize=fontsize_legend)

ax['broadband'].set_xticks(xtick_idx)
ax['broadband'].set_xticklabels([])
ax['broadband_pred'].set_xticks(xtick_idx)
ax['broadband_pred'].set_xticklabels(np.tile(x_label_single, 6))

# save figure
fig.align_ylabels()
plt.tight_layout()
plt.savefig(dir+'mkFigure/Fig8_' + str(threshold_d_prime).replace('.', '-') + '.png', dpi=300, bbox_inches='tight')
plt.savefig(dir+'mkFigure/Fig8_' + str(threshold_d_prime).replace('.', '-') +'.svg', format='svg', bbox_inches='tight')
# plt.show()

############################################################################
######################################################## STATISTICAL TESTING

metrics = ['Avg. adaptation', 'Intercept']
for i in range(2):

    print(30*'--')
    print(metrics[i])

    if i == 0: # avg. adaptation

        print('#'*30)
        print('NEURAL DATA')

        # early vs. ventral
        sample1 = adaptation_medians[0, :]
        sample2 = adaptation_medians[1, :]
        param_diffs = sample1 - sample2

        p = np.min([len(param_diffs[param_diffs < 0]), len(param_diffs[param_diffs > 0])])/B_repetitions
        print('p-value: ', p)

        print('#'*30)
        print('MODEL')

        # early vs. ventral
        sample1 = adaptation_pred_medians[0, :]
        sample2 = adaptation_pred_medians[1, :]
        param_diffs = sample1 - sample2

        p = np.min([len(param_diffs[param_diffs < 0]), len(param_diffs[param_diffs > 0])])/B_repetitions
        print('p-value: ', p)

    elif i == 1:

        print('#'*30)
        print('NEURAL DATA')

        # early vs. ventral
        sample1 = intercept_medians[0, :]
        sample2 = intercept_medians[1, :]
        param_diffs = sample1 - sample2

        p = np.min([len(param_diffs[param_diffs < 0]), len(param_diffs[param_diffs > 0])])/B_repetitions
        print('p-value: ', p)


        print('#'*30)
        print('MODEL')

        # early vs. ventral
        sample1 = intercept_pred_medians[0, :]
        sample2 = intercept_pred_medians[1, :]
        param_diffs = sample1 - sample2

        p = np.min([len(param_diffs[param_diffs < 0]), len(param_diffs[param_diffs > 0])])/B_repetitions
        print('p-value: ', p)
