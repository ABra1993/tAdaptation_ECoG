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
from modelling_utils_fitObjective import model_csDN, model_DN, model_csDN_withoutGeneralScaling, OF_ISI_recovery_log


"""

Author: A. Brands

"""

############################################################################################## ADAPT CODE HERES
##############################################################################################################
##############################################################################################################
##############################################################################################################

# define root directory
file = open('setDir.txt')
dir = file.readline()

# specifiy the trial types
# img_type = 'all'
# img_type = 'preferred'
img_type = 'nonpreferred'

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

# trials used for computing the ratio of the second and first peak
trials = ['onepulse-4', 'twopulse', 'twopulse_repeat']
axis = [None, 'TEMP', 'TEMP']

# define model
# model = 'DN'
model = 'csDN'
# model = 'csDN_withoutGeneralScaling'

# retrieve parameters
params_names, _, _, _ = paramInit(model)
sample_rate = 512

# create stimulus timecourse
stim_twopulse = np.zeros((len(tempCond), len(t))) 
for i in range(len(tempCond)):
    stim_twopulse[i, :] = generate_stimulus_timecourse(trials[2], i, dir)
stim_onepulse = generate_stimulus_timecourse('onepulse', 3, dir)

# visual areas (labels)
VA = ['V1-V3', 'VOTC', 'LOTC']
VA_n = np.zeros(len(VA))  # number of electrodes
VA_labels = ['V1-V3', 'VOTC', 'LOTC']
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

# fit curve for recovery of adaptation initial parameter values
p0 = [1, 0]

# timescales to fit and plot curves
t1_plot = np.linspace(min(tempCond), max(tempCond), 1000)

############################################################################
################################################################### ANALYSIS

# initiate dataframes to store data
broadband = []
broadband_pulse1 = []
broadband_pulse2 = []

broadband_bootstrap = []
broadband_pulse1_bootstrap = []
broadband_pulse2_pred_bootstrap = []

broadband_pred = []
broadband_pulse1_pred = []
broadband_pulse2_pred = []

broadband_pred_bootstrap = []
broadband_pulse1_pred_bootstrap = []
broadband_pulse2_bootstrap = []

ISI_recovery = []
ISI_recovery_bootstrap = []
ISI_recovery_avg = np.zeros((len(VA), len(tempCond)))
ISI_recovery_CI = np.zeros((len(VA), len(tempCond), 2))
ISI_recovery_log = np.zeros((len(VA), len(t1_plot)))

ISI_recovery_pred = []
ISI_recovery_pred_bootstrap = []
ISI_recovery_pred_avg = np.zeros((len(VA), len(tempCond)))
ISI_recovery_pred_CI = np.zeros((len(VA), len(tempCond), 2))
ISI_recovery_pred_log = np.zeros((len(VA), len(t1_plot)))

adaptation = []
adaptation_avg = np.zeros(len(VA))
adaptation_CI = np.zeros((len(VA), 2))

adaptation_pred = []
adaptation_pred_avg = np.zeros(len(VA))
adaptation_pred_CI = np.zeros((len(VA), 2))

intercept = []
intercept_avg = np.zeros(len(VA))
intercept_CI = np.zeros((len(VA), 2))

intercept_pred = []
intercept_pred_avg = np.zeros(len(VA))
intercept_pred_CI = np.zeros((len(VA), 2))

# save medians for statistical testing
adaptation_medians = np.zeros((len(VA), B_repetitions))
adaptation_pred_medians = np.zeros((len(VA), B_repetitions))

intercept_medians = np.zeros((len(VA), B_repetitions))
intercept_pred_medians = np.zeros((len(VA), B_repetitions))

current_subject = ''
count_VA = 0
for key, value in VA_name_idx.items():

    # count number of electrodes
    n_electrodes = len(value)
    VA_n[count_VA] = n_electrodes

    # initiat dataframes
    broadband_current = np.zeros((n_electrodes, len(tempCond), len(t)))
    broadband_pred_current = np.zeros((n_electrodes, len(tempCond), len(t)))

    broadband_pulse1_current = np.zeros((n_electrodes, len(t)))
    broadband_pulse1_pred_current = np.zeros((n_electrodes, len(t)))

    broadband_pulse2_current = np.zeros((n_electrodes, len(tempCond), len(t)))
    broadband_pulse2_pred_current = np.zeros((n_electrodes, len(tempCond), len(t)))

    broadband_bootstrap_current = np.zeros((B_repetitions, len(tempCond), len(t)))
    broadband_bootstrap_pred_current = np.zeros((B_repetitions, len(tempCond), len(t)))

    broadband_pulse1_bootstrap_current = np.zeros((B_repetitions, len(t)))
    broadband_pulse1_bootstrap_pred_current = np.zeros((B_repetitions, len(t)))

    broadband_pulse2_bootstrap_current = np.zeros((B_repetitions, len(tempCond), len(t)))
    broadband_pulse2_bootstrap_pred_current = np.zeros((B_repetitions, len(tempCond), len(t)))

    ISI_recovery_current = np.zeros((B_repetitions, len(tempCond)))
    ISI_recovery_pred_current = np.zeros((B_repetitions, len(tempCond)))

    ISI_recovery_bootstrap_current = np.zeros((B_repetitions, len(tempCond)))
    ISI_recovery_bootstrap_pred_current = np.zeros((B_repetitions, len(tempCond)))

    adaptation_current = np.zeros(B_repetitions)
    adaptation_pred_current = np.zeros(B_repetitions)

    intercept_current = np.zeros(B_repetitions)
    intercept_pred_current = np.zeros(B_repetitions)

    # iterate over electrodes
    for i in range(n_electrodes):
    # for i in range(1):

        # retrieve info current electrode
        subject = electrodes_visuallyResponsive.subject[value[i]]
        electrode_name = electrodes_visuallyResponsive.electrode[value[i]]
        electrode_idx = int(electrodes_visuallyResponsive.electrode_idx[value[i]])

        # print progress
        print(30*'-')
        print(key)
        print(30*'-')
        print('Computing trials for ' + subject + ', electrode ' + electrode_name + ' (' + str(i+1) + '/' + str(n_electrodes) + ')')

        # retrieve model parameters for current electrode
        temp = pd.read_csv(dir+'modelFit/visuallyResponsive/' + subject + '_' + electrode_name + '/param_' + model + '.txt', header=0, delimiter=' ', index_col=0)
        temp.reset_index(inplace=True,drop=True)
        params_current = list(temp.loc[0, params_names])

        if current_subject != subject:

            # update current subject
            current_subject = subject

            # import info
            _, events, channels, _ = import_info(subject, dir)

            # import excluded trials
            excluded_epochs = pd.read_csv(dir+'subject_data/' + subject + '/excluded_epochs.txt', sep=' ', header=0, dtype=int)

        # extract data
        epochs_b = import_epochs(subject, electrode_idx, dir)
        index_epochs_b = [j for j in range(len(events)) if excluded_epochs.iloc[electrode_idx, j+1] == 1]
        epochs_b.iloc[:, index_epochs_b] = np.nan

        # extract data
        cat = None
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
            event_idx = select_events_repetitionTrials(events, tempCond, preference, cat)
        
        else:

            # select events
            event_idx = select_events_repetitionTrials(events, tempCond, preference)  

        # get onepulse trials
        # NEURAL DATA
        data_first_pulse = estimate_first_pulse(t, epochs_b, event_idx, timepoints_twopulse)
        broadband_pulse1_current[i, :] = data_first_pulse

        # for 5 remaining img classes
        if model == 'csDN':
            if preference == 0:     # all  
                temp = np.zeros((len(stim_cat), len(t))) # all image categories expect preferred
                for l in range(len(stim_cat)):
                    _, temp[l, :] = model_csDN(stim_onepulse, 'onepulse', 3, stim_cat[l], sample_rate, params_current, dir) 
                broadband_pulse1_pred_current[i, :] = np.mean(temp, 0)
            elif preference == 1:   # preferred
                _, broadband_pulse1_pred_current[i, :] = model_csDN(stim_onepulse, 'onepulse', 3, cat[0], sample_rate, params_current, dir)  
            elif preference == 2:   # nonpreferred
                temp = np.zeros((len(stim_cat)-1, len(t))) # all image categories expect preferred
                num = 0
                for l in range(len(stim_cat)):
                    if stim_cat[l] != cat[0]:
                        _, temp[num, :] = model_csDN(stim_onepulse, 'onepulse', 3, stim_cat[l], sample_rate, params_current, dir) 
                        num+=1
                broadband_pulse1_pred_current[i, :] = np.mean(temp, 0)
        elif model == 'csDN_withoutGeneralScaling':
            if preference == 0:     # all  
                temp = np.zeros((len(stim_cat), len(t))) # all image categories expect preferred
                for l in range(len(stim_cat)):
                    _, temp[l, :] = model_csDN_withoutGeneralScaling(stim_onepulse, 'onepulse', 3, stim_cat[l], sample_rate, params_current, dir) 
                broadband_pulse1_pred_current[i, :] = np.mean(temp, 0)
            elif preference == 1:   # preferred
                _, broadband_pulse1_pred_current[i, :] = model_csDN_withoutGeneralScaling(stim_onepulse, 'onepulse', 3, cat[0], sample_rate, params_current, dir)  
            elif preference == 2:   # nonpreferred
                temp = np.zeros((len(stim_cat)-1, len(t))) # all image categories expect preferred
                num = 0
                for l in range(len(stim_cat)):
                    if stim_cat[l] != cat[0]:
                        _, temp[num, :] = model_csDN_withoutGeneralScaling(stim_onepulse, 'onepulse', 3, stim_cat[l], sample_rate, params_current, dir) 
                        num+=1
                broadband_pulse1_pred_current[i, :] = np.mean(temp, 0)
        elif model == 'DN':
            broadband_pulse1_pred_current[i, :] = model_DN(stim_onepulse[j, :], sample_rate, params_current)

        # retrieve broadband data
        for j in range(len(tempCond)):

            # select twouplse
            # NEURAL DATA
            broadband_current[i, j, :] = np.nanmean(epochs_b[event_idx[2][j]], axis=1)

            # MODEL
            if model == 'csDN':
                if preference == 0:     # all  
                    temp = np.zeros((len(stim_cat), len(t))) # all image categories expect preferred
                    for l in range(len(stim_cat)):
                        _, temp[l, :] = model_csDN(stim_twopulse[j, :], 'twopulse_repeat', j, stim_cat[l], sample_rate, params_current, dir) 
                    broadband_pred_current[i, j, :] = np.mean(temp, 0)
                elif preference == 1:
                    _, broadband_pred_current[i, j, :] = model_csDN(stim_twopulse[j, :], 'twopulse_repeat', j, cat[0], sample_rate, params_current, dir)  
                elif preference == 2:
                    temp = np.zeros((len(stim_cat)-1, len(t))) # all image categories expect preferred
                    num = 0
                    for l in range(len(stim_cat)):
                        if stim_cat[l] != cat[0]:
                            _, temp[num, :] = model_csDN(stim_twopulse[j, :], 'twopulse_repeat', j, stim_cat[l], sample_rate, params_current, dir) 
                            num+=1
                    broadband_pred_current[i, j, :] = np.mean(temp, 0)
            elif model == 'csDN_withoutGeneralScaling':
                if preference == 0:     # all  
                    temp = np.zeros((len(stim_cat), len(t))) # all image categories expect preferred
                    for l in range(len(stim_cat)):
                        _, temp[l, :] = model_csDN_withoutGeneralScaling(stim_twopulse[j, :], 'twopulse_repeat', j, stim_cat[l], sample_rate, params_current, dir) 
                    broadband_pred_current[i, j, :] = np.mean(temp, 0)
                elif preference == 1:
                    _, broadband_pred_current[i, j, :] = model_csDN_withoutGeneralScaling(stim_twopulse[j, :], 'twopulse_repeat', j, cat[0], sample_rate, params_current, dir)  
                elif preference == 2:
                    temp = np.zeros((len(stim_cat)-1, len(t))) # all image categories expect preferred
                    num = 0
                    for l in range(len(stim_cat)):
                        if stim_cat[l] != cat[0]:
                            _, temp[num, :] = model_csDN_withoutGeneralScaling(stim_twopulse[j, :], 'twopulse_repeat', j, stim_cat[l], sample_rate, params_current, dir) 
                            num+=1
                    broadband_pred_current[i, j, :] = np.mean(temp, 0)
            elif model == 'DN':
                broadband_pred_current[i, j, :] = model_DN(stim_twopulse[j, :], sample_rate, params_current)
        
            # compute isolated second pulse
            # NEURAL DATA
            broadband_pulse2_current[i, j, :] = broadband_current[i, j, :] - data_first_pulse

            # MODEL
            broadband_pulse2_pred_current[i, j, :] = broadband_pred_current[i, j, :] - broadband_pulse1_pred_current[i, :]

    # perform bootstrap over broadband timecourse
    ISI_recovery_log_current = np.zeros((B_repetitions, len(t1_plot)))
    ISI_recovery_log_pred_current = np.zeros((B_repetitions, len(t1_plot)))
    for i in range(B_repetitions):

        # draw random sample
        idx_temp = np.arange(n_electrodes)
        n_samples = len(idx_temp)
        boot = resample(idx_temp, replace=True, n_samples=n_samples)

        # compute first pulse
        data_mean = np.zeros((len(boot), len(t)))
        model_mean = np.zeros((len(boot), len(t)))
        for l in range(len(boot)):
            data_mean[l, :] = broadband_pulse1_current[boot[l], :]
            model_mean[l, :] = broadband_pulse1_pred_current[boot[l], :]
        broadband_pulse1_bootstrap_current[i, :] = np.nanmean(data_mean, 0)
        broadband_pulse1_bootstrap_pred_current[i, :] = np.nanmean(model_mean, 0)

        # compute tempCond recovery
        adaptation_temp = np.zeros(len(tempCond))
        adaptation_pred_temp = np.zeros(len(tempCond))
        
        for j in range(len(tempCond)):

            # compute degree of recovery
            start_firstpulse        = timepoints_onepulse[0, 0]
            start_second_pulse      = timepoints_twopulse[j, 2]

            # retrieve broadband
            # NEURAL DATA
            data = np.zeros((len(boot), len(t)))
            data_pulse1 = np.zeros((len(boot), len(t)))
            for l in range(len(boot)):
                data[l, :] = broadband_current[boot[l], j, :]
                data_pulse1[l, :] = broadband_pulse2_current[boot[l], j, :]
            broadband_bootstrap_current[i, j, :] = np.nanmean(data, 0)
            broadband_pulse2_bootstrap_current[i, j, :] = np.nanmean(data_pulse1, 0)

            # compute degree of recovery
            AUC1 = np.trapz(broadband_pulse1_bootstrap_current[i, :][start_firstpulse: start_firstpulse+time_window])
            AUC2 = np.trapz(broadband_pulse2_bootstrap_current[i, j, :][start_second_pulse:start_second_pulse+time_window])

            ISI_recovery_bootstrap_current[i, j] = AUC2/AUC1
            adaptation_temp[j] = AUC2/AUC1

            # MODEL
            pred = np.zeros((len(boot), len(t)))
            pred_pulse1 = np.zeros((len(boot), len(t)))
            for l in range(len(boot)):
                pred[l, :] = broadband_pred_current[boot[l], j, :]
                pred_pulse1[l, :] = broadband_pulse2_pred_current[boot[l], j, :]
            broadband_bootstrap_pred_current[i, j, :] = np.nanmean(pred, 0)
            broadband_pulse2_bootstrap_pred_current[i, j, :] = np.nanmean(pred_pulse1, 0)
            
            # compute degree of recovery
            AUC1 = np.trapz(broadband_pulse1_bootstrap_pred_current[i, :][start_firstpulse: start_firstpulse+time_window])
            AUC2 = np.trapz(broadband_pulse2_bootstrap_pred_current[i, j, :][start_second_pulse:start_second_pulse+time_window])

            ISI_recovery_bootstrap_pred_current[i, j] = AUC2/AUC1
            adaptation_pred_temp[j] = AUC2/AUC1

        # NEURAL DATA
        # popt, _ = curve_fit(OF_ISI_recovery_log, tempCond, tempCond_recovery_bootstrap_current[i, :], p0, maxfev=100000) #, bounds=((0, 0), (np.inf, np.inf)))
        popt, _ = curve_fit(OF_ISI_recovery_log, tempCond/1000, adaptation_temp, p0, maxfev=1000) #, bounds=((0, 0), (np.inf, np.inf)))
        ISI_recovery_log_current[i, :] = OF_ISI_recovery_log(t1_plot/1000, *popt)
        intercept_current[i] =  popt[0]
        intercept_medians[count_VA, i] = popt[0]
        adaptation_current[i] = np.mean(ISI_recovery_bootstrap_current[i, :])
        adaptation_medians[count_VA, i] = np.mean(ISI_recovery_bootstrap_current[i, :])

        # MODEL
        popt, _ = curve_fit(OF_ISI_recovery_log, tempCond/1000, adaptation_pred_temp, p0, maxfev=1000) #, bounds=((0, 0), (np.inf, np.inf)))
        ISI_recovery_log_pred_current[i, :] = OF_ISI_recovery_log(t1_plot/1000, *popt)
        intercept_pred_current[i] = popt[0]
        intercept_pred_medians[count_VA, i] = popt[0]
        adaptation_pred_current[i] = np.mean(ISI_recovery_bootstrap_pred_current[i, :])
        adaptation_pred_medians[count_VA, i] = np.mean(ISI_recovery_bootstrap_pred_current[i, :])

    # compute spread
    # NEURAL DATA
    adaptation_avg[count_VA] = np.mean(adaptation_current)
    adaptation_CI[count_VA, :] = np.nanpercentile(adaptation_current, [CI_low, CI_high])

    intercept_avg[count_VA] = np.mean(intercept_current)
    intercept_CI[count_VA, :] = np.nanpercentile(intercept_current, [CI_low, CI_high])

    ISI_recovery_log[count_VA] = np.mean(ISI_recovery_log_current, 0)
    for i in range(len(tempCond)):
        ISI_recovery_avg[count_VA, i] = np.mean(ISI_recovery_bootstrap_current[:, i])
        ISI_recovery_CI[count_VA, i, :] = np.nanpercentile(ISI_recovery_bootstrap_current[:, i], [CI_low, CI_high])

    # MODEL
    adaptation_pred_avg[count_VA] = np.mean(adaptation_pred_current)
    adaptation_pred_CI[count_VA, :] = np.nanpercentile(adaptation_pred_current, [CI_low, CI_high])

    intercept_pred_avg[count_VA] = np.mean(intercept_pred_current)
    intercept_pred_CI[count_VA, :] = np.nanpercentile(intercept_pred_current, [CI_low, CI_high])

    ISI_recovery_pred_log[count_VA] = np.mean(ISI_recovery_log_pred_current, 0)
    for i in range(len(tempCond)):
        ISI_recovery_pred_avg[count_VA, i] = np.mean(ISI_recovery_bootstrap_pred_current[:, i])
        ISI_recovery_pred_CI[count_VA, i, :] = np.nanpercentile(ISI_recovery_bootstrap_pred_current[:, i], [CI_low, CI_high])

    # append dataframes
    # NEURAL DATA
    broadband.append(broadband_current)
    broadband_pulse1.append(broadband_pulse1_current)
    broadband_pulse2.append(broadband_pulse2_current)

    broadband_bootstrap.append(broadband_bootstrap_current)
    broadband_pulse1_bootstrap.append(broadband_pulse1_bootstrap_current)
    broadband_pulse2_bootstrap.append(broadband_pulse2_bootstrap_current)

    ISI_recovery.append(ISI_recovery_current)
    ISI_recovery_bootstrap.append(ISI_recovery_bootstrap_current)

    adaptation.append(adaptation_current)
    intercept.append(intercept_current)

    # MODEL
    broadband_pred.append(broadband_pred_current)
    broadband_pulse1_pred.append(broadband_pulse1_pred_current)
    broadband_pulse2_pred.append(broadband_pulse2_pred_current)

    broadband_pred_bootstrap.append(broadband_bootstrap_pred_current)
    broadband_pulse1_pred_bootstrap.append(broadband_pulse1_bootstrap_pred_current)
    broadband_pulse2_pred_bootstrap.append(broadband_pulse2_bootstrap_pred_current)

    ISI_recovery_pred.append(ISI_recovery_pred_current)
    ISI_recovery_pred_bootstrap.append(ISI_recovery_bootstrap_pred_current)

    adaptation_pred.append(adaptation_pred_current)
    intercept_pred.append(intercept_pred_current)

    # increment count
    count_VA+=1


############################################################################
############################################################### PLOT RESULTS

# initiate figure
fig = plt.figure(figsize=(18, 24))
gs = fig.add_gridspec(36, 20)
ax = dict()

# initiate plots
ax['broadband'] = fig.add_subplot(gs[0:3, 0:20])
ax['broadband_pred'] = fig.add_subplot(gs[4:7, 0:20])

ax['broadband_isolation_E'] = fig.add_subplot(gs[10:14, 0:3])
ax['broadband_isolation_V'] = fig.add_subplot(gs[10:14, 7:10])
ax['broadband_isolation_L'] = fig.add_subplot(gs[10:14, 14:17])
ax_broadband_isolation = [ax['broadband_isolation_E'], ax['broadband_isolation_V'], ax['broadband_isolation_L']]

ax['broadband_isolation_E_pred'] = fig.add_subplot(gs[10:14, 3:6])
ax['broadband_isolation_V_pred'] = fig.add_subplot(gs[10:14, 10:13])
ax['broadband_isolation_L_pred'] = fig.add_subplot(gs[10:14, 17:20])
ax_broadband_isolation_pred = [ax['broadband_isolation_E_pred'], ax['broadband_isolation_V_pred'], ax['broadband_isolation_L_pred']]

ax['ISI_recovery'] = fig.add_subplot(gs[17:24, 0:9])  
ax['ISI_recovery_pred'] = fig.add_subplot(gs[17:24, 11:20])  

ax['adaptation'] = fig.add_subplot(gs[30:36, 0:8])
ax['intercept'] = fig.add_subplot(gs[30:36, 12:20])

# seperate axes
sns.despine(offset=10)

# set ticks
add = np.zeros((len(VA), len(tempCond)))
start_add = [0.016, 0.017, 0.018]
add[:, 0] = start_add
for i in range(len(VA)):
    for j in range(1, len(tempCond)):
        add[i, j] = add[i, j - 1]*2

# plot specs/adjustments
start = 50
end = 700
sep = 100

# fontsizes
fontsize_tick =        20
fontsize_legend =      20
fontsize_label =       20

# initiate legend data holders
line = []
marker = []
marker_pred = []

# plot styles
alpha = np.linspace(0.2, 1, len(tempCond))
linestyle = ['solid', 'solid', 'solid']
lw = 2

# metrics scatter points
s = 120

# y limits
y_lim_in_isolation = [[-0.2, 1.1], [-0.2, 1.1], [-0.2, 1.1]]
y_lim_recovery = [25, 120]
y_lim_metrics = [-0.5, 2.7]

# compute timepoint of the start of both first and second pulse
start_1 = timepoints_twopulse[0, 0]

# adjust axes
ax['broadband'].spines['top'].set_visible(False)
ax['broadband'].spines['right'].set_visible(False)
ax['broadband'].tick_params(axis='both', which='major', labelsize=fontsize_tick)
ax['broadband'].axhline(0, color='grey', lw=0.5, alpha=0.5)
ax['broadband'].set_ylim(-0.2, 1.1)
# ax['broadband'].set_ylabel('Neural data', fontsize=fontsize_label)

ax['broadband_pred'].spines['top'].set_visible(False)
ax['broadband_pred'].spines['right'].set_visible(False)
ax['broadband_pred'].tick_params(axis='both', which='major', labelsize=fontsize_tick)
ax['broadband_pred'].axhline(0, color='grey', lw=0.5, alpha=0.5)
ax['broadband_pred'].set_ylim(-0.2, 1.1)
# ax['broadband_pred'].set_xlabel('ISI (ms)', fontsize=fontsize_label)
# ax['broadband_pred'].set_ylabel('DN model', fontsize=fontsize_label)

for i in range(len(ax_broadband_isolation)):
    ax_broadband_isolation[i].spines['top'].set_visible(False)
    ax_broadband_isolation[i].spines['right'].set_visible(False)
    ax_broadband_isolation[i].tick_params(axis='both', which='major', labelsize=fontsize_tick)
    ax_broadband_isolation[i].axhline(0, color='grey', lw=0.5, alpha=0.5)
    ax_broadband_isolation[i].set_xlabel('Time (ms)', fontsize=fontsize_label)
    ax_broadband_isolation[i].set_ylim(y_lim_in_isolation[i])
    ax_broadband_isolation[i].set_xticks([0, 200])
    # ax_broadband_isolation[i].set_title('Neural data', fontsize=fontsize_title)
    # if i == 0:
        # ax_broadband_isolation[i].set_ylabel('Change in power (x-fold)', fontsize=fontsize_label)

for i in range(len(ax_broadband_isolation_pred)):
    ax_broadband_isolation_pred[i].spines['top'].set_visible(False)
    ax_broadband_isolation_pred[i].spines['right'].set_visible(False)
    ax_broadband_isolation_pred[i].tick_params(axis='both', which='major', labelsize=fontsize_tick)
    ax_broadband_isolation_pred[i].axhline(0, color='grey', lw=0.5, alpha=0.5)
    ax_broadband_isolation_pred[i].set_xlabel('Time (ms)', fontsize=fontsize_label)
    ax_broadband_isolation_pred[i].set_ylim(y_lim_in_isolation[i])
    ax_broadband_isolation_pred[i].set_yticklabels([])
    ax_broadband_isolation_pred[i].set_xticks([0, 200])
    # ax_broadband_isolation_pred[i].set_title('DN model', fontsize=fontsize_title)

# ax['ISI_recovery'].set_xlabel('ISI (ms)', fontsize=fontsize_label)
# ax['ISI_recovery'].set_ylabel('Recovery (%)', fontsize=fontsize_label)
ax['ISI_recovery'].set_ylim(y_lim_recovery)
ax['ISI_recovery'].axhline(100, color='grey', linestyle='dotted')
ax['ISI_recovery'].tick_params(axis='both', which='major', labelsize=fontsize_tick)
ax['ISI_recovery'].set_xticks(add[1, :])
ax['ISI_recovery'].set_xticklabels(label_tempCond, rotation=45)
ax['ISI_recovery'].spines['top'].set_visible(False)
ax['ISI_recovery'].spines['right'].set_visible(False)
if preference == 0:
    ax['ISI_recovery'].set_ylim(20, 120)
elif preference == 1:
    ax['ISI_recovery'].set_ylim(10, 120)
elif preference == 2:
    ax['ISI_recovery'].set_ylim(20, 120)

# ax['ISI_recovery_pred'].set_xlabel('ISI (ms)', fontsize=fontsize_label)
# ax['ISI_recovery_pred'].set_ylabel('Recovery (%)', fontsize=fontsize_label)
ax['ISI_recovery_pred'].set_ylim(y_lim_recovery)
ax['ISI_recovery_pred'].axhline(100, color='grey', linestyle='dotted')
ax['ISI_recovery_pred'].tick_params(axis='both', which='major', labelsize=fontsize_tick)
ax['ISI_recovery_pred'].set_xticks(add[1, :])
ax['ISI_recovery_pred'].set_xticklabels(label_tempCond, rotation=45)
ax['ISI_recovery_pred'].spines['top'].set_visible(False)
ax['ISI_recovery_pred'].spines['right'].set_visible(False)
if preference == 0:
    ax['ISI_recovery_pred'].set_ylim(20, 120)
elif preference == 1:
    ax['ISI_recovery_pred'].set_ylim(10, 120)
elif preference == 2:
    ax['ISI_recovery_pred'].set_ylim(20, 120)

ax['adaptation'].spines['top'].set_visible(False)
ax['adaptation'].spines['right'].set_visible(False)
ax['adaptation'].set_xticks(np.arange(len(VA))+0.1)
ax['adaptation'].set_xlim(y_lim_metrics)
ax['adaptation'].tick_params(axis='both', which='major', labelsize=fontsize_tick)
ax['adaptation'].set_xticklabels(VA_labels, fontsize=fontsize_label)
# ax['adaptation'].set_ylabel('Avg. recovery (%)', fontsize=fontsize_label)
if preference == 0:
    ax['adaptation'].set_ylim(30, 80)
elif preference == 1:
    ax['adaptation'].set_ylim(30, 80)
elif preference == 2:
    ax['adaptation'].set_ylim(30, 80)

ax['intercept'].spines['top'].set_visible(False)
ax['intercept'].spines['right'].set_visible(False)
ax['intercept'].set_xticks(np.arange(len(VA))+0.1)
ax['intercept'].set_xlim(y_lim_metrics)
ax['intercept'].tick_params(axis='both', which='major', labelsize=fontsize_tick)
ax['intercept'].set_xticklabels(VA_labels, fontsize=fontsize_label)
# ax['intercept'].set_ylabel('Long-term recovery (%)', fontsize=fontsize_label)
if preference == 0:
    ax['intercept'].set_ylim(55, 110)
elif preference == 1:
    ax['intercept'].set_ylim(45, 120)
elif preference == 2:
    ax['intercept'].set_ylim(55, 120)

# plot first pulse in isolation
max_data = [0, 0, 0]
max_model = [0, 0, 0]
for i in range(len(VA)):

    # NEURAL DATA
    data_pulse1 = np.mean(broadband_pulse1_bootstrap[i], axis=0)
    max_data[i] = max(data_pulse1)
    data_temp = gaussian_filter1d(data_pulse1[start_1 - start: start_1 - start + time_window]/max_data[i], 10)
    ax_broadband_isolation[i].plot(np.arange(time_window), data_temp, color='black', zorder=1)

    # MODEL
    data_pulse1 = np.mean(broadband_pulse1_pred_bootstrap[i], axis=0)
    max_model[i] = max(data_pulse1)
    model_temp = gaussian_filter1d(data_pulse1[start_1 - start: start_1 - start + time_window]/max_model[i], 10)
    ax_broadband_isolation_pred[i].plot(np.arange(time_window), model_temp, color='black', zorder=1)

# plot stimulus timecourse and time courses of neural data & model
t_zero          = np.argwhere(t > 0)[0][0]
t_twohundred    = np.argwhere(t > 0.5)[0][0]

x_label_single = ['0', '500']

xtick_idx = []
for i in range(len(tempCond)):

    # append x-tick
    xtick_idx = xtick_idx + ([i*(end+sep) + t_zero, i*(end+sep) + t_twohundred])

    # compute timepoint of the start of both first and second pulse
    start_2 = timepoints_twopulse[i, 2]

    for j in range(len(VA)):
    # for j in range(1):

        # plot stimulus timecourse
        if (j == 0) & (i == 0):
            ax['broadband'].axvspan(i*(end+sep) - start + timepoints_twopulse[i, 0], i*(
                end+sep) - start + timepoints_twopulse[i, 1], facecolor='grey', alpha=0.2, label='stimulus')
            ax['broadband'].axvspan(i*(end+sep) - start + timepoints_twopulse[i, 2], i*(
                end+sep) - start + timepoints_twopulse[i, 3], facecolor='grey', alpha=0.2)
            ax['broadband_pred'].axvspan(i*(end+sep) - start + timepoints_twopulse[i, 0], i*(
                end+sep) - start + timepoints_twopulse[i, 1], facecolor='grey', alpha=0.2, label='stimulus')
            ax['broadband_pred'].axvspan(i*(end+sep) - start + timepoints_twopulse[i, 2], i*(
                end+sep) - start + timepoints_twopulse[i, 3], facecolor='grey', alpha=0.2)
        elif (j == 0):
            ax['broadband'].axvspan(i*(end+sep) - start + timepoints_twopulse[i, 0], i*(
                end+sep) - start + timepoints_twopulse[i, 1], facecolor='grey', alpha=0.2)
            ax['broadband'].axvspan(i*(end+sep) - start + timepoints_twopulse[i, 2], i*(
                end+sep) - start + timepoints_twopulse[i, 3], facecolor='grey', alpha=0.2)
            ax['broadband_pred'].axvspan(i*(end+sep) - start + timepoints_twopulse[i, 0], i*(
                end+sep) - start + timepoints_twopulse[i, 1], facecolor='grey', alpha=0.2)
            ax['broadband_pred'].axvspan(i*(end+sep) - start + timepoints_twopulse[i, 2], i*(
                end+sep) - start + timepoints_twopulse[i, 3], facecolor='grey', alpha=0.2)

        # plot broadband timecourse per visual area
        if i == 0:

            # plot broadband per visual area
            data_temp = np.mean(broadband_bootstrap[j][:, i, :], axis=0)
            data_temp = gaussian_filter1d(data_temp[start:end]/max(data_temp[start:end]), 10)
            ax['broadband'].plot(np.arange(end - start)+i*(end+sep), data_temp, color=np.array(colors_VA[j])/255, label=VA_labels[j], lw=lw)

            # plot broadband per visual area
            model_temp = np.mean(broadband_pred_bootstrap[j][:, i, :], axis=0)
            model_temp = gaussian_filter1d(model_temp[start:end]/max(model_temp[start:end]), 10)
            ax['broadband_pred'].plot(np.arange(end - start)+i*(end+sep), model_temp, color=np.array(colors_VA[j])/255, label=VA_labels[j], lw=lw)

        else:

            # plot broadband per visual area
            data_temp = np.mean(broadband_bootstrap[j][:, i, :], axis=0)
            data_temp = gaussian_filter1d(data_temp[start:end]/max(data_temp[start:end]), 10)
            ax['broadband'].plot(np.arange(end - start)+i*(end+sep), data_temp, color=np.array(colors_VA[j])/255, lw=lw)

            # plot broadband per visual area
            model_temp = np.mean(broadband_pred_bootstrap[j][:, i, :], axis=0)
            model_temp = gaussian_filter1d(model_temp[start:end]/max(model_temp[start:end]), 10)
            ax['broadband_pred'].plot(np.arange(end - start)+i*(end+sep), model_temp, color=np.array(colors_VA[j])/255, lw=lw)

        # plot stimulus in isolation
        # NEURAL DATA
        data_pulse2 = gaussian_filter1d(np.mean(broadband_pulse2_bootstrap[j][:, i, :], axis=0)/max_data[j], 10)
        ax_broadband_isolation[j].plot(np.arange(time_window), data_pulse2[start_2 - start: start_2 - start + time_window], color=np.array(colors_VA[j])/255, alpha=alpha[i])

        # MODEL
        model_pulse2 = gaussian_filter1d(np.mean(broadband_pulse2_pred_bootstrap[j][:, i, :], axis=0)/max_model[j], 10)
        ax_broadband_isolation_pred[j].plot(np.arange(time_window), model_pulse2[start_2 - start: start_2 - start + time_window], color=np.array(colors_VA[j])/255, alpha=alpha[i])

        # plot mean data points
        # NEURAL DATA
        if i == 0:
            data_temp = ISI_recovery_avg[j, i]*100
            marker_temp = ax['ISI_recovery'].scatter(add[j, i], data_temp, color=np.array(colors_VA[j])/255, edgecolor='white', marker='o', s=150)
            marker.append(marker_temp)
        else:
            data_temp = ISI_recovery_avg[j, i]*100
            ax['ISI_recovery'].scatter(add[j, i], data_temp, color=np.array(colors_VA[j])/255, edgecolor='white', marker='o', s=150)

        error_min = ISI_recovery_CI[j, i, 0]*100
        error_max = ISI_recovery_CI[j, i, 1]*100
        ax['ISI_recovery'].plot([add[j, i], add[j, i]], [error_min, error_max], color='black', zorder=1)

        # MODEL
        if i == 0:
            data_temp = ISI_recovery_pred_avg[j, i]*100
            marker_temp = ax['ISI_recovery_pred'].scatter(add[j, i], data_temp, color=np.array(colors_VA[j])/255, edgecolor='white', marker='^', s=150)
            marker_pred.append(marker_temp)
        else:
            data_temp = ISI_recovery_pred_avg[j, i]*100
            ax['ISI_recovery_pred'].scatter(add[j, i], data_temp, color=np.array(colors_VA[j])/255, edgecolor='white', marker='^', s=150)

        error_min = ISI_recovery_pred_CI[j, i, 0]*100
        error_max = ISI_recovery_pred_CI[j, i, 1]*100
        ax['ISI_recovery_pred'].plot([add[j, i], add[j, i]], [error_min, error_max], color='black', zorder=1)

# plot recovery curve
for i in range(len(VA)):
# for i in range(1):

    line_temp, = ax['ISI_recovery'].plot(t1_plot/1000, ISI_recovery_log[i, :]*100, color=np.array(colors_VA[i])/255, zorder=-5, linestyle=linestyle[i])
    ax['ISI_recovery_pred'].plot(t1_plot/1000, ISI_recovery_pred_log[i, :]*100, color=np.array(colors_VA[i])/255, zorder=-5, linestyle=linestyle[i])
    line.append(line_temp)

# add ticks
ax['broadband'].set_xticks(xtick_idx)
ax['broadband'].set_xticklabels([])
ax['broadband'].legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left", borderaxespad=0, ncol=4, frameon=False, fontsize=fontsize_legend)

ax['broadband_pred'].set_xticks(xtick_idx)
ax['broadband_pred'].set_xticklabels(np.tile(x_label_single, 6))
ax['broadband_pred'].set_xlabel('Time (ms)', fontsize=fontsize_label)

# plot bargraph (slope log-linear fit)
error_min = adaptation_CI[:, 0]*100
error_max = adaptation_CI[:, 1]*100
ax['adaptation'].scatter(np.arange(len(VA)), adaptation_avg*100, color=np.array(colors_VA)/255, s=s)
ax['adaptation'].plot([np.arange(len(VA)), np.arange(len(VA))], [error_min, error_max], color='k', zorder=1)

error_min = adaptation_pred_CI[:, 0]*100
error_max = adaptation_pred_CI[:, 1]*100
ax['adaptation'].scatter(np.arange(len(VA))+0.2, adaptation_pred_avg*100, color=np.array(colors_VA)/255, s=s, marker='^')
ax['adaptation'].plot([np.arange(len(VA))+0.2, np.arange(len(VA))+0.2], [error_min, error_max], color='black', zorder=1)

# plot bargraph (slope log-linear fit)
error_min = intercept_CI[:, 0]*100
error_max = intercept_CI[:, 1]*100
ax['intercept'].scatter(np.arange(len(VA)), intercept_avg*100, color=np.array(colors_VA)/255, s=s)
ax['intercept'].plot([np.arange(len(VA)), np.arange(len(VA))], [error_min, error_max], color='k', zorder=1)

error_min = intercept_pred_CI[:, 0]*100
error_max = intercept_pred_CI[:, 1]*100
ax['intercept'].scatter(np.arange(len(VA))+0.2, intercept_pred_avg*100, color=np.array(colors_VA)/255, s=s, marker='^')
ax['intercept'].plot([np.arange(len(VA))+0.2, np.arange(len(VA))+0.2], [error_min, error_max], color='black', zorder=1)

# add legend to tempCond recovery plot
vertical_line1 = lines.Line2D([], [], color='black', marker='|', linestyle='None', markersize=30)
vertical_line2 = lines.Line2D([], [], color='black', marker='|', linestyle='None', markersize=30)    
vertical_line3 = lines.Line2D([], [], color='black', marker='|', linestyle='None', markersize=30)    
ax['ISI_recovery'].legend([(vertical_line1, marker[0], line[0]), (vertical_line2 , marker[1], line[1]), (vertical_line3 , marker[2], line[2])], [VA_labels[0], VA_labels[1], VA_labels[2]], loc='upper left', ncol=3, frameon=False, fontsize=fontsize_legend)
ax['ISI_recovery_pred'].legend([(vertical_line1, marker_pred[0], line[0]), (vertical_line2 , marker_pred[1], line[1]), (vertical_line3 , marker_pred[2], line[2])], [VA_labels[0], VA_labels[1], VA_labels[2]], loc='upper left', ncol=3, frameon=False, fontsize=fontsize_legend)

# save figure
plt.tight_layout()
plt.savefig(dir+'mkFigure/Fig5_6_' + img_type + '.svg', format='svg', bbox_inches='tight')
plt.savefig(dir+'mkFigure/Fig5_6_' + img_type, dpi=300, bbox_inches='tight')
# plt.show()

############################################################################
######################################################## STATISTICAL TESTING

alpha = 0.05
Bonferroni = 6

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
        if p < alpha:
            print('V1-3 vs. VOTC: ', p, ' SIGNIFICANT')
        else:
            print('V1-3 vs. VOTC: ', p)

        # early vs. LO
        sample1 = adaptation_medians[0, :]
        sample2 = adaptation_medians[2, :]
        param_diffs = sample1 - sample2

        p = np.min([len(param_diffs[param_diffs < 0]), len(param_diffs[param_diffs > 0])])/B_repetitions
        if p < alpha:
            print('V1-3 vs. LOTC: ', p, ' SIGNIFICANT')
        else:
            print('V1-3 vs. LOTC: ', p)

        # ventral vs. LO
        sample1 = adaptation_medians[1, :]
        sample2 = adaptation_medians[2, :]
        param_diffs = sample1 - sample2

        p = np.min([len(param_diffs[param_diffs < 0]), len(param_diffs[param_diffs > 0])])/B_repetitions
        if p < alpha:
            print('VOTC vs. LOTC: ', p, ' SIGNIFICANT')
        else:
            print('VOTC vs. LOTC: ', p)

        print('#'*30)
        print('MODEL')

        # early vs. ventral
        sample1 = adaptation_pred_medians[0, :]
        sample2 = adaptation_pred_medians[1, :]
        param_diffs = sample1 - sample2

        p = np.min([len(param_diffs[param_diffs < 0]), len(param_diffs[param_diffs > 0])])/B_repetitions
        if p < alpha:
            print('V1-3 vs. VOTC: ', p, ' SIGNIFICANT')
        else:
            print('V1-3 vs. VOTC: ', p)

        # early vs. LO
        sample1 = adaptation_pred_medians[0, :]
        sample2 = adaptation_pred_medians[2, :]
        param_diffs = sample1 - sample2

        p = np.min([len(param_diffs[param_diffs < 0]), len(param_diffs[param_diffs > 0])])/B_repetitions
        if p < alpha:
            print('V1-3 vs. LOTC: ', p, ' SIGNIFICANT')
        else:
            print('V1-3 vs. LOTC: ', p)

        # ventral vs. LO
        sample1 = adaptation_pred_medians[1, :]
        sample2 = adaptation_pred_medians[2, :]
        param_diffs = sample1 - sample2

        p = np.min([len(param_diffs[param_diffs < 0]), len(param_diffs[param_diffs > 0])])/B_repetitions
        if p < alpha:
            print('VOTC vs. LOTC: ', p, ' SIGNIFICANT')
        else:
            print('VOTC vs. LOTC: ', p)

    elif i == 1: # intercept`

        print('#'*30)
        print('NEURAL DATA')

        # early vs. ventral
        sample1 = intercept_medians[0, :]
        sample2 = intercept_medians[1, :]
        param_diffs = sample1 - sample2

        p = np.min([len(param_diffs[param_diffs < 0]), len(param_diffs[param_diffs > 0])])/B_repetitions
        if p < alpha:
            print('V1-3 vs. VOTC: ', p, ' SIGNIFICANT')
        else:
            print('V1-3 vs. VOTC: ', p)

        # early vs. LO
        sample1 = intercept_medians[0, :]
        sample2 = intercept_medians[2, :]
        param_diffs = sample1 - sample2

        p = np.min([len(param_diffs[param_diffs < 0]), len(param_diffs[param_diffs > 0])])/B_repetitions
        if p < alpha:
            print('V1-3 vs. LOTC: ', p, ' SIGNIFICANT')
        else:
            print('V1-3 vs. LOTC: ', p)

        # ventral vs. LO
        sample1 = intercept_medians[1, :]
        sample2 = intercept_medians[2, :]
        param_diffs = sample1 - sample2

        p = np.min([len(param_diffs[param_diffs < 0]), len(param_diffs[param_diffs > 0])])/B_repetitions
        if p < alpha:
            print('VOTC vs. LOTC: ', p, ' SIGNIFICANT')
        else:
            print('VOTC vs. LOTC: ', p)

        print('#'*30)
        print('MODEL')

        # early vs. ventral
        sample1 = intercept_pred_medians[0, :]
        sample2 = intercept_pred_medians[1, :]
        param_diffs = sample1 - sample2

        p = np.min([len(param_diffs[param_diffs < 0]), len(param_diffs[param_diffs > 0])])/B_repetitions
        if p < alpha:
            print('V1-3 vs. VOTC: ', p, ' SIGNIFICANT')
        else:
            print('V1-3 vs. VOTC: ', p)

        # early vs. LO
        sample1 = intercept_pred_medians[0, :]
        sample2 = intercept_pred_medians[2, :]
        param_diffs = sample1 - sample2

        p = np.min([len(param_diffs[param_diffs < 0]), len(param_diffs[param_diffs > 0])])/B_repetitions
        if p < alpha:
            print('V1-3 vs. LOTC: ', p, ' SIGNIFICANT')
        else:
            print('V1-3 vs. LOTC: ', p)

        # ventral vs. LO
        sample1 = intercept_pred_medians[1, :]
        sample2 = intercept_pred_medians[2, :]
        param_diffs = sample1 - sample2

        p = np.min([len(param_diffs[param_diffs < 0]), len(param_diffs[param_diffs > 0])])/B_repetitions
        if p < alpha:
            print('VOTC vs. LOTC: ', p, ' SIGNIFICANT')
        else:
            print('VOTC vs. LOTC: ', p)