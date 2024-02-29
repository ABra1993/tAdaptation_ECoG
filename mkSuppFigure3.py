# required packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import sys

# import functions and scripts
from utils import import_info, select_events, select_electrodes, import_epochs, recovery_perISI
from modelling_utils_fitObjective import OF_ISI_recovery_log

"""
Author: A. Brands

Description: 

"""

# define root directory
file = open('setDir.txt')
dir = file.readline().strip('\n')
print(dir)

# import timepoints of on- and offset of stimulus for one and twopulse trials
t                         = np.loadtxt(dir+'variables/t.txt', dtype=float)
timepoints_onepulse       = np.loadtxt(dir+'variables/timepoints_onepulse.txt', dtype=int)
timepoints_twopulse       = np.loadtxt(dir+'variables/timepoints_twopulse.txt', dtype=int)
ISI                       = np.loadtxt(dir+'variables/cond_temp.txt', dtype=float)
label_ISI                 = np.array(np.array(ISI, dtype=int), dtype=str)
time_window               = np.loadtxt(dir+'variables/time_window.txt', dtype=int)

# define model
cond                        = 4 # temporal condition (i.e. duration of the ISI)
trial_onepulse              = 'onepulse'
trials_twopulse             = ['onepulse-4', 'twopulse', 'twopulse_repeat']

# subject info
subject             = 'sub-p14'
electrode_name      = 'LO02'

# import info
_, events, channels, _ = import_info(subject, dir)

# select electrode(s)
electrode_idx = select_electrodes(channels, electrode_name)
if electrode_idx == -1:
    sys.exit('\n' + electrode_name + ' does not exist for ' +
                subject + '. Please chose another electrode ... \n')

# select twopulse events
event_idx = []
cond = None
for j in range(len(trials_twopulse)):
    if trials_twopulse[j] == 'onepulse-4':
        temp = select_events(events, None, trials_twopulse[j], dir)
        event_idx.append(temp)
    else:
        temp = select_events(events, 'TEMP', trials_twopulse[j], dir)
        event_idx.append(temp)

event_idx_onepulse = select_events(events, 'TEMP', 'onepulse', dir)

# extract broadband data
epochs_b = import_epochs(subject, electrode_idx, dir)

# plot ISI recovery
recovery_from_adaptation, estimate_first_pulse = recovery_perISI(t, epochs_b, event_idx, 2, time_window, dir)

# initiate (sub)plots
fig = plt.figure(figsize=(22, 3))
gs = fig.add_gridspec(5, 30)
ax = dict()

# settings fontsizes
fontsize_tick       = 10
fontsize_legend     = 10
fontsize_label      = 10

# plot specs/adjustments
end = 600
sep = 100

# initiate grids
ax['broadband'] = fig.add_subplot(gs[0:2, 0:19])
ax['broadband_sep'] = fig.add_subplot(gs[3:5, 0:19])
ax['ISI_recovery'] = fig.add_subplot(gs[0:5, 21:28])

# adjust axes
ax['broadband'].spines['top'].set_visible(False)
ax['broadband'].spines['right'].set_visible(False)
ax['broadband'].tick_params(axis='both', which='major', labelsize=fontsize_tick)
ax['broadband'].set_xlabel('Stimulus duration (ms)',                           fontsize=fontsize_label)

ax['broadband_sep'].spines['top'].set_visible(False)
ax['broadband_sep'].spines['right'].set_visible(False)
ax['broadband_sep'].tick_params(axis='both', which='major', labelsize=fontsize_tick)
ax['broadband_sep'].set_xlabel('ISI (ms)',                           fontsize=fontsize_label)

ax['ISI_recovery'].spines['top'].set_visible(False)
ax['ISI_recovery'].spines['right'].set_visible(False)
ax['ISI_recovery'].tick_params(axis='both', which='major', labelsize=fontsize_tick)
ax['ISI_recovery'].set_xlabel('ISI (ms)',                        fontsize=fontsize_label)
ax['ISI_recovery'].set_ylabel('Recovery (%)', fontsize=fontsize_label)

ymin = -1
ymax = 30
ax['broadband'].set_ylim(ymin, ymax)
ax['broadband_sep'].set_ylim(ymin, ymax)

ax['broadband'].axhline(0, color='grey', lw=0.5, alpha=0.5)
ax['broadband_sep'].axhline(0, color='grey', lw=0.5, alpha=0.5)   

# labels
label_AUC = [r'AUC$_{1}$', 'AUC$_{2}$']
label_points = [r'$\frac{AUC_{2}}{AUC_{1}}$']

broadband_color1 = ['dodgerblue']
broadband_fill1 = ['lightcyan'] 

broadband_color2 = ['darkorange']
broadband_fill2 = ['navajowhite']

start_1 = timepoints_twopulse[0, 0]
xtick_idx = []
alpha = 0.1
for i in range(len(ISI)):

    # append x-tick
    xtick_idx.append(i*(end+sep))

    # retrieve timepoint
    start_2 = timepoints_twopulse[i, 2]

    # retrieve data
    data_second_pulse = np.nanmean(epochs_b[event_idx[2][i]], axis=1)
    data_second_pulse_sep = np.nanmean(epochs_b[event_idx[2][i]], axis=1) - estimate_first_pulse

    # plot stimulus
    ax['broadband'].axvspan(i*(end+sep) - start_1 + timepoints_twopulse[i, 0], 
                            i*(end+sep) - start_1 + timepoints_twopulse[i, 1], facecolor='grey', alpha=0.2)
    ax['broadband'].axvspan(i*(end+sep) - start_1 + timepoints_twopulse[i, 2], 
                            i*(end+sep) - start_1 + timepoints_twopulse[i, 3], facecolor='grey', alpha=0.2)

    ax['broadband_sep'].axvspan(i*(end+sep) - start_1 + timepoints_twopulse[i, 0], 
                            i*(end+sep) - start_1 + timepoints_twopulse[i, 1], facecolor='grey', alpha=0.2)
    ax['broadband_sep'].axvspan(i*(end+sep) - start_1 + timepoints_twopulse[i, 2], 
                            i*(end+sep) - start_1 + timepoints_twopulse[i, 3], facecolor='grey', alpha=0.2)


    ax['broadband_sep'].axvspan(i*(end+sep) - start_1 + timepoints_onepulse[i, 0], 
                            i*(end+sep) - start_1 + timepoints_onepulse[i, 1], facecolor='grey', alpha=0.2)
    ax['broadband_sep'].axvspan(i*(end+sep) - start_1 + timepoints_twopulse[i, 2], 
                            i*(end+sep) - start_1 + timepoints_twopulse[i, 3], facecolor='grey', alpha=0.2)


    one_pulse_mean = np.nanmean(epochs_b[event_idx_onepulse[i]], axis=1)

    # plot broadband (first and second pulse)
    if i == 0:

        # plot broadband
        ax['broadband'].plot(np.arange(end)+i*(end+sep),
                                    data_second_pulse[start_1:start_1+end], color='black', label='Neural response')

        # plot broadband (first and second pulse seperately)
        ax['broadband_sep'].plot(np.arange(time_window)+i*(end+sep), estimate_first_pulse[start_1:start_1 +
                                    time_window], color=broadband_color1[0], label='First response', zorder=-10)
        ax['broadband_sep'].plot(np.arange(time_window)+i*(end+sep)+start_2-start_1,
                                    data_second_pulse_sep[start_2:start_2+time_window], color=broadband_color2[0], label='Second response')

        ax['broadband_sep'].fill_between(np.arange(time_window)+i*(end+sep), 0, 
                                        estimate_first_pulse[start_1:start_1+time_window], color=broadband_fill1[0], edgecolor=broadband_color1[0], hatch='|||||', label=label_AUC[0])
        ax['broadband_sep'].fill_between(np.arange(time_window)+i*(end+sep)+start_2-start_1, 0,
                                        data_second_pulse_sep[start_2:start_2+time_window], color=broadband_fill2[0], edgecolor=broadband_color2[0], hatch='|||||', label=label_AUC[1])

    else:

        # plot broadband
        ax['broadband'].plot(np.arange(end)+i*(end+sep),
                                    data_second_pulse[start_1:start_1+end], color='black')

        # plot broadband (first and second pulse seperately)
        ax['broadband_sep'].plot(np.arange(time_window)+i*(end+sep), estimate_first_pulse[start_1:start_1 +
                                    time_window], color=broadband_color1[0], zorder=-10)
        ax['broadband_sep'].plot(np.arange(time_window)+i*(end+sep)+start_2-start_1,
                                    data_second_pulse_sep[start_2:start_2+time_window], color=broadband_color2[0])

        ax['broadband_sep'].fill_between(np.arange(time_window)+i*(end+sep), 0, 
                                        estimate_first_pulse[start_1:start_1+time_window], color=broadband_fill1[0], edgecolor=broadband_color1, hatch='|||||')
        ax['broadband_sep'].fill_between(np.arange(time_window)+i*(end+sep)+start_2-start_1, 0,
                                        data_second_pulse_sep[start_2:start_2+time_window], color=broadband_fill2[0], edgecolor=broadband_color2, hatch='|||||')
        
# plot legends
ax['broadband'].legend(ncol=2, fontsize=fontsize_legend, frameon=False, loc='upper left') # bbox_to_anchor=(0.95, 0.05))
ax['broadband'].legend(fontsize=fontsize_legend, frameon=False, loc='upper left') # bbox_to_anchor=(0.95, 0.05))
ax['broadband_sep'].legend(fontsize=fontsize_legend, frameon=False, ncol=4, loc='upper left') # bbox_to_anchor=(0.95, 0.05))

# plot ISI recovery from RS
ax['ISI_recovery'].axhline(
    1, color='grey', linestyle='--', alpha=0.3, zorder=-1)  # 1 (wrt first pulse)

# plot ISI recovery (points and line fit)
ax['ISI_recovery'].scatter(
    ISI, recovery_from_adaptation, color='k', label=label_points[0], s=30)

# fit function to data
p0 = [1, 0]
popt, pcov = curve_fit(OF_ISI_recovery_log, ISI, recovery_from_adaptation,
                        p0, maxfev=1000, bounds=((0, 0), (np.inf, np.inf)))
t_temp = np.linspace(min(ISI), max(ISI), 100)

y = OF_ISI_recovery_log(t_temp, *popt)
ax['ISI_recovery'].plot(t_temp, y, color='lightgrey',
                        lw=1.5, zorder=-10)

t_temp = np.linspace(min(ISI), 4, 1000)
y = OF_ISI_recovery_log(t_temp, *popt)

# set x-ticks of ISI
ax['broadband'].set_xticks(xtick_idx)
ax['broadband_sep'].set_xticks(xtick_idx)
ax['ISI_recovery'].set_xticks(ISI)

ax['broadband'].set_xticklabels(label_ISI, rotation=45, fontsize=fontsize_label)
ax['broadband_sep'].set_xticklabels(label_ISI, rotation=45, fontsize=fontsize_label)
ax['ISI_recovery'].set_xticklabels(label_ISI, rotation=45, fontsize=fontsize_label)

# save figure
plt.tight_layout()
plt.savefig(dir+'mkFigure/SuppFig2.svg', format='svg')
plt.savefig(dir+'mkFigure/SuppFig2', dpi=300) 
plt.show()
