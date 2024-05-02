# import required packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os


"""
Author: A. Brands

Description: Applies a normalization to the epoched time courses per run based on a specified time interval within the epoch.

"""

# define root directory
file = open('setDir.txt')
dir = file.readline()

# user-defined values (ONLY CHANGE CODE HERE)
subjects = ['sub-p11', 'sub-p12', 'sub-p13', 'sub-p14']
# subjects = ['sub-p11']

# determine timepoint of stimulus onset
timepoints_onepulse = np.loadtxt(dir+'variables/timepoints_onepulse.txt', dtype=int)
start_stimulus = timepoints_onepulse[0, 0]

for subject in subjects:
    
    # import data and remove excluded epochs
    t                       = pd.read_csv(dir+'data_subjects/' + subject + '/t.txt', header=None)
    events                  = pd.read_csv(dir+'data_subjects/' + subject + '/events.txt', header=0)
    channels                = pd.read_csv(dir+'data_subjects/' + subject + '/channels.txt', header=0)
    electrodes              = pd.read_csv(dir+'data_subjects/' + subject + '/electrodes.txt', header=0)

    # import excluded epochs
    excluded_epochs = pd.read_csv(dir+'data_subjects/' + subject + '/excluded_epochs.txt', sep=' ', header=0, index_col=0, dtype=int)

    # determine number of runs
    runs = events.stim_file.unique()
    n_runs = len(runs)

    # retrieve data per run and perform baseline correction
    for i in range(len(channels)):
    # for i in range(69, 70):

        # electrode_name
        electrode_name = electrodes.name[i]

        # print progress
        print('Computing response of electrode ' + electrode_name + '... (' + str(i+1) + '/' + str(len(channels))+ ')')

        # import broadband timecourses
        # fig, axs = plt.subplots(1, 2)
        epochs_b = pd.read_csv(dir+'data_subjects/' + subject + '/epochs_b/epochs_b_channel' + str(i+1) + '.txt', sep=',', header=None)
        # axs[0].plot(epochs_b, label='no excl. epochs')
        index_epochs = [j for j in range(len(events)) if excluded_epochs.iloc[i, j] == 1]
        epochs_b.iloc[:, index_epochs] = np.nan
        # axs[1].plot(epochs_b, label='excl. epochs')
        # plt.show()
        # plt.close()

        # fig, axs = plt.subplots(1, 2)
        # axs[0].plot(epochs_b.mean(1))

        # plot epochs
        for j in range(n_runs):
        # for j in range(1):
        
            # select current run
            events_idx_current = events[events.stim_file == runs[j]].index

            # compute average pre-stimulus baseline activity
            baseline_activity_avg = epochs_b.iloc[0:start_stimulus, events_idx_current].mean(axis=0).mean()

            # perform baseline correction
            epochs_b.iloc[:, events_idx_current] = (epochs_b.iloc[:, events_idx_current] - baseline_activity_avg)/abs(baseline_activity_avg)
        
        # axs[1].plot(epochs_b.mean(1))
        # plt.show()
        # plt.close()
        
        # save baseline-corrected timecourse
        np.savetxt(dir+'data_subjects/' + subject + '/epochs_b/epochs_b_channel' + str(i+1) + '_baselineCorrection.txt', epochs_b)
