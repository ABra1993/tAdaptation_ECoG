# import required packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


"""
Author: A. Brands

Description: excludes epochs based on maximal response

"""

# define root directory
dir = '/home/amber/OneDrive/code/git_nAdaptation_ECoG/'

# threshold for epoch selection (n x standard deviation)
n_std = 2

# user-defined values (ONLY CHANGE CODE HERE)
subjects = ['sub-p11', 'sub-p12', 'sub-p13', 'sub-p14']

# track exclusion percentages
excl_perc_all = np.zeros(len(subjects))

# iterate over subjects and exclude trials
for idx, subject in enumerate(subjects):

    # print progress
    print('Data selection for ', subject, '...')
    
    # import data
    events              = pd.read_csv(dir+'subject_data/' + subject + '/events.txt', header=0)
    channels            = pd.read_csv(dir+'subject_data/' + subject + '/channels.txt', header=0)
    electrodes          = pd.read_csv(dir+'subject_data/' + subject + '/electrodes.txt', header=0)

    # select epochs based on broadband or voltage data
    n_epochs = len(events)
    max_values = np.zeros(n_epochs)
    excluded_epochs_all = np.zeros((len(channels), n_epochs))

    for i in range(len(channels)):

        # electrode_name
        electrode_name = electrodes.name[i]

        # import broadband timecourses
        epochs_b = pd.read_csv(dir+'subject_data/' + subject + '/epochs_b/epochs_b_channel' + str(i+1) + '.txt', sep=',', header=None).to_numpy()

        # maximal value
        max_values = np.amax(epochs_b, axis=0)

        # compute standard deviation
        mean = np.mean(max_values)
        std = np.std(max_values)

        # iterate over epochs and exclude large values (> n*std)
        excluded_epochs = [idx for idx in range(n_epochs) if (max_values[idx] > (mean + n_std*std))]
        excluded_epochs_all[i, excluded_epochs] = 1

    # print exclusion
    excluded_epochs_flatten = excluded_epochs_all.reshape(len(channels)*n_epochs)
    excl_perc_all[idx] = sum(excluded_epochs_flatten)/len(excluded_epochs_flatten)
    print('Exclusion ', subject, ':', excl_perc_all[idx]*100, '%')

    # save excluded epochs
    pd_save = pd.DataFrame(excluded_epochs_all)
    pd_save.to_csv(dir+'subject_data/' + subject + '/excluded_epochs.txt', sep=' ', index=True)

# print average exclusion percentage over all subjects
print('Average exclusion over all subjects: ', np.mean(excl_perc_all)*100)

