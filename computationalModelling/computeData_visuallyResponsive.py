# required packages
from random import sample
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

from utils import select_events

"""
Author: A. Brands
 
Description: creates a pandas DataFrame holidng the samples (72 time courses) that are used to fit the computational models (two trials x 6 temp. conditions x 6 image categories)

"""

# define root directory
dir = '/home/amber/OneDrive/code/git_nAdaptation_ECoG/'

# import variables
img_cat     = np.loadtxt(dir+'variables/cond_stim.txt', dtype=str)
temp_cond   = np.loadtxt(dir+'variables/cond_temp.txt', dtype=int)
t           = np.loadtxt(dir+'variables/t.txt', dtype=float)
t_index     = np.arange(len(t)).astype(str).tolist()

# assign trials
trial_type = ['onepulse', 'twopulse_repeat']

# import electrodes
responsive_electrodes = pd.read_csv(dir+'subject_data/electrodes_visuallyResponsive_manuallyAssigned.txt', header=0, index_col=0, delimiter=' ')
n_electrodes = len(responsive_electrodes)

# create dataframe
df = pd.DataFrame(columns=['trial_type', 'img_cat', 'temp_cond'] + t_index)
df['trial_type'] = np.repeat(trial_type, len(img_cat)*len(temp_cond))
df['img_cat'] = np.tile(np.repeat(img_cat, len(temp_cond)), 2)
df['temp_cond'] = np.tile(temp_cond, len(trial_type)*len(img_cat))
# print(df)

# retrieve info
current_subject = ''
for i in range(n_electrodes):
# for i in range(1):

    # defines electrode and subject
    subject                     = responsive_electrodes.loc[i, 'subject']
    electrode_name              = responsive_electrodes.loc[i, 'electrode']
    electrode_idx               = int(responsive_electrodes.loc[i, 'electrode_idx'])

    # print progress
    print('Computing trials for ' + subject + ', electrode ' + electrode_name + ' (' + str(i+1) + '/' + str(n_electrodes) + ')')

    # ----------------------------------------------------------- 
    try: # create electrode directory to store data
        os.mkdir(dir+'computationalModelling/modelFit/visuallyResponsive/' + subject + '_' + electrode_name)
    except:
        print('Folder for electrode already exist...')
    # -----------------------------------------------------------

    if subject != current_subject:

        # update info
        current_subject = subject

        # import info
        events = pd.read_csv(dir+'subject_data/' + subject + '/events.txt', header=0)

        # import excluded trials
        excluded_epochs = pd.read_csv(dir+'subject_data/' + subject + '/excluded_epochs.txt', sep=' ', index_col=0, header=0, dtype=int)

    # import broadband timecourses
    epochs_b = pd.read_csv(dir+'subject_data/' + subject + '/epochs_b/epochs_b_channel' + str(electrode_idx+1) + '_baselineCorrection.txt', sep=' ', header=None)
    index_epochs = [j for j in range(len(events)) if excluded_epochs.iloc[i, j] == 1]
    epochs_b.iloc[:, index_epochs] = np.nan

    # copy dataframe
    df_current_electrode = df.copy()

    # plot trials
    count = 0 # count trial
    for j in range(len(trial_type)):
    # for i in range(1):

            # select events
            event_idx = select_events(events, 'both', trial_type[j], dir)

            # iterate and extract data
            for l in range(len(img_cat)):
                for k in range(len(temp_cond)):

                    # extract data
                    data = np.nanmean(epochs_b.iloc[:, event_idx[l][k]], axis=1)
                    df_current_electrode.loc[count, t_index] = data

                    # increment count
                    count = count + 1

    # save dataframe
    df_current_electrode.to_csv(dir+'computationalModelling/modelFit/visuallyResponsive/' + subject + '_' + electrode_name + '/data.txt', sep=' ', index=False)