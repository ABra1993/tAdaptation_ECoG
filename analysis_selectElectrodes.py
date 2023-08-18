# required packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils import stimulus_onset, select_events, d_prime_perImgCat

"""
Author: A. Brands

Description: select visually responsive electrodes and assigns to early or higher (VOTC or LOTC)
visual areas based on anatomc (Benson et al., 2014) or probabilistic atlas (Wang et al., 2015).

"""

# define root directory
file = open('setDir.txt')
dir = file.readline()

##### SPECIFY ELECTRODE TYPE
# electrode_type = 'visualllyResponsive'
electrode_type = 'categorySelective'

# SUBJECTS
subjects = ['sub-p11', 'sub-p12', 'sub-p13', 'sub-p14']

# select responsive electrodes
if electrode_type == 'visualllyResponsive':

    # user-defined values (ONLY CHANGE CODE HERE)
    threshold_CV = 0.2

    # predefined variables
    timepoints_onepulse = np.loadtxt(dir+'variables/timepoints_onepulse.txt', dtype=int)
    start = timepoints_onepulse[0, 0]   # start trial
    end = timepoints_onepulse[5, 1]     # end pulse longest duration

    # dataframe columns
    column_names = ['subject', 'electrode_idx', 'electrode', 'wang15_mplbl', 'benson14_varea', 'varea', 'x', 'y', 'z']
    df = pd.DataFrame(columns=column_names)

    # merge responsive electrodes
    count = 0
    for subject in subjects:

        # import data and remove excluded epochs
        t                       = pd.read_csv(dir+'subject_data/' + subject + '/t.txt', header=None)
        events                  = pd.read_csv(dir+'subject_data/' + subject + '/events.txt', header=0)
        channels                = pd.read_csv(dir+'subject_data/' + subject + '/channels.txt', header=0)
        electrodes              = pd.read_csv(dir+'subject_data/' + subject + '/electrodes.txt', header=0)

        # import coordintes
        column_names = ['x', 'y', 'z', 'type', 'nan']
        coordinates  = pd.read_csv(dir+'subject_data/' + subject + '/coordinates.txt', delimiter=' ', names=['electrode', 'x', 'y', 'z', 'type'], index_col=False)
        
        # determine number of channels
        n_electrodes = len(channels)

        # import excluded epochs
        excluded_epochs = pd.read_csv(dir+'subject_data/' + subject + '/excluded_epochs.txt', sep=' ', index_col=0, header=0, dtype=int)

        # select events
        event_idx = select_events(events=events, cond=None, trial='onepulse', dir=dir)

        # iterate over electrodes and check signal
        for i in range(n_electrodes):
        # for i in range(66, 67):

            # electrode_name
            electrode_name = electrodes.name[i]

            # import broadband timecourses
            epochs_b = pd.read_csv(dir+'subject_data/' + subject + '/epochs_b/epochs_b_channel' + str(i+1) + '_baselineCorrection.txt', sep=' ', header=None)
            index_epochs = [j for j in range(len(events)) if excluded_epochs.iloc[i, j] == 1]
            epochs_b.iloc[:, index_epochs] = np.nan

            # print progress
            print('Computing response of electrode ' + subject + ', ' + electrode_name + '... (' + str(i+1) + '/' + str(len(channels))+ ')')

            # compute mean signal for one and two pulses
            epochs_mean_onepulse = epochs_b.iloc[:, event_idx].mean(axis=1)

            # INCLUSION CRITERIUM 1: Response-Onset Latency
            _, onset_timepts_onepulse, isOnset = stimulus_onset(t, epochs_mean_onepulse, dir)

            # INCLUSION CRITERIUM 2: Coefficient of variation (CV)
            std_onepulse = epochs_b.iloc[:, event_idx].std(axis=1)
            mean_onepulse = epochs_b.iloc[:, event_idx].mean(axis=1)
            cv_one_pulse = list(mean_onepulse/std_onepulse)
            cv_one_pulse = np.mean(cv_one_pulse[start:end])

            if isOnset & (cv_one_pulse > threshold_CV):

                # compute coefficient of correlation
                std_avg = np.mean(std_onepulse[start+onset_timepts_onepulse:end])
                mean_avg = np.mean(mean_onepulse[start+onset_timepts_onepulse:end])
                cv_one_pulse = std_avg/mean_avg

                print('Electrode selected :)')

                # add info to dataframe
                df.loc[count, ['subject', 'electrode_idx', 'electrode']] = [subject, i, electrode_name]
                
                # retrieve visual areas (Wang and Benson)
                df.loc[count, 'wang15_mplbl'] = electrodes.loc[i, 'wang15_mplbl']
                df.loc[count, 'benson14_varea'] = electrodes.loc[i, 'benson14_varea']

                if (electrodes.loc[i, 'wang15_mplbl'] == 'V1v') | \
                    (electrodes.loc[i, 'wang15_mplbl'] == 'V1d') | \
                    (electrodes.loc[i, 'wang15_mplbl'] == 'V2v') | \
                    (electrodes.loc[i, 'wang15_mplbl'] == 'V2d') | \
                    (electrodes.loc[i, 'wang15_mplbl'] == 'V3v') | \
                    (electrodes.loc[i, 'wang15_mplbl'] == 'V3d') | \
                    (electrodes.loc[i, 'benson14_varea'] == 'V1') | \
                    (electrodes.loc[i, 'benson14_varea'] == 'V2') | \
                    (electrodes.loc[i, 'benson14_varea'] == 'V3'):

                    df.loc[count, 'varea'] = 'V1-V3'

                elif (electrodes.loc[i, 'wang15_mplbl'] == 'hV4') | \
                    (electrodes.loc[i, 'wang15_mplbl'] == 'VO1') | \
                    (electrodes.loc[i, 'wang15_mplbl'] == 'VO2') | \
                    (electrodes.loc[i, 'benson14_varea'] == 'hV4') | \
                    (electrodes.loc[i, 'benson14_varea'] == 'VO1') | \
                    (electrodes.loc[i, 'benson14_varea'] == 'VO2'):

                    df.loc[count, 'varea'] = 'VOTC'
                    
                elif (electrodes.loc[i, 'wang15_mplbl'] == 'V3a') | \
                    (electrodes.loc[i, 'wang15_mplbl'] == 'V3b') | \
                    (electrodes.loc[i, 'wang15_mplbl'] == 'LO1') | \
                    (electrodes.loc[i, 'wang15_mplbl'] == 'LO2') | \
                    (electrodes.loc[i, 'wang15_mplbl'] == 'TO1') | \
                    (electrodes.loc[i, 'wang15_mplbl'] == 'TO2') | \
                    (electrodes.loc[i, 'benson14_varea'] == 'V3a') | \
                    (electrodes.loc[i, 'benson14_varea'] == 'V3b') | \
                    (electrodes.loc[i, 'benson14_varea'] == 'LO1') | \
                    (electrodes.loc[i, 'benson14_varea'] == 'LO2') | \
                    (electrodes.loc[i, 'benson14_varea'] == 'TO1') | \
                    (electrodes.loc[i, 'benson14_varea'] == 'TO2'):

                    df.loc[count, 'varea'] = 'LOTC'

                else:

                    df.loc[count, 'varea'] = 'none'

                # add coordinates

                df.loc[count, ['x', 'y', 'z']] = coordinates.loc[i, ['x', 'y', 'z']]

                # increment count
                count+=1

    # save dataframe
    df.to_csv(dir+'subject_data/electrodes_visuallyResponsive.txt', index=True, sep=' ')

    # print count
    print(count, ' electrodes selected!')

elif electrode_type == 'categorySelective':

    # thresholds
    threshold_d_prime = 1.0
    threshold_cv = 0.2

    # predefined variables
    timepoints_onepulse = np.loadtxt(dir+'variables/timepoints_onepulse.txt', dtype=int)
    start = timepoints_onepulse[0, 0]   # start trial
    end = timepoints_onepulse[5, 1]     # end pulse longest duration

    # import image categories
    stim = np.loadtxt(dir+'variables/cond_stim.txt', dtype=str)

    # dataframe columns
    column_names = ['subject', 'electrode_idx', 'electrode', 'wang15_mplbl', 'benson14_varea', 'varea', 'x', 'y', 'z']
    df = pd.DataFrame(columns=column_names)

    # merge responsive electrodes
    count = 0
    for subject in subjects:

        # import data and remove excluded epochs
        t                       = pd.read_csv(dir+'subject_data/' + subject + '/t.txt', header=None)
        events                  = pd.read_csv(dir+'subject_data/' + subject + '/events.txt', header=0)
        channels                = pd.read_csv(dir+'subject_data/' + subject + '/channels.txt', header=0)
        electrodes              = pd.read_csv(dir+'subject_data/' + subject + '/electrodes.txt', header=0)

        # import coordintes
        column_names = ['x', 'y', 'z', 'type', 'nan']
        coordinates  = pd.read_csv(dir+'subject_data/' + subject + '/coordinates.txt', delimiter=' ', names=['electrode', 'x', 'y', 'z', 'type'], index_col=False)
        
        # determine number of channels
        n_electrodes = len(channels)

        # import excluded epochs
        excluded_epochs = pd.read_csv(dir+'subject_data/' + subject + '/excluded_epochs.txt', sep=' ', index_col=0, header=0, dtype=int)

        # select events
        event_idx = select_events(events=events, cond='STIM', trial='onepulse', dir=dir)

        # iterate over electrodes and check signal
        for i in range(n_electrodes):
        # for i in range(1):

            # electrode_name
            electrode_name = electrodes.name[i]

            # import broadband timecourses
            epochs_b = pd.read_csv(dir+'subject_data/' + subject + '/epochs_b/epochs_b_channel' + str(i+1) + '_baselineCorrection.txt', sep=' ', header=None)
            index_epochs = [j for j in range(len(events)) if excluded_epochs.iloc[i, j] == 1]
            epochs_b.iloc[:, index_epochs] = np.nan

            # print progress
            print('Computing response of electrode ' + subject + ', ' + electrode_name + '... (' + str(i+1) + '/' + str(len(channels))+ ')')

            # INCLUSION CRITERION 1: d' indicating category-selectivity
            d_prime = d_prime_perImgCat(epochs_b, event_idx, stim)
            preferred_cat_idx = np.argmax(d_prime)
            nonpreferred_cat_idx = np.argmin(d_prime)

            # INCLSUION CRRITERION 2: Stimulus-onset Latency
            epochs_onepulse = epochs_b.iloc[:, event_idx[nonpreferred_cat_idx]]
            epochs_mean_onepulse = epochs_b.iloc[:, event_idx[nonpreferred_cat_idx]].mean(axis=1)
            onset_onepulse, onset_timepts_onepulse, onset = stimulus_onset(t, epochs_mean_onepulse, dir)

            # INCLUSION CRITERIUM 3: Coefficient of variation (CV)
            std = epochs_onepulse.loc[start+onset_timepts_onepulse:end, :].std(axis=1)
            mean = epochs_onepulse.loc[start+onset_timepts_onepulse:end, :].mean(axis=1)
            cv = np.mean(mean/std)

            if (d_prime[preferred_cat_idx] > threshold_d_prime) & (onset == True) & (cv > threshold_cv):

                if (electrodes.loc[i, 'wang15_mplbl'] == 'V3a') | \
                    (electrodes.loc[i, 'wang15_mplbl'] == 'V3b') | \
                    (electrodes.loc[i, 'wang15_mplbl'] == 'LO1') | \
                    (electrodes.loc[i, 'wang15_mplbl'] == 'LO2') | \
                    (electrodes.loc[i, 'wang15_mplbl'] == 'TO1') | \
                    (electrodes.loc[i, 'wang15_mplbl'] == 'TO2') | \
                    (electrodes.loc[i, 'benson14_varea'] == 'V3a') | \
                    (electrodes.loc[i, 'benson14_varea'] == 'V3b') | \
                    (electrodes.loc[i, 'benson14_varea'] == 'LO1') | \
                    (electrodes.loc[i, 'benson14_varea'] == 'LO2') | \
                    (electrodes.loc[i, 'benson14_varea'] == 'TO1') | \
                    (electrodes.loc[i, 'benson14_varea'] == 'TO2')| \
                    (electrodes.loc[i, 'wang15_mplbl'] == 'hV4') | \
                    (electrodes.loc[i, 'wang15_mplbl'] == 'VO1') | \
                    (electrodes.loc[i, 'wang15_mplbl'] == 'VO2') | \
                    (electrodes.loc[i, 'benson14_varea'] == 'hV4') | \
                    (electrodes.loc[i, 'benson14_varea'] == 'VO1') | \
                    (electrodes.loc[i, 'benson14_varea'] == 'VO2') | \
                    (electrodes.loc[i, 'wang15_mplbl'] == 'V1v') | \
                    (electrodes.loc[i, 'wang15_mplbl'] == 'V1d') | \
                    (electrodes.loc[i, 'wang15_mplbl'] == 'V2v') | \
                    (electrodes.loc[i, 'wang15_mplbl'] == 'V2d') | \
                    (electrodes.loc[i, 'wang15_mplbl'] == 'V3v') | \
                    (electrodes.loc[i, 'wang15_mplbl'] == 'V3d') | \
                    (electrodes.loc[i, 'benson14_varea'] == 'V1') | \
                    (electrodes.loc[i, 'benson14_varea'] == 'V2') | \
                    (electrodes.loc[i, 'benson14_varea'] == 'V3'):

                    print('Electrode selected :)')

                    # add info to dataframe
                    df.loc[count, ['subject', 'electrode_idx', 'electrode', 'preferred_cat']] = [subject, i, electrode_name, stim[preferred_cat_idx]]
                    
                    # retrieve visual areas (Wang and Benson)
                    df.loc[count, 'wang15_mplbl'] = electrodes.loc[i, 'wang15_mplbl']
                    df.loc[count, 'benson14_varea'] = electrodes.loc[i, 'benson14_varea']

                    # add coordinates
                    df.loc[count, ['x', 'y', 'z']] = coordinates.loc[i, ['x', 'y', 'z']]

                    # increment count
                    count+=1

    # save dataframe
    df.to_csv(dir+'subject_data/electrodes_categorySelective_' + str(threshold_d_prime).replace('.', '-') + '.txt', index=True, sep=' ')

    # print count
    print(count, ' electrodes selected!')


