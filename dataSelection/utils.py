import numpy as np
import scipy.stats as st
from scipy.signal import savgol_filter
import math
import pandas as pd

def select_events(events, cond, trial, dir):
    """ Returns list of the visual areas measured

    params
    -----------------------
    events : array dim(n , m)
        contains the events (n) with additional info (m)
    cond : string ('TEMP', 'STIM')
        indicates the type of conditions (either temporal - TEMP - or img. cat. - STIM)
    trial : string
        indicates type of trial (e.g. 'onepulse')
    dir : string
        indicates root directory (to save figure)

    returns
    -----------------------
    event_idx : array dim(1, n)
        contains the index numbers for the events belonging to one experimental condition

    """

    if cond == None:
        if trial == 'all':              # select ALL epochs

            return np.arange(len(events))

        elif trial == 'onepulse-4':     # used to compute AUC of second pulse

            return events[(events.trial_name.str.contains('ONEPULSE-4'))].index

        else:                           # select epochs for a specific trial type

            event_idx = []
            if trial == 'onepulse':
                event_idx = events[(events.stim_file.str.contains('sixcatloctemporal')) & (events.trial_name.str.contains('ONE'))].index
            elif trial == 'twopulse':
                event_idx = events[(events.trial_name.str.contains('TWO'))].index
            elif trial == 'twopulse_repeat':
                event_idx = events[(events.stim_file.str.contains('sixcatloctemporal')) & (events.trial_name.str.contains('TWO'))].index
            elif trial == 'twopulse_nonrepeat':
                event_idx = events[((events.stim_file.str.contains('sixcatlocisidiff')) | (events.stim_file.str.contains('sixcatlocdiffisi')))].index
            elif trial == 'twopulse_nonrepeat_same':
                event_idx = events[((events.stim_file.str.contains('sixcatlocisidiff')) | (events.stim_file.str.contains('sixcatlocdiffisi'))) & (events.trial_name.str.contains('SAME'))].index
            elif trial == 'twopulse_nonrepeat_diff':
                event_idx = events[((events.stim_file.str.contains('sixcatlocisidiff')) | (events.stim_file.str.contains('sixcatlocdiffisi'))) & (events.trial_name.str.contains('DIFF'))].index

            return event_idx

    else:

        # define conditions within axis
        axis_cond = []
        if cond == 'TEMP':
            axis_cond = ['1', '2', '3', '4', '5', '6']
        elif cond == 'STIM':
            axis_cond = np.loadtxt(dir+'variables/cond_stim.txt', dtype=str)

        # obtain event indices
        event_idx = []
        for i in range(len(axis_cond)):
            if trial == 'all':
                temp = events[(events.trial_name.str.contains(axis_cond[i]))].index
                event_idx.append(temp)
            elif trial == 'onepulse':
                temp = events[(events.stim_file.str.contains('sixcatloctemporal')) & \
                                (events.trial_name.str.contains('ONE')) & \
                                    (events.trial_name.str.contains(axis_cond[i]))].index
                event_idx.append(temp)
            elif trial == 'twopulse':
                temp = events[(events.trial_name.str.contains('TWO')) & \
                                (events.trial_name.str.contains(axis_cond[i]))].index
                event_idx.append(temp)
            elif trial == 'twopulse_repeat': # twopulse_repeat
                temp = events[(events.stim_file.str.contains('sixcatloctemporal')) & \
                                (events.trial_name.str.contains('TWO')) & \
                                    (events.trial_name.str.contains(axis_cond[i]))].index
                event_idx.append(temp)
            elif trial == 'twopulse_nonrepeat':
                temp = events[((events.stim_file.str.contains('sixcatlocisidiff')) | (events.stim_file.str.contains('sixcatlocdiffisi'))) & \
                                (events.trial_name.str.contains(axis_cond[i]))].index
                event_idx.append(temp)
            elif trial == 'twopulse_nonrepeat_same':
                temp = events[((events.stim_file.str.contains('sixcatlocisidiff')) | (events.stim_file.str.contains('sixcatlocdiffisi'))) & \
                                (events.trial_name.str.contains('SAME')) & \
                                    (events.trial_name.str.contains(axis_cond[i]))].index
                event_idx.append(temp)
            elif trial == 'twopulse_nonrepeat_diff':
                temp = events[((events.stim_file.str.contains('sixcatlocisidiff')) | (events.stim_file.str.contains('sixcatlocdiffisi'))) & \
                                (events.trial_name.str.contains('DIFF')) & \
                                    (events.trial_name.str.contains(axis_cond[i]))].index
                event_idx.append(temp)

        return event_idx
    
def stimulus_onset(t, data, dir):
    """ Computes metrics for the different conditions (e.g. ISI, stim duration).

    params
    -----------------------
    t : array dim(1, T)
        contains timepoints for one trial
    data : array dim(1, T)
        data

    returns
    -----------------------
    onset : float
        contains response onset (in ms) relative to start of stimulus
    idx_first : int
        contains response onset (in timepoints) relative to start of stimulus
    is_onset: Bool
        indicated whether there is a ROL present (True, else False)

    """

    # import timepoints
    timepoints_onepulse = np.loadtxt(dir+'variables/timepoints_onepulse.txt', dtype=int)
    start = timepoints_onepulse[0, 0]

    # standardize data (z-score)
    data_smooth = savgol_filter(data, 101, 3)

    mean = np.mean(data_smooth)
    std = np.std(data_smooth)
    z_score = (data_smooth - mean)/std

    # initiate range over which to compute the ROL
    range_compute = 150
    onset_timepoints = np.zeros(range_compute)
    threshold = st.norm.ppf(0.85)

    # iterate through data and checks if the response is above threshold
    persistence = 60
    count = 0
    idx_first = 0
    for i in range(range_compute):
        if (z_score[i + start] > threshold) & (count == 0):
            onset_timepoints[i] = 1
            idx_first = i
            count = 1
        elif (z_score[i + start] > threshold):
            onset_timepoints[i] = 1

    # check for activation persistence
    is_onset = False
    onset = 0
    if onset_timepoints[idx_first: idx_first + persistence].all() == 1:
        onset = float(t[0][idx_first + start])
        is_onset = True

    # return statement
    if is_onset:
        return onset, idx_first, is_onset
    else:
        return np.nan, np.nan, is_onset
    

def d_prime(data_cat, data_other):
    """ Returns d' value for category selectivity between to datasets.

    params
    -----------------------
    data_cat : array dim(T, events)
        data of the category of which the selectivity is being determined
    data_other : array dim(T, events)
        data of the other categories
    axis : int (0 or 1)
        measure variance over trials (axis=0) or over time (axis=1)

    returns
    -----------------------
    d_prime : float
        selectivity measure for a given category

        """

    # compute means
    mean_cat = np.mean(data_cat) # average over all events (len after first time averaging is 819)
    mean_other = np.mean(data_other)

    # compute variances
    var_cat = np.std(data_cat)**2
    var_other = np.std(data_other)**2

    # compute category selectivity
    d_prime = (mean_cat - mean_other)/(math.sqrt(var_cat + var_other)/2)

    return d_prime

def d_prime_perImgCat(epochs, event_idx, stim, axis=1):
    """ Computes category selectivity using d', to determine category-selectivity.

    params
    -----------------------
    epochs : DataFrame dim(t, n)
        contains the broadband data for each event (n) over time (t)
    event_idx : array dim(1, n)
        contains the index numbers for the events belonging to one experimental condition
    stim : string array dim(1, n)
        contains categories (n)
    axis : int (0 or 1) (optional)
        measure variance over trials (axis=0) or over time (axis=1)

    returns
    -----------------------
    category_selectivity : float
        d' values for every stimulus condition

        """

    # compute category selectivity for all categories
    cat_selectivity = np.zeros(len(stim))

    # iterate over stimulus dataframes and compute d'
    for i in range(len(stim)):

        # event idx of stimulus for which d' is to be computed
        event_idx_cat = event_idx[i]
        data_cat = epochs[event_idx_cat].T

        # initiate dataframe to store data of other categories
        data_other = pd.DataFrame()

        # iterate over all categories and add other categories to dataframe
        for j in range(len(stim)):
            if j != i:
                data = epochs[event_idx[j]]
                data_other = pd.concat([data_other, data.T])

        # average data
        mean_data_cat = np.mean(data_cat, axis=axis)
        mean_data_other = np.mean(data_other, axis=axis)

        # compute d'
        cat_selectivity[i] = d_prime(mean_data_cat, mean_data_other)

    return cat_selectivity