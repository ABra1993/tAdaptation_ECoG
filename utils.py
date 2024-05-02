import numpy as np
from scipy.signal import savgol_filter
import scipy.stats as st
import math
import numpy as np
import pandas as pd


def import_info(subject, dir):
    """ Returns experimental info of the subject.

    params
    -----------------------
    subject : string
        indicates subject
    dir : string
        indicates root directory

    returns
    -----------------------
    t : DataFrame
        contains timepoints
    events : array dim(n , m)
        contains the events (n) with additional info (m)
    channels : DataFrame dim(n, m)
        contains electrodes (n) with additional info (m).
    electrodes : DataFrame dim(n, m)
        contains electrodes (n) with additional info (m).

        """

    # read files
    t = pd.read_csv(dir + 'data_subjects/' + subject + '/t.txt', header=None)
    events = pd.read_csv(dir + 'data_subjects/' + subject + '/events.txt', header=0)
    channels = pd.read_csv(dir + 'data_subjects/' + subject + '/channels.txt', header=0)
    electrodes = pd.read_csv(dir + 'data_subjects/' + subject + '/electrodes.txt', header=0)

    return t, events, channels, electrodes

def import_epochs(subject, electrode_idx, dir):
    """ Returns broadband data.

    params
    -----------------------
    subject : string
        indicates subject
    electrode_idx : int
        contains index number of specified electrode
    dir : string
        indicates root directory

    returns
    -----------------------
    epochs_b : array dim(n, T)
        contains the broadband data for each event (n) and timepiont (T)

        """

    # os.chdir(dir+'Documents/ECoG_MATLAB/' + subject + '/epochs_b/')
    epochs_b = []
    if type(electrode_idx) == int:                                              # import data single electrode
        epochs_b = pd.read_csv(dir+'data_subjects/' + subject + '/epochs_b/epochs_b_channel' + str(electrode_idx + 1) + '_baselineCorrection.txt', delimiter=' ', header=None)
    else:                                                                       # import data multiple electrodes
        for i in range(len(electrode_idx)):
            temp = pd.read_csv(dir+'data_subjects/' + subject + '/epochs_b/epochs_b_channel' + str(electrode_idx[i] + 1) + '_baselineCorrection.txt', delimiter=' ', header=None)
            epochs_b.append(temp)

    return epochs_b

def select_electrodes(channels, electrode_name):
    """ Returns index of selected electrodes.

        params
        -----------------------
        channels : DataFrame dim(n, m)
            contains electrodes (n) with additional info (m).
        electrode_name : string
            indicates name of electrode for which index should be determined.

        returns
        -----------------------
        electrode_idx : DataFrame dim(1)
            contains index number of specified electrode(s)
    """

    electrode_idx = []
    if electrode_name == 'all':
        electrode_idx = np.arange(0, len(channels))
    elif electrode_name == 'grid':
        for i in range(len(channels)):
            if channels.name[i][0] == 'G':
                electrode_idx.append(i)
    elif electrode_name == 'depth':
        for i in range(len(channels)):
            if channels.name[i][0] == 'D':
                electrode_idx.append(i)
    else:
        if len(channels.loc[channels.name == electrode_name].index) == 0:
            return -1
        else:
            electrode_idx = int(channels.loc[channels.name == electrode_name].index[0])

    return electrode_idx

def select_events(events, cond, trial, dir):
    """ Returns index of events for either duration or repetition trials (per experimental condition).

    params
    -----------------------
    events : array dim(n , m)
        contains the events (n) with additional info (m)
    cond : string ('TEMP', 'STIM')
        indicates the type of experimental manipulations 
            - temporal ('TEMP') by ISI (for repetition trials) or stimulus duration (for duration trials) 
            - stimulus ('STIM') by image category
    trial : string
        indicates type of trial (e.g. 'onepulse')
    dir : string
        indicates root directory

    returns
    -----------------------
    event_idx : nested list
        contains the index numbers for the events belonging to one experimental condition

    """

    
    if cond == None:

        if trial == 'onepulse-4':     # used to compute AUC of second pulse

            event_idx = events[(events.trial_name.str.contains('ONEPULSE-4'))].index
            return event_idx

        else: 
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
    
    elif cond == 'both':

        # define temporal and category conditions
        axis_stim = np.loadtxt(dir+'variables/cond_stim.txt', dtype=str)
        axis_temp = ['1', '2', '3', '4', '5', '6']

        # retrieve event indices
        event_idx = []
        if trial == 'all':
            for j in range(len(axis_stim)):
                temp = []
                for k in range(len(axis_temp)):
                    temp2 = events[(events.trial_name.str.contains(axis_stim[j])) & \
                                    (events.trial_name.str.contains(axis_temp[k]))].index
                    temp.append(temp2)
                event_idx.append(temp)

        elif trial == 'onepulse':
            for j in range(len(axis_stim)):
                temp = []
                for k in range(len(axis_temp)):
                    temp2 = events[(events.stim_file.str.contains('sixcatloctemporal')) & \
                                    (events.trial_name.str.contains('ONE')) & \
                                        (events.trial_name.str.contains(axis_stim[j])) & \
                                            (events.trial_name.str.contains(axis_temp[k]))].index
                    temp.append(temp2)
                event_idx.append(temp)

        elif trial == 'twopulse':
            for j in range(len(axis_stim)):
                temp = []
                for k in range(len(axis_temp)):
                    temp2 = events[(events.trial_name.str.contains('TWO')) & \
                                        (events.trial_name.str.contains(axis_stim[j])) & \
                                            (events.trial_name.str.contains(axis_temp[k]))].index
                    temp.append(temp2)
                event_idx.append(temp)

        elif trial == 'twopulse_repeat':
            for j in range(len(axis_stim)):
                temp = []
                for k in range(len(axis_temp)):
                    temp2 = events[(events.stim_file.str.contains('sixcatloctemporal')) & \
                                    (events.trial_name.str.contains('TWO')) & \
                                        (events.trial_name.str.contains(axis_stim[j])) & \
                                            (events.trial_name.str.contains(axis_temp[k]))].index
                    temp.append(temp2)
                event_idx.append(temp)

        elif trial == 'twopulse_nonrepeat':
            for j in range(len(axis_stim)):
                temp = []
                for k in range(len(axis_temp)):
                    temp2 = events[((events.stim_file.str.contains('sixcatlocisidiff')) | (events.stim_file.str.contains('sixcatlocdiffisi'))) & \
                                    (events.trial_name.str.contains(axis_stim[j])) & \
                                        (events.trial_name.str.contains(axis_temp[k]))].index
                    temp.append(temp2)
                event_idx.append(temp)

        elif trial == 'twopulse_nonrepeat_same':
            for j in range(len(axis_stim)):
                temp = []
                for k in range(len(axis_temp)):
                    temp2 = events[((events.stim_file.str.contains('sixcatlocisidiff')) | (events.stim_file.str.contains('sixcatlocdiffisi'))) & \
                                    (events.trial_name.str.contains('SAME')) & \
                                        (events.trial_name.str.contains(axis_stim[j])) & \
                                            (events.trial_name.str.contains(axis_temp[k]))].index
                    temp.append(temp2)
                event_idx.append(temp)

        elif trial == 'twopulse_nonrepeat_diff':
            for j in range(len(axis_stim)):
                temp = []
                for k in range(len(axis_temp)):
                    temp2 = events[((events.stim_file.str.contains('sixcatlocisidiff')) | (events.stim_file.str.contains('sixcatlocdiffisi'))) & \
                                    (events.trial_name.str.contains('DIFF')) & \
                                        (events.trial_name.str.contains(axis_stim[j])) & \
                                            (events.trial_name.str.contains(axis_temp[k]))].index
                    temp.append(temp2)
                event_idx.append(temp)

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
            elif trial == 'twopulse_repeat': #twopulse_repeat
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

def select_events_durationTrials(events, tempCond, preference, cat=None):
    """ Returns index of events for duration trials per temporal condition.

    params
    -----------------------
    events : array dim(n , m)
        contains the events (n) with additional info (m)
    tempCond: list float
        list containing the stimulus durations for duration trials ([17, 33, 67, 133, 267, 533])
    preference: int
        0 including ALL trials, 1 including PREFERRED trials or 2 including NON-PREFERRED trials
    cat: string list
        includes preferred and non-preferred image category (e.g. [FACES, SCENES])
        
    returns
    -----------------------
    event_idx : nested list
        contains the index numbers for the events belonging to one experimental condition

    """

    event_idx = []
    if preference == 0:

        # ONEPULSE trials per stimulus duration
        for j in range(len(tempCond)):
            temp = events[(events.trial_name.str.contains('ONE')) & \
                            (events.trial_name.str.contains(str(j+1)))].index
            event_idx.append(temp)

    elif preference == 1: # PREFERRED trials

        # ONEPULSE trials per stimulus duration
        for j in range(len(tempCond)):
            temp = events[(events.trial_name.str.contains('ONE')) & \
                            (events.trial_name.str.contains(str(j+1))) & \
                                (events.trial_name.str.contains(cat[0]))].index
            event_idx.append(temp)

    elif preference == 2: # NONPREFERRED trials

        # ONEPULSE trials per stimulus duration
        for j in range(len(tempCond)):
            temp = events[(events.trial_name.str.contains('ONE')) & \
                            (events.trial_name.str.contains(str(j+1))) & \
                                (~events.trial_name.str.contains(cat[0]))].index
            event_idx.append(temp)

    return event_idx

def select_events_repetitionTrials(events, ISI, preference, cat=None):
    """ Returns index of events for repetition trials per temporal condition.

    params
    -----------------------
    events : array dim(n , m)
        contains the events (n) with additional info (m)
    tempCond: list float
        list containing the ISI for repetition trials ([17, 33, 67, 133, 267, 533])
    preference: int
        0 including ALL trials, 1 including PREFERRED trials or 2 including NON-PREFERRED trials
    cat: string list
        includes preferred and non-preferred image category (e.g. [FACES, SCENES])
        
    returns
    -----------------------
    event_idx : nested list
        contains the index numbers for the events belonging to one experimental condition

    """

    event_idx = []
    if preference == 0: # all trials

        # ONEPULSE trials
        temp = events[(events.trial_name.str.contains('ONEPULSE-4'))].index
        event_idx.append(temp)

        # TWOPULSE trials per ISI
        temp = []
        for j in range(len(ISI)):
            temp2 = events[(events.trial_name.str.contains('TWO')) & \
                            (events.trial_name.str.contains(str(j+1)))].index
            temp.append(temp2)
        event_idx.append(temp)

        # REPEAT trials per ISI
        temp = []
        for j in range(len(ISI)):
            temp2 = events[(events.stim_file.str.contains('sixcatloctemporal')) & \
                            (events.trial_name.str.contains('TWO')) & \
                                    (events.trial_name.str.contains(str(j+1)))].index
            temp.append(temp2)
        event_idx.append(temp)
    
    elif preference == 1: # PREFERRED trials

        # ONEPULSE trials
        temp = events[(events.trial_name.str.contains('ONEPULSE-4')) & \
                        (events.trial_name.str.contains(cat[0]))].index
        event_idx.append(temp)

        # TWOPULSE trials per ISI
        temp = []
        for j in range(len(ISI)):
            temp2 = events[(events.trial_name.str.contains('TWO')) & \
                            (events.trial_name.str.contains(str(j+1))) & \
                                (events.trial_name.str.contains(cat[0]))].index
            temp.append(temp2)
        event_idx.append(temp)

        # REPEAT trials per ISI
        temp = []
        for j in range(len(ISI)):
            temp2 = events[(events.stim_file.str.contains('sixcatloctemporal')) & \
                            (events.trial_name.str.contains('TWO')) & \
                                (events.trial_name.str.contains(cat[0])) & \
                                    (events.trial_name.str.contains(str(j+1)))].index
            temp.append(temp2)
        event_idx.append(temp)

    elif preference == 2: # NONPREFERRED trials

        # ONEPULSE trials
        temp = events[(events.trial_name.str.contains('ONEPULSE-4')) & \
                        (~events.trial_name.str.contains(cat[0]))].index
        event_idx.append(temp)

        # TWOPULSE trials per ISI
        temp = []
        for j in range(len(ISI)):
            temp2 = events[(events.trial_name.str.contains('TWO')) & \
                            (events.trial_name.str.contains(str(j+1))) & \
                                (~events.trial_name.str.contains(cat[0]))].index
            temp.append(temp2)
        event_idx.append(temp)

        # REPEAT trials per ISI
        temp = []
        for j in range(len(ISI)):
            temp2 = events[(events.stim_file.str.contains('sixcatloctemporal')) & \
                            (events.trial_name.str.contains('TWO')) & \
                                (~events.trial_name.str.contains(cat[0])) & \
                                    (events.trial_name.str.contains(str(j+1)))].index
            temp.append(temp2)
        event_idx.append(temp)

    return event_idx

def stimulus_onset(t, data, dir):
    """ Computes stimulus onset of the neural response (from the start of the stimulus presentation).

    params
    -----------------------
    t : array dim(1, T)
        contains timepoints for one trial
    data : array dim(1, T)
        data
    dir : string
        indicates root directory

    returns
    -----------------------
    onset : float
        contains response onset (in ms) relative to start of stimulus
    idx_first : int
        contains response onset (in timepoints) relative to start of stimulus
    is_onset: Bool
        indicated whether there is a ROL detected (True, else False)

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

def estimate_first_pulse(t, epochs_b, event_idx, timepoints_twopulse):
    """ Estimates timecourse of onepulse trials.

    params
    -----------------------
    t : array dim(1, T)
        contains timepoints for one trial
    epochs_b : DataFrame dim(t, n)
        contains the broadband data for each event (n) over time (t)
    event_idx : array dim(1, n)
        contains the index numbers for the events belonging to one experimental condition
    timepoints_twopulse : array dim(n, m)
        timepoints of stimulus on- and offset(m) for the different ISIs (n)

    returns
    -----------------------
    onepulse_mean : array dim(1, T)
        timecourse of the estimation of onepulse trials

    """

    # average data first pulse
    onepulse_mean = np.zeros((1, len(t)))
    onepulse_mean[0, :] = np.nanmean(epochs_b[event_idx[0]] , axis=1)                          # final array

    # average data second pulse
    start_trial = int(timepoints_twopulse[0, 0])                                # start first pulse
    for i in range(len(event_idx[1])):                                          # iterate through 'twopulse trials'

        # determine timepoints up to which data should be collected
        end_trial = timepoints_twopulse[i, 2] # tmp start second pulse

        # initiate array
        twopulse_mean = np.zeros((1, len(t)))

        # obtain data and average over events
        # print(event_idx[1][i])
        twopulse_mean[0, :] = np.nanmean(epochs_b[event_idx[1][i]], axis=1)

        # add selected data two array
        temp = np.append(onepulse_mean, twopulse_mean, axis=0)
        onepulse_mean[0, start_trial:end_trial] = np.mean(temp[:, start_trial:end_trial], axis=0)

    onepulse_mean = onepulse_mean[0, :]

    return onepulse_mean

def recovery_perISI(t, epochs_b, event_idx, num, time_window, dir):
    """ Returns Area Under the Curve (AUC) of the second pulse as a proportion of the first pulse.

    params
    -----------------------
    t : array dim(1, T)
        contains timepoints for one trial
    epochs_b : DataFrame dim(t, n)
        contains the broadband data for each event (n) over time (t)
    event_idx : array dim(1, n)
        contains the index numbers for the events belonging to one experimental condition
        [ONEPULSE-4, TWOPULSE, events-TRIAL1, ..., events-TRIALn]
    num : int
        indicates trial type
    time_window : float
        time range over which to measure the AUC
    dir : string
        root directory

    returns
    -----------------------
    AUC2_prop : array dim(1, n)
        AUC of second pulse proportional to the first pulse for all the ISIs (n)
        data_onepulse_mean : float (optional)
        array with the estmiated broadband of the first pulse

    """

    # initiate arrays
    AUC2_prop = np.zeros(len(event_idx[num]))

    # import timepoints of on- and offset of stimulus for one and twopulse trials
    timepoints_onepulse = np.loadtxt(dir+'variables/timepoints_onepulse.txt', dtype=int)
    timepoints_twopulse = np.loadtxt(dir+'variables/timepoints_twopulse.txt', dtype=int)
    start_trial = timepoints_onepulse[0, 0]

    # compute estimate of first pulse
    data_onepulse_mean = estimate_first_pulse(t, epochs_b, event_idx, timepoints_twopulse)

    # compute AUC first pulse
    end = start_trial + time_window
    AUC1 = np.trapz(data_onepulse_mean[start_trial:end])

    # determine AUC second pulse and compute ISI recovery
    for i in range(len(event_idx[num])):
            
        # extract data and average data
        data = epochs_b[event_idx[num][i]]
        data_mean = np.nanmean(data, axis=1)

        # extract first pulse from second pulse
        data_twopulse_mean = data_mean - data_onepulse_mean

        # compute AUC second pulse
        start = timepoints_twopulse[i, 2]
        end = start + time_window
        AUC2 = np.trapz(data_twopulse_mean[start:end])

        # compute AUC2
        AUC2_prop[i] = AUC2/AUC1

    # return function
    return AUC2_prop, data_onepulse_mean

def d_prime(data_cat, data_other):
    """ Returns d' value for category selectivity between to datasets.

    params
    -----------------------
    data_cat : array dim(T, events)
        data of the category of which the selectivity is being determined
    data_other : array dim(T, events)
        data of the other categories

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

def generate_stimulus_timecourse(trial, cond, dir):
    """ Generates stimulus timecourse

    params
    -----------------------
    trial : string
        indicates type of trial (e.g. 'onepulse')
    cond : int
        ISI condition
    dir : str
        root directory

    returns
    -----------------------
    T_input : array dim(1, T)
        stimulus timecourse

    """

    # initaite time course stimulus
    t = np.loadtxt(dir+'variables/t.txt')
    timepts = len(t)
    T_input = np.zeros(timepts)

    # extract timepoints on- and offset stimulus
    if 'onepulse' in trial:
        timepoints_onepulse = np.loadtxt(dir+'variables/timepoints_onepulse.txt')
        T_input[int(timepoints_onepulse[cond, 0]):int(timepoints_onepulse[cond, 1])] = 1
    elif 'twopulse' in trial:
        timepoints_twopulse = np.loadtxt(dir+'variables/timepoints_twopulse.txt')
        T_input[int(timepoints_twopulse[cond, 0]):int(timepoints_twopulse[cond, 1])] = 1
        T_input[int(timepoints_twopulse[cond, 2]):int(timepoints_twopulse[cond, 3])] = 1

    return T_input

def r_squared(data, fit):
    """ Computes r-square which represents the proportion of the variance for a
    dependent variable that's explained by an independent variable.

    params
    -----------------------
    data : array dim(1, n)
        data with n timepoints
    fit : array dim(1, n)
        simulation of the model with n timepoints

    returns
    -----------------------
    r_squared: float
        value of the r-square

    """

    # average
    mean = np.nanmean(data)

    # compute residual sum of squares
    SS_res = np.nansum((data-fit)**2)

    # compute total sum of squares
    SS_tot = np.nansum((data-mean)**2)

    # coefficient of determination
    try:
        r_squared = 1 - SS_res/SS_tot
    except:
        r_squared = np.nan

    return r_squared

