import numpy as np

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

    if cond == 'both':

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
                    # print(len(temp2))
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
                    # print(len(temp2))
                event_idx.append(temp)

        return event_idx

        return event_idx