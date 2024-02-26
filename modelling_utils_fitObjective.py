import numpy as np
import matplotlib.pyplot as plt

from models.Models_DN import Models_DN
from models.Models_TTC17 import Models_TTC17
from models.Models_TTC19 import Models_TTC19


# models (with or without category-selectivity):
# DN
# TTC17
# TTC19

def objective(params, model_type, scaling, X, y, sample_rate, root=None, t=None, info=None):
    """ Objective function used to train the Delayed Normalization (DN) model. It iterates over the input
        samples and returns the loss (i.e. Mean Squared Error, MSE)

    params
    -----------------------
    model_type : string
        type of DN model (DN, TTC17, TTC19)
    params : string
        array containing the names of the free model parameters
    X : float
        array containing N rows of input samples (i.e. stimulus timecourses)
    y : float
        array containing per input sample, the actual neural response
    sample_rate : int
        sample frequency
    root : str (default = None)
        root directory
    info : pandas dataframe (default = None)
        containing trial info

    returns
    -----------------------
    fit : float
        loss (i.e. MSE)

    """

    # compute model and fit
    fit = 0
    for i in range(len(X)):

        # simulate model
        if (scaling == 'S') | (scaling == 'P') | (scaling == 'S_withoutScrambled'):

            # retrieve trial info
            trial = info.loc[i, 'trial_type']
            cond = int(info.loc[i, 'temp_cond'])
            cat = info.loc[i, 'img_cat']

            # simulate model
            model_run = model(model_type, scaling, X[i, :], sample_rate, params, root, trial, cond, cat)

        elif (model_type == 'TTC19'):

            # retrieve trial info
            trial = info.loc[i, 'trial_type']
            cond = int(info.loc[i, 'temp_cond'])

            # simulate model
            model_run = model(model_type, scaling, X[i, :], sample_rate, params, root, trial, cond, t=t)
        
        else:
            
            model_run = model(model_type, scaling, X[i, :], sample_rate, params)

        # compute cost
        fit = fit + np.sum((model_run - y[i, :])**2)

    return fit


def model(model_type, scaling, stim, sample_rate, params, root=None, trial=None, cond=None, cat=None, t=None, denom=False):
    """ Initiates the Delayed Normalization (DN) model and computes the predicted response by the model.

    params
    -----------------------
    stim : float
        array representing input sample (i.e. stimulus timecourse) with N timepoits
    sample_rate : int
        sample frequency
    params : string
        array containing the names of the free model parameters
    root : str (default = None)
        root directory
    trial : str (default = None)
        duration or repetition trial
    cond : int (default = None)
        temporal condition (i.e. stimulus duration or inter-trial-interval)
    cat : str (default = None)
        stimulus category (e.g. 'FACE')
    denom : Bool (default = False)
        specifies whehter to return the input drive and normalisation pool seperately
    
    returns
    -----------------------
    linear_rectf_exp_norm_delay : float
        array containing predicted response of N timepoints
    linear_rectf_exp : float
        input drive
    demrsp : float
        normalisation pool

    """

    if model_type == 'DN':

        shift       = params[0]
        scale       = params[1]
        tau         = params[2]
        n           = params[3]
        sigma       = params[4]
        tau_a       = params[5]

        if (scaling == 'S') | (scaling == 'P'):

            sf_bodies       = params[6]
            sf_buildings    = params[7]
            sf_faces        = params[8]
            sf_objects      = params[9]
            sf_scenes       = params[10]
            sf_scrambled    = params[11]

            model = Models_DN(stim, sample_rate, shift, scale, tau, n, sigma, tau_a, sf_bodies, sf_buildings, sf_faces, sf_objects, sf_scenes, sf_scrambled)

        # elif (scaling == 'S_withoutScrambled'):

        #     sf_bodies       = params[6]
        #     sf_buildings    = params[7]
        #     sf_faces        = params[8]
        #     sf_objects      = params[9]
        #     sf_scenes       = params[10]

        #     model = Models_DN(stim, sample_rate, shift, scale, tau, n, sigma, tau_a, sf_bodies, sf_buildings, sf_faces, sf_objects, sf_scenes)

        else:

            model = Models_DN(stim, sample_rate, shift, scale, tau, n, sigma, tau_a)

    elif model_type ==  'TTC17':

        shift       = params[0]
        scale       = params[1]
        weight      = params[2]

        if (scaling == 'S') | (scaling == 'P'):

            sf_bodies       = params[3]
            sf_buildings    = params[4]
            sf_faces        = params[5]
            sf_objects      = params[6]
            sf_scenes       = params[7]
            sf_scrambled    = params[8]

            model = Models_TTC17(stim, sample_rate, shift, scale, weight, sf_bodies, sf_buildings, sf_faces, sf_objects, sf_scenes, sf_scrambled)

        else:

            model = Models_TTC17(stim, sample_rate, shift, scale, weight)

    elif model_type == 'TTC19':

        shift       = params[0]
        scale       = params[1]
        weight      = params[2]
        tau         = params[3]
        k_on        = params[4]
        k_off       = params[5]
        alpha       = params[6]
        lamb        = params[7]

        if (scaling == 'S') | (scaling == 'P'):

            sf_bodies       = params[8]
            sf_buildings    = params[9]
            sf_faces        = params[10]
            sf_objects      = params[11]
            sf_scenes       = params[12]
            sf_scrambled    = params[13]

            model = Models_TTC19(stim, sample_rate, shift, scale, weight, tau, k_on, k_off, alpha, lamb, sf_bodies, sf_buildings, sf_faces, sf_objects, sf_scenes, sf_scrambled)

        else:

            model = Models_TTC19(stim, sample_rate, shift, scale, weight, tau, k_on, k_off, alpha, lamb)

    # apply category scaling on (S)timulus timecourse
    if (scaling == 'S') | (scaling == 'S_withoutScrambled'):
        stim = model.scaling_stimulus(stim, trial, cond, cat, root)

    # shift onset latency
    stim_shift = model.response_shift(stim)

    # compute model
    if model_type ==  'DN':

        linear = model.lin(stim_shift)
        linear_rectf = model.rectf(linear)
        linear_rectf_exp = model.exp(linear_rectf)
        rsp, demrsp = model.norm_delay(linear_rectf_exp, linear)
    
    elif model_type == 'TTC17':

        rsp_sustained = model.sustained(stim_shift)
        rsp_transient = model.transient(stim_shift)
        rsp = model.weighted_sum(rsp_sustained, rsp_transient)

    elif model_type == 'TTC19':

        rsp_sustained = model.sustained(stim_shift, trial, cond, root, t)   # adaptation (exp. decay)
        rsp_transient = model.transient(stim_shift)                         # sigmoidal nonlinearity
        rsp = model.weighted_sum(rsp_sustained, rsp_transient)

    # apply scaling on (P)redicted neural timecourse
    if scaling == 'P':
        rsp = model.scaling_prediction(rsp, cat)

    # return i) predicted response, ii) input drive, iii) normalisation pool
    if model_type == 'DN':
        if denom:
            return rsp, linear_rectf_exp, demrsp
        else:
            return rsp
    elif (model_type == 'TTC17') | (model_type == 'TTC19'):
        return rsp


def OF_ISI_recovery_log(t, c, a):
    """ Linear fitting curve

    params
    -----------------------
    t : array (1xT-dimensional) or scalar
        table/scalar containing timepoint(s)
    b, c : scalar
        model parameters to be fitted

    returns
    -----------------------
    function : scalar
        value of function at timepoints t with parameters a, b and c.

    """

    y = c + a * np.log(t)

    return y

    



