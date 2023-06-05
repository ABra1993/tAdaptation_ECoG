import numpy as np

from models.Models_DN import Models_DN
from models.Models_csDN import Models_csDN

def objective_DN(params, X, y, sample_rate, model):
    """ Objective function used to train the Delayed Normalization (DN) model. It iterates over the input
        samples and returns the loss (i.e. Mean Squared Error, MSE)

    params
    -----------------------
    params : string
        array containing the names of the free model parameters
    X : float
        array containing N rows of input samples (i.e. stimulus timecourses)
    y : float
        array containing per input sample, the actual neural response
    sample_rate : int
        sample frequency
    model : string
        type of DN model (e.g. Zhou or Groen)

    returns
    -----------------------
    fit : float
        loss (i.e. MSE)

    """

    # compute model and fit
    fit = 0
    for i in range(len(X)):

        # simulate model
        model_run = model_DN(X[i, :], sample_rate, params, model)

        # compute cost
        fit = fit + np.sum((model_run - y[i, :])**2)

    return fit

def model_DN(stim, sample_rate, params, denom=False):
    """ Initiates the Delayed Normalization (DN) model and computes the predicted response by the model.

    params
    -----------------------
    stim : float
        array representing input sample (i.e. stimulus timecourse) with N timepoits
    sample_rate : int
        sample frequency
    params : string
        array containing the names of the free model parameters
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

    # define params
    tau         = params[0]
    shift       = params[1]
    scale       = params[2]
    n           = params[3]
    sigma       = params[4]
    tau_a       = params[5]

    # initiate model
    model = Models_DN(stim, sample_rate, tau, shift, scale, n, sigma, tau_a)

    # compute delayed normalisation model
    stim_shift = model.response_shift(stim)
    linear = model.lin(stim_shift)
    linear_rectf = model.rectf(linear)
    linear_rectf_exp = model.exp(linear_rectf)
    if denom:
        linear_rectf_exp_norm_delay, demrsp = model.norm_delay(linear_rectf_exp, linear, denom=denom)
    else:
        linear_rectf_exp_norm_delay = model.norm_delay(linear_rectf_exp, linear)

    # return
    if denom:
        return linear_rectf_exp_norm_delay, linear_rectf_exp, demrsp
    else:
        return linear_rectf_exp_norm_delay
    

# def objective_DN(params, X, y, sample_rate, model, plaw=False):
def objective_csDN(params, X, y, info, sample_rate, dir):
    """ Objective function used to train the Category-based Delayed Normalization (CbDN) model. It iterates over the input
        samples and returns the loss (i.e. Mean Squared Error, MSE)

    params
    -----------------------
    params : string
        array containing the names of the free model parameters
    X : float
        array containing N rows of input samples (i.e. stimulus timecourses)
    y : float
        array containing per input sample, the actual neural response
    info : pandas dataframe
        contains info (i.e. trialtype and condition, ISI) per input sample (i.e. stimulus timecourse)
    sample_rate : int
        sample frequency
    dir : string
        root directory

    returns
    -----------------------
    fit : float
        loss (i.e. MSE)

    """

    # compute model and fit
    fit = 0
    for i in range(len(X)):

        # retrieve trial and condition
        trial = info.loc[i, 'trial']
        cond = int(info.loc[i, 'ISI'])
        cat = info.loc[i, 'img_cat']

        # simulate model
        _, model_run = model_cdDN(X[i, :], trial, cond, cat, sample_rate, params, dir)

        # compute cost
        fit = fit + np.sum((model_run - y[i, :])**2)

    return fit


def model_cdDN(stim, trial, cond, cat, sample_rate, params, dir, denom=False):
    """ Initiates the Category-based Delayed Normalization (CbDN) model and computes the predicted response by the model.

    params
    -----------------------
    stim : float
        array representing input sample (i.e. stimulus timecourse) with N timepoits
    S : int
        array of 0's and 1's, where the 1 indicated stimulus index which is adapted (i.e. first or second stimulus timecourse)
    trial : string
        indicates type of trial (e.g. 'onepulse', 'twopulse_repeat_pref')
    cond : int
        ISI condition
    sample_rate : int
        sample frequency
    params : string
        array containing the names of the free model parameters
    dir : string
        root directory
    denom : Bool (default = False)
        specifies whehter to return the input drive and normalisation pool seperately

    returns
    -----------------------
    stim_adapt : float
        array containing adapted stimulus timecourse based on categeroy preference
    linear_rectf_exp_norm_delay : float
        array containing predicted response of N timepoints
    linear_rectf_exp : float
        input drive
    demrsp : float
        normalisation pool

    """

    # define params
    tau             = params[0]
    shift           = params[1]
    scale           = params[2]
    n               = params[3]
    sigma           = params[4]
    tau_a           = params[5]
    sf_bodies       = params[6]
    sf_buildings    = params[7]
    sf_faces        = params[8]
    sf_objects      = params[9]
    sf_scenes       = params[10]
    sf_scrambled    = params[11]

    # initiate model
    model = Models_csDN(stim, sample_rate, tau, shift, scale, n, sigma, tau_a, sf_bodies, sf_buildings, sf_faces, sf_objects, sf_scenes, sf_scrambled)

    # adapt stim according to image cateogory
    stim_adapt = model.response_adapt(stim, trial, cond, cat, dir)

    # introduce shift
    stim_shift = model.response_shift(stim_adapt)

    # compute models
    linear = model.lin(stim_shift)
    linear_rectf = model.rectf(linear)
    linear_rectf_exp = model.exp(linear_rectf)
    if denom: # return values of nominator and denominator seperately
        linear_rectf_exp_norm_delay, demrsp = model.norm_delay(linear_rectf_exp, linear, cat, denom=denom)
    else:
        linear_rectf_exp_norm_delay = model.norm_delay(linear_rectf_exp, linear, cat)

    # return
    if denom: 
        return stim_adapt, linear_rectf_exp_norm_delay, linear_rectf_exp, demrsp
    else:
        return stim_adapt, linear_rectf_exp_norm_delay