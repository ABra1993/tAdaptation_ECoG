import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd

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

class Models_TTC19:
    """ Simulation of several temporal models to predict a neuronal response given a stimulus
    time series as input.

    This class contains the following modeling components:
    lin : convolves input with an Impulse Response Function (IRF)
    rectf : full-wave rectification
    exp : exponentiation

    Options for divise normalization (i.e. computation for the value of the denominator):
    norm : normalization of the input with a semi-saturation constant
    delay: delayed normalization of the input with a semi-saturation constant

    params
    -----------------------
    stim : array dim(T)
        stimulus time course
    sample_rate : int
        frequency with which the timepoints are measured
    adapt : float
        controls adaptation of stimuli
    tau : float
        time to peak for positive IRF (seconds)
    weight : float
        ratio of negative to positive IRFs
    shift : float
        time between stimulus onset and when the signal reaches the cortex (seconds)
    scale : float
        response gain
    n : float
        exponent
    sigma : float
        semi-saturation constant
    tau_a : time window of adaptation (seconds)

    """

    def __init__(self, stim, sample_rate, shift, scale, weight, tau, k_on, k_off, alpha, lamb, sf_bodies=None, sf_buildings=None, sf_faces=None, sf_objects=None, sf_scenes=None, sf_scrambled=None):

        # assign class variables
        self.shift = shift
        self.scale = scale
        self.weight = weight
        self.tau = tau
        self.k_on = k_on
        self.k_off = k_off
        self.alpha = alpha
        self.lamb = lamb

        self.sf = [sf_bodies, sf_buildings, sf_faces, sf_objects, sf_scenes, sf_scrambled]
        self.stim = ['BODIES', 'BUILDINGS', 'FACES', 'OBJECTS', 'SCENES', 'SCRAMBLED']

        # pre-defined/extracted variables
        self.numtimepts     = len(stim)
        self.srate          = sample_rate

        # stimulus timecourse to add shift
        t = np.multiply(np.arange(1, self.numtimepts+1)/self.srate, 1000)

        # iniate temporal variables
        self.numtimepts = len(stim)
        self.srate = sample_rate

        # compute timepoints (in milliseconds)
        self.t = np.multiply(np.arange(0, self.numtimepts)/self.srate, 1000)

        # parameter values shared by both transient and sustained
        k       = 1.33
        n1      = 9
        n2      = 10

        # IRF sustained response
        gain    = 1
        trans   = 1
        self.irf_sustained  = self.gammaPDF(self.t, tau, n1)

        # IRF sustained response
        gain    = 1.44
        trans   = 1
        self.irf_transient = self.IRF(self.t, k, n1, n2, gain, trans)
    
    def sigmoid(self, x):
        """ Returns values of sigmoidal nonlinearity for on- and off-responses (transient channel).

        params
        -----------------------
        x : array dim(1, T)
            contains timepoints

        returns
        -----------------------
        y : array dim(1, T)
            contains output values for each timepoint
        """

        # initiate
        X = np.abs(x)
        x_p = X.copy(); x_p[x_p < 0] = 0
        x_n = X.copy(); x_n[x_n > 0] = 0
        plt.plot(x_p)
        plt.plot(x_n)

        # Weibull distribution
        weibull_p = 1 - np.exp(-(x_p/self.lamb)**self.k_on)
        weibull_n = 1 - np.exp(-(-x_n/self.lamb)**self.k_off)

        y = np.add(weibull_p, weibull_n)
            
        return y

    def scaling_stimulus(self, input, trial, cond, cat, root):
        """ Adapt stimulus height.

        params
        -----------------------
        input : float
            array containing values of input timecourse
        trial : string
            indicates type of trial (e.g. 'onepulse')
        cond : int
            ISI condition
        cat : str
            image category
        dir : str
            root directory

        returns
        -----------------------
        stim : float
            adapted response

        """

        # create copy of input
        stim = np.zeros(len(input))

        # determine which scaling factor to use
        cat_idx = self.stim.index(cat)
        sf = self.sf[cat_idx]

        # scale stimulus timecourse
        if 'onepulse' in trial:

            # import stimulus timepoints
            timepoints_onepulse = np.loadtxt(root+'variables/timepoints_onepulse.txt', dtype=int)

            # define start and end of stimulus (expressed as timepts)
            start   = timepoints_onepulse[cond, 0]
            end     = timepoints_onepulse[cond, 1]

            # scale timecourse
            stim[start:end] = input[start:end] * sf

        elif 'twopulse' in trial:

            # import stimulus timepoints
            timepoints_twopulse = np.loadtxt(root+'variables/timepoints_twopulse.txt', dtype=int)

            # define start and end of stimulus (expressed as timepts)
            start1  = timepoints_twopulse[cond, 0]
            end1    = timepoints_twopulse[cond, 1]
            start2  = timepoints_twopulse[cond, 2]
            end2    = timepoints_twopulse[cond, 3]

            # scale timecourse
            stim[start1:end1] = input[start1:end1] * sf
            stim[start2:end2] = input[start2:end2] * sf

        return stim

    def response_shift(self, input):
        """ Shifts response in time in the case that there is a delay betwween stimulus onset and response.

        params
        -----------------------
        input : float
            array containing values of input timecourse

        returns
        -----------------------
        stim : float
            adapted response

        """

        # add shift to the stimulus
        # sft = self.shift/(1/self.srate)
        stimtmp = np.pad(input, (int(self.shift), 0), 'constant', constant_values=0)
        stim = stimtmp[0: self.numtimepts]

        return stim
    
    def sustained(self, input, trial, cond, root, t):
        """ Convolves input with the Impulse Resone Function (irf)

        params
        -----------------------
        input : float
            array containing values of input timecourse

        returns
        -----------------------
        linrsp : float
            adapted linear response

        """

        # compute the convolution
        rsp_sustained = np.convolve(input, self.irf_sustained, 'full')
        rsp_sustained = rsp_sustained[0:self.numtimepts]

        # adapt with exponential decay
        rsp_sustained = np.multiply(rsp_sustained, self.exponential_decay(trial, cond, root, t))

        return rsp_sustained

    def transient(self, input):
        """ Convolves input with the Impulse Resone Function (irf)

        params
        -----------------------
        input : float
            array containing values of input timecourse

        returns
        -----------------------
        linrsp : float
            adapted linear response

        """

        # compute the convolution
        rsp_transient = np.convolve(input, self.irf_transient, 'full')
        rsp_transient = rsp_transient[0:self.numtimepts]
        rsp_transient = self.sigmoid(rsp_transient)

        return rsp_transient
    
    def weighted_sum(self, rsp_sustained, rsp_transient):

        rsp = self.scale * (self.weight * rsp_transient + (1 - self.weight) * rsp_sustained)

        return rsp
    
    def scaling_prediction(self, input, cat):
        """ Adapt stimulus height.

        params
        -----------------------
        input : float
            array containing values of input timecourse
        cat : str
            image category

        returns
        -----------------------
        stim : float
            adapted response

        """

        # determine which scaling factor to use
        cat_idx = self.stim.index(cat)
        sf = self.sf[cat_idx]

        # scale with gain
        rsp = sf * input

        return rsp
    
    def sigmoid(self, x):
        """ Returns values of sigmoidal nonlinearity for on- and off-responses (transient channel).

        params
        -----------------------
        x : array dim(1, T)
            contains timepoints

        returns
        -----------------------
        y : array dim(1, T)
            contains output values for each timepoint
        """

        # initiate
        x_p = x.copy(); x_p[x_p < 0] = 0
        x_n = x.copy(); x_n[x_n > 0] = 0

        # Weibull distribution
        y_p = 1 - np.exp(-(x_p/self.lamb)**self.k_on)
        y_n = 1 - np.exp(-(-x_n/self.lamb)**self.k_off)

        y = np.add(y_p, y_n)
            
        return y

    
    def IRF(self, t, k, n1, n2, gain, trans):
        """ Returns values of two weighted gamma functions for a given timeseries.

        params
        -----------------------
        t : array dim(1, T)
            contains timepoints
        tau : float
            peak time
        n1 : float
            effects response decrease after peak
        n2 : float
            effects response decrease after peak for transient response
        gain : float
            scaling 
        trans : float
            transience parameter

        returns
        -----------------------
        y_norm : array dim(1, T)
            contains gamma values for each timepoint
        """

        # exc. and inh. channel
        h1 = self.gammaPDF(t, self.tau, n1)
        h2 = self.gammaPDF(t, k*self.tau, n2)
        
        # weighted sum
        y = np.multiply(gain, np.subtract(h1, np.multiply(h2, trans)))

        return y
        
    def gammaPDF(self, t, tau, n):
        """ Returns values of a gamma function for a given timeseries.

        params
        -----------------------
        t : array dim(1, T)
            contains timepoints
        tau : float
            peak time
        n : float
            effects response decrease after peak

        returns
        -----------------------
        y : array dim(1, T)
            contains gamma values for each timepoint
        """

        y = (t/tau)**(n - 1) * np.exp(-t/tau) / (tau * math.factorial(n - 1))

        return y
    
    def exponential_decay(self, trial, cond, root, t):
        """ Impulse Response Function

        params
        -----------------------
        timepots : int
            length of timeseries
        alpha : float
            peak time

        returns
        -----------------------
        y : array dim(1, T)
            contains value for each timepoint
        """

        # initiate dataframe
        adapt = np.zeros(len(t))
        tempCond = np.loadtxt(root + 'variables/cond_temp.txt', dtype=float)
        tempCond = tempCond/1000 # convert to s

        # duration stimulus
        duration = tempCond[3]

        #  import timepoints
        if trial == 'contrast':

            start = np.argwhere(t > 0)[0][0]

            # apply nonlinearity
            adapt[start:] = np.exp(-np.arange(len(t)-start)/self.alpha)

        elif 'onepulse' in trial:

            start = np.argwhere(t > 0)[0][0]

            # apply nonlinearity
            adapt[start:] = np.exp(-np.arange(len(t)-start)/self.alpha)

        elif 'twopulse' in trial:

            # first stim
            start = np.argwhere(t > 0)[0][0]

            # apply nonlinearity
            adapt[start:] = np.exp(-np.arange(len(t)-start)/self.alpha)

            # first stim
            start = np.argwhere(t > duration + tempCond[cond-1])[0][0]

            # apply nonlinearity
            adapt[start:] = np.exp(-np.arange(len(t)-start)/self.alpha)

        return adapt

# set directory
dir = '/home/amber/OneDrive/code/nAdaptation_ECoG_git_revised/'

t = np.loadtxt(dir+'variables/t.txt', dtype=float)
sample_rate = 512

# generate stimulus timecourse
trial = 'twopulse_repeat'
cond = 5
stim = generate_stimulus_timecourse(trial, cond, dir)

# initiate params
shift = 30 # ca. 0.06 s
scale = 2
weight = 0.5
tau = 4.93
k_on = 0.1
k_off = 0.01
alpha = 500
lamb = 0.1

# initiate model
model = Models_TTC19(stim, sample_rate, shift, scale, weight, tau, k_on, k_off, alpha, lamb)

# shift onset latency
stim_shift = model.response_shift(stim)

# compute response
rsp_sustained = model.sustained(stim_shift, trial, cond, dir, t)   # adaptation (exp. decay)
rsp_transient = model.transient(stim_shift)                         # sigmoidal nonlinearity
rsp = model.weighted_sum(rsp_sustained, rsp_transient)

# plot response
plt.plot(t, stim, label='stim', color='grey')
plt.plot(t, rsp, label='rsp')
plt.plot(t, rsp_sustained, label='sustained')
plt.plot(t, rsp_transient, label='transient')
plt.legend()
plt.savefig(dir + '/mkFigure/TTC19', dpi=300)
