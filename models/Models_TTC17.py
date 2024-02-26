import numpy as np
import math

class Models_TTC17:
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

    def __init__(self, stim, sample_rate, shift, scale, weight, sf_bodies=None, sf_buildings=None, sf_faces=None, sf_objects=None, sf_scenes=None, sf_scrambled=None):

        # assign class variables
        self.shift = shift
        self.scale = scale
        self.weight = weight

        self.sf = [sf_bodies, sf_buildings, sf_faces, sf_objects, sf_scenes, sf_scrambled]

        # iniate temporal variables
        self.numtimepts = len(stim)
        self.srate = sample_rate

        # image classes
        self.stim = ['BODIES', 'BUILDINGS', 'FACES', 'OBJECTS', 'SCENES', 'SCRAMBLED']

        # compute timepoints (in milliseconds)
        self.t = np.multiply(np.arange(0, self.numtimepts)/self.srate, 1000)

        # parameter values shared by both transient and sustained
        tau     = 4.94
        k       = 1.33
        n1      = 9
        n2      = 10

        # IRF sustained response
        gain    = 1
        trans   = 0
        self.irf_transient  = self.IRF(tau, k, n1, n2, gain, trans)

        # IRF sustained response
        gain    = 1.44
        trans   = 1
        self.irf_sustained = self.IRF(tau, k, n1, n2, gain, trans)

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
    
    def sustained(self, input):
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
        rsp_transient = np.abs(rsp_transient)

        return rsp_transient
    
    def weighted_sum(self, rsp_sustained, rsp_transient):
        """ Response by weighting sum of transient and sustained channel

        params
        -----------------------
        rsp_sustained : array (float)
            array containing values of input timecourse of sustained channel
        rsp_sustained : array (float)
            array containing values of input timecourse of transient channel

        returns
        -----------------------
        rsp : float
            predicted neural response

        """

        rsp = np.multiply(self.scale, np.add(np.multiply(self.weight, rsp_transient), np.multiply(1 - self.weight, rsp_sustained)))

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
    
    def IRF(self, tau, k, n1, n2, gain, trans):
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
        h1 = self.gammaPDF(self.t, tau, n1)
        h2 = self.gammaPDF(self.t, k*tau, n2)
        
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
        y_norm : array dim(1, T)
            contains gamma values for each timepoint
        """

        y = (t/tau)**(n - 1) * np.exp(-t/tau) / (tau * math.factorial(n - 1))
        # y = y/np.sum(y)

        return y
