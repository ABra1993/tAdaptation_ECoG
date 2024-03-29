import numpy as np
import math

class Models_DN:
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

    def __init__(self, stim, sample_rate, shift, scale, tau, n, sigma, tau_a, sf_bodies=None, sf_buildings=None, sf_faces=None, sf_objects=None, sf_scenes=None, sf_scrambled=None):

        # assign class variables
        self.shift = shift
        self.scale = scale
        self.tau = tau
        self.n = n
        self.sigma = sigma
        self.tau_a = tau_a

        self.sf = [sf_bodies, sf_buildings, sf_faces, sf_objects, sf_scenes, sf_scrambled]
        # self.sf = [sf_bodies, sf_buildings, sf_faces, sf_objects, sf_scenes]

        # image classes
        self.stim = ['BODIES', 'BUILDINGS', 'FACES', 'OBJECTS', 'SCENES', 'SCRAMBLED']
        # self.stim = ['BODIES', 'BUILDINGS', 'FACES', 'OBJECTS', 'SCENES']

        # iniate temporal variables
        self.numtimepts = len(stim)
        self.srate = sample_rate

        # compute timepoints
        self.t = np.arange(0, self.numtimepts)/self.srate

        # compute the impulse response function (used in the nominator, convolution of the stimulus) were m = 2
        self.irf = self.gammaPDF(self.t, self.tau, 2)
        
        # create exponential decay filter (for the normalization, convolution of the linear response)
        self.norm_irf = self.exponential_decay(self.t, self.tau_a)

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

    def lin(self, input):
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
        linrsp = np.convolve(input, self.irf, 'full')
        linrsp = linrsp[0:self.numtimepts]

        return linrsp

    def rectf(self, input):
        """ Full-wave rectification of the input.

        params
        -----------------------
        input : float
            array containing values of input timecourse

        returns
        -----------------------
        rectf : float
            adapted rectified response

        """

        rectf = abs(input)

        return rectf

    def exp(self, input):
        """ Exponentiation of the input.

        params
        -----------------------
        input : float
            array containing values of input timecourse

        returns
        -----------------------
        exp : float
            adapted exponated response

        """

        exp = input**self.n

        return exp

    # def norm(self, input, linrsp):
    #     """ Normalization of the input.

    #     params
    #     -----------------------
    #     input : float
    #         array containing values of input timecourse
    #     linrsp : float
    #         array containing values of linear response

    #     returns
    #     -----------------------
    #     rsp : float
    #         adapted response

    #     """

    #     # compute the normalized response
    #     demrsp = self.sigma**self.n + abs(linrsp)**self.n                       # semi-saturate + exponentiate
    #     normrsp = input/demrsp                                                  # divide

    #     # scale with gain
    #     rsp = self.scale * normrsp

    #     return rsp

    def norm_delay(self, input, linrsp):
        """ Introduces delay in input

        params
        -----------------------
        input : float
            array containing values of linear + rectf + exp
        linrsp : float
            array containing values of linear response

        returns
        -----------------------
        rsp : float
            adapted response

        """

        # compute the normalized delayed response
        poolrsp = np.convolve(linrsp, self.norm_irf, 'full')                    # delay
        poolrsp = poolrsp[0:self.numtimepts]
        demrsp = self.sigma**self.n + abs(poolrsp)**self.n                      # semi-saturate + exponentiate
        normrsp = input/demrsp                                                  # divide

        # scale with gain
        rsp = self.scale * normrsp

        return rsp, demrsp

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
        
    def gammaPDF(self, t, tau, n):
        """ Returns values of a gamma function for a given timeseries.

        params
        -----------------------
        t : array dim(1, T)
            contains timepoints
        tau : float
            peak time
        n : int
            effects response decrease after peak

        returns
        -----------------------
        y_norm : array dim(1, T)
            contains gamma values for each timepoint
        """

        y = (t/tau)**(n - 1) * np.exp(-t/tau) / (tau * math.factorial(n - 1))
        y_norm = y/np.sum(y)

        return y_norm

    def exponential_decay(self, t, tau):
        """ Impulse Response Function

        params
        -----------------------
        timepots : int
            length of timeseries
        tau : float
            peak time

        returns
        -----------------------
        y_norm : array dim(1, T)
            contains value for each timepoint
        """

        y = np.exp(-t/tau)
        y_norm = y/np.sum(y)

        return y_norm
