import numpy as np

from utils import gammaPDF, exponential_decay

class Models_csDN:
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

    def __init__(self, stim, sample_rate, tau, shift, scale, n, sigma, tau_a, sf_bodies, sf_buildings, sf_faces, sf_objects, sf_scenes, sf_scrambled):

        # assign class variables
        self.tau = tau
        self.shift = shift
        self.scale = scale
        self.n = n
        self.sigma = sigma
        self.tau_a = tau_a

        self.sf = [sf_bodies, sf_buildings, sf_faces, sf_objects, sf_scenes, sf_scrambled]

        # iniate temporal variables
        self.numtimepts = len(stim)
        self.srate = sample_rate

        # image classes
        self.stim = ['BODIES', 'BUILDINGS', 'FACES', 'OBJECTS', 'SCENES', 'SCRAMBLED']

        # compute timepoints
        self.t = np.arange(0, self.numtimepts)/self.srate

        # compute the impulse response function (used in the nominator, convolution of the stimulus)
        self.irf = gammaPDF(self.t, self.tau, 2)
        
        # create exponential decay filter (for the normalization, convolution of the linear response)
        self.norm_irf = exponential_decay(self.t, self.tau_a)

    def response_adapt(self, input, trial, cond, cat, dir, model_ISI=False, timepts=None):
        """ Adapt stimulus height.

        params
        -----------------------
        input : float
            array containing values of input timecourse
        S : int
            array of 0's and 1's, where the 1 indicated stimulus index which is adapted
        trial : string
            indicates type of trial (e.g. 'onepulse')
        cond : int
            ISI condition
        dir : str
            root directory
        model_ISI : Boolean (optional)
            indicated whether ISIs are based on experimental conditions (6 options) or self-chosen values
        timepts : int (optional)
            in the case of model_ISI = True, array containing the timepoints of start and end of first (and second) stimulus

        returns
        -----------------------
        stim : float
            adapted response

        """

        # create copy of input
        stim = np.zeros(len(input))

        # determine which scaling factor to use
        cat_idx = self.stim.index(cat)
        adapt = self.sf[cat_idx]

        # scale stimulus timecourse
        if 'onepulse' in trial:

            # import stimulus timepoints
            timepoints_onepulse = np.loadtxt(dir+'variables/timepoints_onepulse.txt', dtype=int)

            # define start and end of stimulus (expressed as timepts)
            start = timepoints_onepulse[cond-1, 0]
            end = timepoints_onepulse[cond-1, 1]

            # scale timecourse
            stim[start:end] = input[start:end] * adapt

        elif 'twopulse' in trial:

            # import stimulus timepoints
            timepoints_twopulse = np.loadtxt(dir+'variables/timepoints_twopulse.txt', dtype=int)

            # define start and end of stimulus (expressed as timepts)
            start1 = timepoints_twopulse[cond-1, 0]
            end1 = timepoints_twopulse[cond-1, 1]
            start2 = timepoints_twopulse[cond-1, 2]
            end2 = timepoints_twopulse[cond-1, 3]

            # scale timecourse
            stim[start1:end1] = input[start1:end1] * adapt
            stim[start2:end2] = input[start2:end2] * adapt

        elif model_ISI: # simulated ISIs used for model prediction for ISI recovery

            # define start and end of stimulus (expressed as timepts)
            start1 = int(timepts[0, 0])
            end1 = int(timepts[0, 1])
            start2 = int(timepts[1, 0])
            end2 = int(timepts[1, 1])

            # scale timecourse
            stim[start1:end1] = input[start1:end1] * adapt
            stim[start2:end2] = input[start2:end2] * adapt

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

    def norm(self, input, linrsp):
        """ Normalization of the input.

        params
        -----------------------
        input : float
            array containing values of input timecourse
        linrsp : float
            array containing values of linear response

        returns
        -----------------------
        rsp : float
            adapted response

        """

        # compute the normalized response
        demrsp = self.sigma**self.n + abs(linrsp)**self.n                       # semi-saturate + exponentiate
        normrsp = input/demrsp                                                  # divide

        # scale with gain
        rsp = self.scale * normrsp

        return rsp

    def norm_delay(self, input, linrsp, cat, denom=False):
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

        if denom:
            return rsp, demrsp
        else:
            return rsp
