import numpy as np

from utils import gammaPDF, exponential_decay

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
    stim : array dim(1, T)
        stimulus time course
    sample_rate : int
        frequency with which the timepoints are measured
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

    def __init__(self, stim, sample_rate, tau, shift, scale, n, sigma, tau_a):

        # assign class variables
        self.tau = tau
        self.shift = shift
        self.scale = scale
        self.n = n
        self.sigma = sigma
        self.tau_a = tau_a

        # iniate temporal variables
        self.numtimepts = len(stim)
        self.srate = sample_rate

        # compute timepoints
        self.t = np.arange(0, self.numtimepts)/self.srate

        # compute the impulse response function (used in the nominator, convolution of the stimulus)
        self.irf = gammaPDF(self.t, self.tau, 2)

        # create exponential decay filter (for the normalization, convolution of the linear response)
        self.norm_irf = exponential_decay(self.t, self.tau_a)

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
        # sft = np.round(self.shift/(1/self.srate))
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

    def norm_delay(self, input, linrsp, denom=False):
        """ Introduces delay in input

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

        # compute the normalized delayed response
        poolrsp = np.convolve(linrsp, self.norm_irf, 'full')
        poolrsp = poolrsp[0:self.numtimepts]
        demrsp = self.sigma**self.n + abs(poolrsp)**self.n                      # semi-saturate + exponentiate
        normrsp = input/demrsp                                                  # divide

        # scale with gain
        rsp = self.scale * normrsp

        if denom:
            return rsp, demrsp
        else:
            return rsp
