# Variability Depth Metric
# Keaton Bell (keatonb@astro.as.utexas.edu)

__all__ = ("VarDepth",)

import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.stats import chi2

from rubin_sim.maf.metrics import BaseMetric


class VarDepth(BaseMetric):
    """Calculate the survey depth that a variable star can be
    reliably identified.

    Parameters
    ----------
    completeness : `float`, opt
        Fractional desired completeness of recovered variable sample.
    contamination : `float`, opt
        Fractional allowed incompleteness of recovered nonvariables.
    numruns : `int`, opt
        Number of simulated realizations of noise.
        Most computationally expensive part of metric.
    signal : `float`, opt
        Sqrt total pulsational power meant to be recovered.
    magres : `float`, opt
        desired resolution of variability depth result.
    """

    def __init__(
        self,
        m5_col="fiveSigmaDepth",
        metric_name="variability depth",
        completeness=0.95,
        contamination=0.05,
        numruns=10000,
        signal=0.01,
        magres=0.01,
        **kwargs,
    ):
        self.m5col = m5_col
        self.completeness = completeness
        self.contamination = contamination
        self.numruns = numruns
        self.signal = signal
        self.magres = magres
        super(VarDepth, self).__init__(col=m5_col, metric_name=metric_name, **kwargs)

    def run(self, data_slice, slice_point=None):
        # Get the visit information
        m5 = data_slice[self.m5col]
        # Number of visits
        N = len(m5)

        # magnitudes to be sampled
        mag = np.arange(16, np.mean(m5), 0.5)
        # hold the distance between the completeness and contamination goals.
        res = np.zeros(mag.shape)
        # make them nans for now
        res[:] = np.nan

        # hold the measured noise-only variances
        noiseonlyvar = np.zeros(self.numruns)

        # Calculate the variance at a reference magnitude and scale from that
        m0 = 20.0
        sigmaref = 0.2 * (10.0 ** (-0.2 * m5)) * (10.0 ** (0.2 * m0))

        # run the simulations
        # Simulate the measured noise-only variances at a reference magnitude
        for i in np.arange(self.numruns):
            # random realization of the Gaussian error distributions
            scatter = np.random.randn(N) * sigmaref
            noiseonlyvar[i] = np.var(scatter)  # store the noise-only variance

        # Since we are treating the underlying signal being representable by a
        # fixed-width gaussian, its variance pdf is a Chi-squared distribution
        # with the degrees of freedom=visits. Since variances add, the variance
        # pdfs convolve. The cumulative distribution function of the sum of two
        # random deviates is the convolution of one pdf with a cdf.

        # We'll consider the cdf of the noise-only variances
        # because it's easier to interpolate
        noisesorted = np.sort(noiseonlyvar)
        # linear interpolation
        interpnoisecdf = UnivariateSpline(
            noisesorted, np.arange(self.numruns) / float(self.numruns), k=1, s=0
        )

        # We need a binned, signal-only variance probability
        # distribution function for numerical convolution
        numsignalsamples = 100
        xsig = np.linspace(chi2.ppf(0.001, N), chi2.ppf(0.999, N), numsignalsamples)
        signalpdf = chi2.pdf(xsig, N)
        # correct x to the proper variance scale
        xsig = (self.signal**2.0) * xsig / N
        pdfstepsize = xsig[1] - xsig[0]
        # Since everything is going to use this stepsize down the line,
        # normalize so the pdf integrates to 1 when summed
        # (no factor of stepsize needed)
        signalpdf /= np.sum(signalpdf)

        # run through the sample magnitudes, calculate distance between cont
        # and comp thresholds.
        # run until solution found.
        solutionfound = False

        for i, mref in enumerate(mag):
            # i counts and mref is the currently sampled magnitude
            # Scale factor from m0
            scalefact = 10.0 ** (0.4 * (mref - m0))

            # Calculate the desired contamination threshold
            contthresh = np.percentile(noiseonlyvar, 100.0 - 100.0 * self.contamination) * scalefact

            # Realize the noise CDF at the required stepsize
            xnoise = np.arange(noisesorted[0] * scalefact, noisesorted[-1] * scalefact, pdfstepsize)

            # Only do calculation if near the solution:
            if (len(xnoise) > numsignalsamples / 10) and (not solutionfound):
                noisecdf = interpnoisecdf(xnoise / scalefact)
                # turn into a noise pdf
                noisepdf = noisecdf[1:] - noisecdf[:-1]
                noisepdf /= np.sum(noisepdf)
                # from cdf to pdf conversion
                xnoise = (xnoise[1:] + xnoise[:-1]) / 2.0

                # calculate and plot the convolution =
                # signal+noise variance dist.
                convolution = 0
                if len(noisepdf) > len(signalpdf):
                    convolution = np.convolve(noisepdf, signalpdf)
                else:
                    convolution = np.convolve(signalpdf, noisepdf)
                xconvolved = xsig[0] + xnoise[0] + np.arange(len(convolution)) * pdfstepsize

                # calculate the completeness threshold
                combinedcdf = np.cumsum(convolution)
                findcompthresh = UnivariateSpline(combinedcdf, xconvolved, k=1, s=0)
                compthresh = findcompthresh(1.0 - self.completeness)

                res[i] = compthresh - contthresh
                if res[i] < 0:
                    solutionfound = True

        # interpolate for where the thresholds coincide
        # print res
        if np.sum(np.isfinite(res)) > 1:
            f1 = UnivariateSpline(mag[np.isfinite(res)], res[np.isfinite(res)], k=1, s=0)
            # sample the magnitude range at given resolution
            magsamples = np.arange(16, np.mean(m5), self.magres)
            vardepth = magsamples[np.argmin(np.abs(f1(magsamples)))]
            return vardepth
        else:
            return min(mag) - 1
