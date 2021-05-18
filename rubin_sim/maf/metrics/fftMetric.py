from scipy import fftpack
from .baseMetric import BaseMetric

__all__ = ['FftMetric']

class FftMetric(BaseMetric):
    """Calculate a truncated FFT of the exposure times."""
    def __init__(self, timesCol='expmjd', metricName='Fft',
                 nCoeffs=100, **kwargs):
        """Instantiate metric.

        'timesCol' = column with the time of the visit (default expmjd),
        'nCoeffs' = number of coefficients of the (real) FFT to keep."""
        self.times = timesCol
        super(FftMetric, self).__init__(col=[self.times], metricName=metricName, **kwargs)
        # Set up length of return values.
        self.nCoeffs = nCoeffs
        return

    def run(self, dataSlice, slicePoint=None):
        fft = fftpack.rfft(dataSlice[self.times])
        return fft[0:self.nCoeffs]

    def reducePeak(self, fftCoeff):
        pass
