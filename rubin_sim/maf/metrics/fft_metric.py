__all__ = ("FftMetric",)

from scipy import fftpack

from .base_metric import BaseMetric


class FftMetric(BaseMetric):
    """Calculate a truncated FFT of the exposure times."""

    def __init__(self, times_col="expmjd", metric_name="Fft", n_coeffs=100, **kwargs):
        """Instantiate metric.

        'times_col' = column with the time of the visit (default expmjd),
        'n_coeffs' = number of coefficients of the (real) FFT to keep."""
        self.times = times_col
        super(FftMetric, self).__init__(col=[self.times], metric_name=metric_name, **kwargs)
        # Set up length of return values.
        self.n_coeffs = n_coeffs
        return

    def run(self, data_slice, slice_point=None):
        fft = fftpack.rfft(data_slice[self.times])
        return fft[0 : self.n_coeffs]

    def reduce_peak(self, fft_coeff):
        pass
