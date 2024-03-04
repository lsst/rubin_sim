# Metric for kilonova detectability based on GW170817 SED used in Scolnic et
# al. 2018 and Setzer et al. 2019. The chosen detection criteria are related
# to those used in the LSST DESC white paper detectability work and the two
# references above.
#
# Contact for this code:
# christian.setzer@fysik.su.se
from pathlib import Path

from .transient_ascii_sed_metric import TransientAsciiSEDMetric

__all__ = ("GW170817DetMetric",)
base_path = Path(__file__).parent


class GW170817DetMetric(TransientAsciiSEDMetric):
    """
    Wrapper metric class for GW170817-like kilonovae based on the
    TransientAsciiSEDMetric. Defaults are set to those corresponding to similar
    detection criteria used in Scolnic et al. 2018 and Setzer et al. 2019.
    However, due to the simplified nature of transient distribution for
    computing this metric, the criteria have been altered to only include
    criteria two and three. The chosen redshift is at the approximate mean
    redshift of the detected cosmological redshift distribution shown in
    Setzer et al. 2019.

    Parameters
    -----------
    ascii_file : `str`, optional
        The ascii file containing the inputs for the SED. The file must
        contain three columns - ['phase', 'wave', 'flux'] -
        of phase/epoch (in days), wavelength (Angstroms), and
        flux (ergs/s/Angstrom). Default, data provided with sims_maf_contrib.
    metric_name : `str`, optional
        Name of the metric, can be overwritten by user or child metric.
    z : `float`, optional
        Cosmological redshift at which to consider observations of the
        tranisent SED. Default 0.08.
    num_filters : `int`, optional
        Number of filters that need to be observed for an object to be
        counted as detected. Default 2. (if num_per_lightcurve is 0, then
        this will be reset to 0).
    filter_time : `float`, optional
        The time within which observations in at least num_filters are
        required (in days). Default 25.0 days.
    num_phases_to_run : `int`, optional
        Sets the number of phases that should be checked.
        One can imagine pathological cadences where many objects pass the
        detection criteria, but would not if the observations were offset
        by a phase-shift. Default 5.
    """

    def __init__(
        self,
        ascii_file=(base_path / "../data/DECAMGemini_SED.txt").resolve(),
        metric_name="GW170817DetMetric",
        z=0.08,
        num_filters=2,
        filter_time=25.0,
        num_phases_to_run=5,
        **kwargs,
    ):
        """"""
        super(GW170817DetMetric, self).__init__(
            ascii_file=ascii_file,
            metric_name=metric_name,
            z=z,
            num_filters=num_filters,
            filter_time=filter_time,
            num_phases_to_run=num_phases_to_run,
            **kwargs,
        )
