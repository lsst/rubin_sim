import matplotlib.pyplot as plt
import numpy as np
import numpy.lib.recfunctions as rf
from scipy import interpolate


class Lims:
    """class to handle light curve of SN

    Parameters
    -------------
    Li_files : `str`
       light curve reference file
    mag_to_flux_files : `str`
       files of magnitude to flux
    band : `str`
        band considered
    SNR : `float`
        Signal-To-Noise Ratio cut
    mag_range : `float`, `float`, optional
        mag range considered
        Default : (23., 27.5)
    dt_range : `float`, `float`, optional
        difference time range considered (cadence)
        Default : (0.5, 25.)
    """

    def __init__(
        self,
        li_files,
        mag_to_flux_files,
        band,
        SNR,
        mag_range=(23.0, 27.5),
        dt_range=(0.5, 25.0),
    ):
        self.band = band
        self.SNR = SNR
        self.lims = []
        self.mag_to_flux = []
        self.mag_range = mag_range
        self.dt_range = dt_range

        for val in li_files:
            self.lims.append(self.get_lims(self.band, np.load(val), SNR))
        for val in mag_to_flux_files:
            self.mag_to_flux.append(np.load(val))
        self.interp()

    def get_lims(self, band, tab, SNR):
        """
        Estimations of the limits

        Parameters
        -------------
        band : `str`
          band to consider
        tab : `np.ndarray`, (N,)
          table of data
        SNR : `float`
           Signal-to-Noise Ratio cut

        Returns
        ---------
        lims : `dict`
            dict of limits with redshift and band as keys.

        """

        lims = {}

        for z in np.unique(tab["z"]):
            idx = (tab["z"] == z) & (tab["band"] == "LSST::" + band)
            idx &= tab["flux_e"] > 0.0
            sel = tab[idx]

            if len(sel) > 0:
                li2 = np.sqrt(np.sum(sel["flux_e"] ** 2))
                lim = 5.0 * li2 / SNR
                if z not in lims.keys():
                    lims[z] = {}
                lims[z][band] = lim

        return lims

    def mesh(self, mag_to_flux):
        """Mesh grid to estimate five-sigma depth values (m5) from mags input.

        Parameters
        -----------
        mag_to_flux : `np.ndarray`, (N,2)
            magnitude to flux values

        Returns
        -------
        m5, Dt, metric : `np.ndarray`, np.ndarray`, `np.ndarray`
            m5 values, time difference dt (cadence),
            metric=sqrt(dt)*f5 where f5 is the 5-sigma flux

        """
        dt = np.linspace(self.dt_range[0], self.dt_range[1], 100)
        m5 = np.linspace(self.mag_range[0], self.mag_range[1], 50)
        ida = mag_to_flux["band"] == self.band
        fa = interpolate.interp1d(mag_to_flux[ida]["m5"], mag_to_flux[ida]["flux_e"])
        f5 = fa(m5)
        f5, DT = np.meshgrid(f5, dt)
        m5, DT = np.meshgrid(m5, dt)
        metric = np.sqrt(DT) * f5

        return m5, DT, metric

    def interp(self):
        """Estimate a grid of interpolated values in the plane
        (m5, cadence, metric).
        """

        m5_all = []
        dt_all = []
        metric_all = []

        for val in self.mag_to_flux:
            m5, DT, metric = self.mesh(val)
            m5_all.append(m5)
            dt_all.append(DT)
            metric_all.append(metric)

        sorted_keys = []
        for i in range(len(self.lims)):
            sorted_keys.append(np.sort([k for k in self.lims[i].keys()])[::-1])
        figa, axa = plt.subplots()

        for kk, lim in enumerate(self.lims):
            ll = [lim[zz][self.band] for zz in sorted_keys[kk]]
            cs = axa.contour(m5_all[kk], dt_all[kk], metric_all[kk], ll)

            points_values = None
            for io, col in enumerate(cs.collections):
                # Update in matplotlib changed get_segments to get_paths
                if hasattr(col, "get_segments"):
                    segments = col.get_segments()
                else:
                    segments = col.get_paths()
                if segments:
                    segments = segments[0]
                    if hasattr(segments, "vertices"):
                        segments = segments.vertices
                    myarray = segments
                    res = np.array(myarray[:, 0], dtype=[("m5", "f8")])
                    res = rf.append_fields(res, "cadence", myarray[:, 1])
                    res = rf.append_fields(res, "z", [sorted_keys[kk][io]] * len(res))
                    if points_values is None:
                        points_values = res
                    else:
                        points_values = np.concatenate((points_values, res))
            self.points_ref = points_values

        plt.close(figa)  # do not display

    def interp_griddata(self, data):
        """Estimate metric interpolation for data (m5,cadence)

        Parameters
        ----------
        data : `np.ndarray`
            data where interpolation has to be done (m5,cadence)

        Returns
        --------
        res : `np.ndarray`
            griddata interpolation (m5,cadence,metric)
        """

        ref_points = self.points_ref
        res = interpolate.griddata(
            (ref_points["m5"], ref_points["cadence"]),
            ref_points["z"],
            (data["m5_mean"], data["cadence_mean"]),
            method="cubic",
        )
        return res


class GenerateFakeObservations:
    """Class to generate Fake observations

    Parameters
    -----------
    config: yaml-like
       configuration file (parameter choice: filter, cadence, m5,Nseasons, ..)
    list : `str`, optional
        Name of the columns used.
        Default : 'observationStartMJD', 'fieldRA', 'fieldDec','filter',
        'fiveSigmaDepth', 'visitExposureTime','numExposures',
        'visitTime','season'

    Returns
    ---------
    observations : `np.ndarray`, (N, M)
        recordarray of observations with the fields:
        MJD, Ra, Dec, band,m5,Nexp, ExpTime, Season
    """

    def __init__(
        self,
        config,
        mjd_col="observationStartMJD",
        ra_col="fieldRA",
        dec_col="fieldDec",
        filter_col="filter",
        m5_col="fiveSigmaDepth",
        exptime_col="visitExposureTime",
        nexp_col="numExposures",
        season_col="season",
    ):
        self.mjd_col = mjd_col
        self.m5_col = m5_col
        self.filter_col = filter_col
        self.ra_col = ra_col
        self.dec_col = dec_col
        self.exptime_col = exptime_col
        self.season_col = season_col
        self.nexp_col = nexp_col

        # now make fake obs
        self.make_fake(config)

    def make_fake(self, config):
        """Generate Fake observations

        Parameters
        -----------
        config: yaml-like
            configuration file (parameter choice: filter,
            cadence, m5,Nseasons, ...)
        """
        bands = config["bands"]
        cadence = dict(zip(bands, config["Cadence"]))
        shift_days = dict(zip(bands, [config["shift_days"] * io for io in range(len(bands))]))
        m5 = dict(zip(bands, config["m5"]))
        nvisits = dict(zip(bands, config["nvisits"]))
        exposure__time = dict(zip(bands, config["exposure__time"]))

        ra = config["ra"]
        dec = config["dec"]
        rtot = []
        # for season in range(1, config['nseasons']+1):
        for il, season in enumerate(config["seasons"]):
            # mjd_min = config['MJD_min'] + float(season-1)*inter_season_gap
            mjd_min = config["MJD_min"][il]
            mjd_max = mjd_min + config["season_length"]

            for i, band in enumerate(bands):
                mjd = np.arange(mjd_min, mjd_max + cadence[band], cadence[band])
                mjd += shift_days[band]
                m5_coadded = self.m5_coadd(m5[band], nvisits[band], exposure__time[band])
                myarr = np.array(mjd, dtype=[(self.mjd_col, "f8")])
                myarr = rf.append_fields(
                    myarr,
                    [self.ra_col, self.dec_col, self.filter_col],
                    [[ra] * len(myarr), [dec] * len(myarr), [band] * len(myarr)],
                )
                myarr = rf.append_fields(
                    myarr,
                    [self.m5_col, self.nexp_col, self.exptime_col, self.season_col],
                    [
                        [m5_coadded] * len(myarr),
                        [nvisits[band]] * len(myarr),
                        [nvisits[band] * exposure__time[band]] * len(myarr),
                        [season] * len(myarr),
                    ],
                )
                rtot.append(myarr)

        res = np.copy(np.concatenate(rtot))
        res.sort(order=self.mjd_col)

        self.observations = res

    def m5_coadd(self, m5, nvisits, tvisit):
        """Coadded m5 estimation

        Parameters
        ----------
        m5 : `list` [`float`]
           list of five-sigma depth values
        nvisits : `list` [`float`]
            list of the number of visits
        tvisit : `list` [`float`]
           list of the visit times

        Returns
        ---------
        m5_coadd : `list` [`float`]
            list of m5 coadded values
        """
        m5_coadd = m5 + 1.25 * np.log10(float(nvisits) * tvisit / 30.0)
        return m5_coadd


class ReferenceData:
    """class to handle light curve of SN

    Parameters
    ------------
    Li_files : `str`
        light curve reference file
    mag_to_flux_files : `str`
        files of magnitude to flux
    band : `str`
        band considered
    z : `float`
        redshift considered
    """

    def __init__(self, li_files, mag_to_flux_files, band, z):
        self.band = band
        self.z = z
        self.fluxes = []
        self.mag_to_flux = []

        for val in li_files:
            self.fluxes.append(self.interp_fluxes(self.band, np.load(val), self.z))
        for val in mag_to_flux_files:
            self.mag_to_flux.append(self.interp_mag(self.band, np.load(val)))

    def interp_fluxes(self, band, tab, z):
        """Flux interpolator

        Parameters
        ---------------
        band : `str`
            band considered
        tab : `np.ndarray`
            reference data with (at least) fields z,band,time,DayMax
        z : `float`
            redshift considered

        Returns
        --------
        fluxes : `list` [`float`]
            list (float) of interpolated fluxes (in e/sec)
        """
        idx = (np.abs(tab["z"] - z) < 1.0e-5) & (tab["band"] == "LSST::" + band)
        sel = tab[idx]
        selc = np.copy(sel)
        difftime = sel["time"] - sel["DayMax"]
        selc = rf.append_fields(selc, "deltaT", difftime)
        return interpolate.interp1d(selc["deltaT"], selc["flux_e"], bounds_error=False, fill_value=0.0)

    def interp_mag(self, band, tab):
        """magnitude (m5) to flux (e/sec) interpolator

        Parameters
        ---------------
        band : `str`
            band considered
        tab : `np.ndarray`
            reference data with (at least) fields band,m5,flux_e,
        z : `float`
            redshift considered

        Returns
        --------
        mags : `list` [`float`]
            list (float) of interpolated magnitudes (in e/sec)
        """
        idx = tab["band"] == band
        sel = tab[idx]
        return interpolate.interp1d(sel["m5"], sel["flux_e"], bounds_error=False, fill_value=0.0)
