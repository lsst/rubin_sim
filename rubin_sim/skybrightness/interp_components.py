__all__ = (
    "id2intid",
    "intid2id",
    "load_spec_files",
    "BaseSingleInterp",
    "ScatteredStar",
    "LowerAtm",
    "UpperAtm",
    "MergedSpec",
    "Airglow",
    "TwilightInterp",
    "MoonInterp",
    "ZodiacalInterp",
)

import glob
import os
import warnings

import healpy as hp
import numpy as np
from rubin_scheduler.data import get_data_dir
from scipy.interpolate import InterpolatedUnivariateSpline, interp1d

from rubin_sim.phot_utils import Bandpass, Sed

from .twilight_func import twilight_func

# Make backwards compatible with healpy
if hasattr(hp, "get_interp_weights"):
    get_neighbours = hp.get_interp_weights
elif hasattr(hp, "get_neighbours"):
    get_neighbours = hp.get_neighbours
else:
    print("Could not find appropriate healpy function for get_interp_weight or get_neighbours")


def id2intid(ids):
    """Take an array of ids, and convert them to an integer id.
    Handy if you want to put things into a sparse array.
    """
    uids = np.unique(ids)
    order = np.argsort(ids)
    oids = ids[order]
    uintids = np.arange(np.size(uids), dtype=int)
    left = np.searchsorted(oids, uids)
    right = np.searchsorted(oids, uids, side="right")
    intids = np.empty(ids.size, dtype=int)
    for i in range(np.size(left)):
        intids[left[i] : right[i]] = uintids[i]
    result = intids * 0
    result[order] = intids
    return result, uids, uintids


def intid2id(intids, uintids, uids, dtype=int):
    """convert an int back to an id"""
    ids = np.zeros(np.size(intids))

    order = np.argsort(intids)
    ointids = intids[order]
    left = np.searchsorted(ointids, uintids, side="left")
    right = np.searchsorted(ointids, uintids, side="right")
    for i, (le, ri) in enumerate(zip(left, right)):
        ids[le:ri] = uids[i]
    result = np.zeros(np.size(intids), dtype=dtype)
    result[order] = ids

    return result


def load_spec_files(filenames, mags=False):
    """Load up the ESO spectra.

    The ESO npz files contain the following arrays:
    * filter_wave: The central wavelengths of the pre-computed magnitudes
    * wave: wavelengths for the spectra
    * spec: array of spectra and magnitudes along with the relevant
    variable inputs.  For example, airglow has dtype =
    [('airmass', '<f8'),
    ('solar_flux', '<f8'),
    ('spectra','<f8', (17001,)),
    ('mags', '<f8', (6,)]

    For each unique airmass and solar_flux value, there is a 17001 elements
    spectra and 6 magnitudes.
    """

    if len(filenames) == 1:
        temp = np.load(filenames[0])
        wave = temp["wave"].copy()
        # Note old camelCase here. Might need to update if sav
        # files regenerated
        filter_wave = temp["filterWave"].copy()
        if mags:
            # don't copy the spectra to save memory space
            dt = np.dtype(
                [
                    (key, temp["spec"].dtype[i])
                    for i, key in enumerate(temp["spec"].dtype.names)
                    if key != "spectra"
                ]
            )
            spec = np.zeros(temp["spec"].size, dtype=dt)
            for key in temp["spec"].dtype.names:
                if key != "spectra":
                    spec[key] = temp["spec"][key].copy()
        else:
            spec = temp["spec"].copy()
    else:
        temp = np.load(filenames[0])
        wave = temp["wave"].copy()
        filter_wave = temp["filterWave"].copy()
        if mags:
            # don't copy the spectra to save memory space
            dt = np.dtype(
                [
                    (key, temp["spec"].dtype[i])
                    for i, key in enumerate(temp["spec"].dtype.names)
                    if key != "spectra"
                ]
            )
            spec = np.zeros(temp["spec"].size, dtype=dt)
            for key in temp["spec"].dtype.names:
                if key != "spectra":
                    spec[key] = temp["spec"][key].copy()
        else:
            spec = temp["spec"].copy()
        for filename in filenames[1:]:
            temp = np.load(filename)
            if mags:
                # don't copy the spectra to save memory space
                dt = np.dtype(
                    [
                        (key, temp["spec"].dtype[i])
                        for i, key in enumerate(temp["spec"].dtype.names)
                        if key != "spectra"
                    ]
                )
                tempspec = np.zeros(temp["spec"].size, dtype=dt)
                for key in temp["spec"].dtype.names:
                    if key != "spectra":
                        tempspec[key] = temp["spec"][key].copy()
            else:
                tempspec = temp["spec"]
            spec = np.append(spec, tempspec)
    return spec, wave, filter_wave


class BaseSingleInterp:
    """Base class for interpolating sky components which only depend
    on airmass.

    Parameters
    ----------
    comp_name : `str`, optional
        Component name.
    sorted_order : `list` [`str`], optional
        Order of the dimensions in the input .npz files.
    mags : `bool`, optional
        Return magnitudes (only) rather than the full spectrum.
    """

    def __init__(self, comp_name=None, sorted_order=["airmass", "nightTimes"], mags=False):
        self.mags = mags

        data_dir = os.path.join(get_data_dir(), "skybrightness", "ESO_Spectra/" + comp_name)

        filenames = sorted(glob.glob(data_dir + "/*.npz"))
        self.spec, self.wave, self.filter_wave = load_spec_files(filenames, mags=self.mags)

        # Take the log of the spectra in case we want to interp in log space.
        if not mags:
            self.log_spec = np.zeros(self.spec["spectra"].shape, dtype=float)
            good = np.where(self.spec["spectra"] != 0)
            self.log_spec[good] = np.log10(self.spec["spectra"][good])
            self.spec_size = self.spec["spectra"][0].size
        else:
            self.spec_size = 0

        # What order are the dimensions sorted by (from how
        # the .npz was packaged)
        self.sorted_order = sorted_order
        self.dim_dict = {}
        self.dim_sizes = {}
        for dt in self.sorted_order:
            self.dim_dict[dt] = np.unique(self.spec[dt])
            self.dim_sizes[dt] = np.size(np.unique(self.spec[dt]))

        # Set up and save the dict to order the filters once.
        self.filter_name_dict = {"u": 0, "g": 1, "r": 2, "i": 3, "z": 4, "y": 5}

    def __call__(self, interp_points, filter_names=["u", "g", "r", "i", "z", "y"]):
        """At `interp_points (e.g. airmass), return values."""
        if self.mags:
            return self.interp_mag(interp_points, filter_names=filter_names)
        else:
            return self.interp_spec(interp_points)

    def _indx_and_weights(self, points, grid):
        """For given 1-D points, find the grid points on
        either side and return the weights assume grid is sorted.

        Parameters
        ----------
        points : `np.ndarray`, (N,)
            The points on the grid to query.
        grid : `np.ndarray`, (N,)
            The grid on which to locate `points`.

        Returns
        -------
        indx_r, indx_l : `np.ndarray`, `np.ndarray`
            The grid indexes for each of the 1-d points
        w_r, w_l : `np.ndarray`, `np.ndarray`
            The weights for each of these grid points.
        """

        order = np.argsort(points)

        indx_r = np.empty(points.size, dtype=int)

        indx_r[order] = np.searchsorted(grid, points[order])
        indx_l = indx_r - 1

        # If points off the grid were requested, just use the edge grid point
        off_grid = np.where(indx_r == grid.size)
        indx_r[off_grid] = grid.size - 1
        full_range = grid[indx_r] - grid[indx_l]

        w_l = np.zeros(full_range.size, dtype=float)
        w_r = np.ones(full_range.size, dtype=float)

        good = np.where(full_range != 0)
        w_l[good] = (grid[indx_r][good] - points[good]) / full_range[good]
        w_r[good] = (points[good] - grid[indx_l[good]]) / full_range[good]

        return indx_r, indx_l, w_r, w_l

    def _weighting(self, interp_points, values):
        """
        given a list/array of airmass values, return a
        dict with the interpolated spectrum at each airmass
        and the wavelength array.

        Input interp_points should be sorted
        """
        results = np.zeros((interp_points.size, np.size(values[0])), dtype=float)

        in_range = np.where(
            (interp_points["airmass"] <= np.max(self.dim_dict["airmass"]))
            & (interp_points["airmass"] >= np.min(self.dim_dict["airmass"]))
        )
        indx_r, indx_l, w_r, w_l = self._indx_and_weights(
            interp_points["airmass"][in_range], self.dim_dict["airmass"]
        )

        nextra = 1

        # XXX--should I use the log spectra?
        # Make a check and switch back and forth?
        results[in_range] = (
            w_r[:, np.newaxis] * values[indx_r * nextra] + w_l[:, np.newaxis] * values[indx_l * nextra]
        )

        return results

    def interp_spec(self, interp_points):
        result = self._weighting(interp_points, self.log_spec)
        mask = np.where(result == 0.0)
        result = 10.0**result
        result[mask] = 0.0
        return {"spec": result, "wave": self.wave}

    def interp_mag(self, interp_points, filter_names=["u", "g", "r", "i", "z", "y"]):
        filterindx = [self.filter_name_dict[key] for key in filter_names]
        result = self._weighting(interp_points, self.spec["mags"][:, filterindx])
        mask = np.where(result == 0.0)
        result = 10.0 ** (-0.4 * (result - np.log10(3631.0)))
        result[mask] = 0.0
        return {"spec": result, "wave": self.filter_wave}


class ScatteredStar(BaseSingleInterp):
    """Interpolate the spectra caused by scattered starlight."""

    def __init__(self, comp_name="ScatteredStarLight", mags=False):
        super(ScatteredStar, self).__init__(comp_name=comp_name, mags=mags)


class LowerAtm(BaseSingleInterp):
    """Interpolate the spectra caused by the lower atmosphere."""

    def __init__(self, comp_name="LowerAtm", mags=False):
        super(LowerAtm, self).__init__(comp_name=comp_name, mags=mags)


class UpperAtm(BaseSingleInterp):
    """Interpolate the spectra caused by the upper atmosphere."""

    def __init__(self, comp_name="UpperAtm", mags=False):
        super(UpperAtm, self).__init__(comp_name=comp_name, mags=mags)


class MergedSpec(BaseSingleInterp):
    """Interpolate the combined spectra caused by the sum of the scattered
    starlight, air glow, upper and lower atmosphere.
    """

    def __init__(self, comp_name="MergedSpec", mags=False):
        super(MergedSpec, self).__init__(comp_name=comp_name, mags=mags)


class Airglow(BaseSingleInterp):
    """Interpolate the spectra caused by airglow."""

    def __init__(self, comp_name="Airglow", sorted_order=["airmass", "solarFlux"], mags=False):
        super(Airglow, self).__init__(comp_name=comp_name, mags=mags, sorted_order=sorted_order)
        self.n_solar_flux = np.size(self.dim_dict["solarFlux"])

    def _weighting(self, interp_points, values):
        results = np.zeros((interp_points.size, np.size(values[0])), dtype=float)
        # Only interpolate point that lie in the model grid
        in_range = np.where(
            (interp_points["airmass"] <= np.max(self.dim_dict["airmass"]))
            & (interp_points["airmass"] >= np.min(self.dim_dict["airmass"]))
            & (interp_points["solar_flux"] >= np.min(self.dim_dict["solarFlux"]))
            & (interp_points["solar_flux"] <= np.max(self.dim_dict["solarFlux"]))
        )
        use_points = interp_points[in_range]
        am_right_index, am_left_index, am_right_w, am_left_w = self._indx_and_weights(
            use_points["airmass"], self.dim_dict["airmass"]
        )

        sf_right_index, sf_left_index, sf_right_w, sf_left_w = self._indx_and_weights(
            use_points["solar_flux"], self.dim_dict["solarFlux"]
        )

        for am_index, amW in zip([am_right_index, am_left_index], [am_right_w, am_left_w]):
            for sf_index, sfW in zip([sf_right_index, sf_left_index], [sf_right_w, sf_left_w]):
                results[in_range] += (
                    amW[:, np.newaxis] * sfW[:, np.newaxis] * values[am_index * self.n_solar_flux + sf_index]
                )
        return results


class TwilightInterp:
    """Use the Solar Spectrum to provide an interpolated spectra or magnitudes
    for the twilight sky.

    Parameters
    ----------
    mags : `bool`
        If True, only return the LSST filter magnitudes,
        otherwise return the full spectrum
    dark_sky_mags : `dict`
        Dict of the zenith dark sky values to be assumed.
        The twilight fits are done relative to the dark sky level.
    fit_results : `dict`
        Dict of twilight parameters based on twilight_func.
        Keys should be filter names.
    """

    def __init__(self, mags=False, dark_sky_mags=None, fit_results=None):
        if dark_sky_mags is None:
            dark_sky_mags = {
                "u": 22.8,
                "g": 22.3,
                "r": 21.2,
                "i": 20.3,
                "z": 19.3,
                "y": 18.0,
                "B": 22.35,
                "G": 21.71,
                "R": 21.3,
            }

        self.mags = mags

        data_dir = os.path.join(get_data_dir(), "skybrightness")

        solar_saved = np.load(os.path.join(data_dir, "solarSpec/solarSpec.npz"))
        self.solar_spec = Sed(wavelen=solar_saved["wave"], flambda=solar_saved["spec"])
        solar_saved.close()

        canon_filters = {}
        fnames = ["blue_canon.csv", "green_canon.csv", "red_canon.csv"]

        # Filter names, from bluest to reddest.
        self.filter_names = ["B", "G", "R"]

        # Supress warning that Canon filters are low sampling
        warnings.filterwarnings("ignore", message="There is an area of")
        warnings.filterwarnings("ignore", message="Wavelength sampling of")
        for fname, filter_name in zip(fnames, self.filter_names):
            bpdata = np.genfromtxt(
                os.path.join(data_dir, "Canon/", fname),
                delimiter=", ",
                dtype=list(zip(["wave", "through"], [float] * 2)),
            )
            bp_temp = Bandpass()
            bp_temp.set_bandpass(bpdata["wave"], bpdata["through"])
            bp_temp.resample_bandpass(
                wavelen_min=self.solar_spec.wavelen.min(),
                wavelen_max=self.solar_spec.wavelen.max(),
                wavelen_step=self.solar_spec.wavelen[1] - self.solar_spec.wavelen[0],
            )
            # Force wavelengths to be identical so
            # it doesn't try to resample again later
            bp_temp.wavelen = self.solar_spec.wavelen
            canon_filters[filter_name] = bp_temp

        # Tack on the LSST filters
        through_path = os.path.join(get_data_dir(), "throughputs", "baseline")
        lsst_keys = ["u", "g", "r", "i", "z", "y"]
        for key in lsst_keys:
            bp = np.loadtxt(
                os.path.join(through_path, "total_" + key + ".dat"),
                dtype=list(zip(["wave", "trans"], [float] * 2)),
            )
            temp_b = Bandpass()
            temp_b.set_bandpass(bp["wave"], bp["trans"])
            canon_filters[key] = temp_b
            self.filter_names.append(key)

        # MAGIC NUMBERS from fitting the all-sky camera:
        # Code to generate values in
        # sims_skybrightness/examples/fitTwiSlopesSimul.py
        # Which in turn uses twilight maps from
        # sims_skybrightness/examples/buildTwilMaps.py
        # values are of the form:
        # 0: ratio of f^z_12 to f_dark^z
        # 1: slope of curve wrt sun alt
        # 2: airmass term (10^(arg[2]*(X-1)))
        # 3: azimuth term.
        # 4: zenith dark sky flux (erg/s/cm^2)

        # For z and y, just assuming the shape parameter
        # fits are similar to the other bands.
        # Looks like the diode is not sensitive enough to detect faint sky.
        # Using the Patat et al 2006 I-band values for z and
        # modeified a little for y as a temp fix.
        if fit_results is None:
            self.fit_results = {
                "B": [
                    7.56765633e00,
                    2.29798055e01,
                    2.86879956e-01,
                    3.01162143e-01,
                    2.58462036e-04,
                ],
                "G": [
                    2.38561156e00,
                    2.29310648e01,
                    2.97733083e-01,
                    3.16403197e-01,
                    7.29660095e-04,
                ],
                "R": [
                    1.75498017e00,
                    2.22011802e01,
                    2.98619033e-01,
                    3.28880254e-01,
                    3.24411056e-04,
                ],
                "z": [2.29, 24.08, 0.3, 0.3, -666],
                "y": [2.0, 24.08, 0.3, 0.3, -666],
            }

            # XXX-completely arbitrary fudge factor to make things
            # brighter in the blue
            # Just copy the blue and say it's brighter.
            self.fit_results["u"] = [
                16.0,
                2.29622121e01,
                2.85862729e-01,
                2.99902574e-01,
                2.32325117e-04,
            ]
        else:
            self.fit_results = fit_results

        # Take out any filters that don't have fit results
        self.filter_names = [key for key in self.filter_names if key in self.fit_results]

        self.eff_wave = []
        self.solar_mag = []
        for filter_name in self.filter_names:
            self.eff_wave.append(canon_filters[filter_name].calc_eff_wavelen()[0])
            self.solar_mag.append(self.solar_spec.calc_mag(canon_filters[filter_name]))

        order = np.argsort(self.eff_wave)
        self.filter_names = np.array(self.filter_names)[order]
        self.eff_wave = np.array(self.eff_wave)[order]
        self.solar_mag = np.array(self.solar_mag)[order]

        # update the fit results to be zeropointed properly
        for key in self.fit_results:
            f0 = 10.0 ** (-0.4 * (dark_sky_mags[key] - np.log10(3631.0)))
            self.fit_results[key][-1] = f0

        self.solar_wave = self.solar_spec.wavelen
        self.solar_flux = self.solar_spec.flambda
        # This one isn't as bad as the model grids, maybe we could get
        # away with computing the magnitudes in the __call__ each time.
        if mags:
            # Load up the LSST filters and convert the
            # solarSpec.flambda and solarSpec.wavelen to fluxes
            self.lsst_filter_names = ["u", "g", "r", "i", "z", "y"]
            self.lsst_equations = np.zeros(
                (np.size(self.lsst_filter_names), np.size(self.fit_results["B"])),
                dtype=float,
            )
            self.lsst_eff_wave = []

            fits = np.empty((np.size(self.eff_wave), np.size(self.fit_results["B"])), dtype=float)
            for i, fn in enumerate(self.filter_names):
                fits[i, :] = self.fit_results[fn]

            through_path = os.path.join(get_data_dir(), "throughputs", "baseline")
            for filtername in self.lsst_filter_names:
                bp = np.loadtxt(
                    os.path.join(through_path, "total_" + filtername + ".dat"),
                    dtype=list(zip(["wave", "trans"], [float] * 2)),
                )
                temp_b = Bandpass()
                temp_b.set_bandpass(bp["wave"], bp["trans"])
                self.lsst_eff_wave.append(temp_b.calc_eff_wavelen()[0])
            # Loop through the parameters and interpolate to new
            # eff wavelengths
            for i in np.arange(self.lsst_equations[0, :].size):
                interp = InterpolatedUnivariateSpline(self.eff_wave, fits[:, i])
                self.lsst_equations[:, i] = interp(self.lsst_eff_wave)
            # Set the dark sky flux
            for i, filter_name in enumerate(self.lsst_filter_names):
                self.lsst_equations[i, -1] = 10.0 ** (-0.4 * (dark_sky_mags[filter_name] - np.log10(3631.0)))

        self.filter_name_dict = {"u": 0, "g": 1, "r": 2, "i": 3, "z": 4, "y": 5}

    def print_fits_used(self):
        """Print out the fit parameters being used"""
        print(
            r"\\tablehead{\colhead{Filter} & \colhead{$r_{12/z}$} & "
            r"\colhead{$a$ (1/radians)} & \colhead{$b$ (1/airmass)} & "
            r"\colhead{$c$ (az term/airmass)} & "
            r"\colhead{$f_z_dark$ (erg/s/cm$^2$)$\\times 10^8$} & "
            r"\colhead{m$_z_dark$}}"
        )
        for key in self.fit_results:
            numbers = ""
            for num in self.fit_results[key]:
                if num > 0.001:
                    numbers += " & %.2f" % num
                else:
                    numbers += " & %.2f" % (num * 1e8)
            print(
                key,
                numbers,
                " & ",
                "%.2f" % (-2.5 * np.log10(self.fit_results[key][-1]) + np.log10(3631.0)),
            )

    def __call__(self, intep_points, filter_names=["u", "g", "r", "i", "z", "y"]):
        if self.mags:
            return self.interp_mag(intep_points, filter_names=filter_names)
        else:
            return self.interp_spec(intep_points)

    def interp_mag(
        self,
        interp_points,
        max_am=3.0,
        limits=(np.radians(15.0), np.radians(-20.0)),
        filter_names=["u", "g", "r", "i", "z", "y"],
    ):
        """
        Parameters
        ----------
        interp_points : `np.ndarray`, (N, 3)
            Interpolation points. Should contain sunAlt, airmass and azRelSun.
        max_am : `float`, optional
            Maximum airmass to calculate twilight sky to.
        limits : `np.ndarray`, (N,), optional
            Sun altitude limits

        Returns
        -------
        spectra, wavelength : `np.ndarray`, (N, 3), `np.ndarray`, (M,)

        Note
        ----
        Originally fit the twilight with a cutoff of sun altitude of
        -11 degrees. I think it can be safely extrapolated farther,
        but be warned you may be entering a regime where it breaks down.
        """
        npts = len(filter_names)
        result = np.zeros((np.size(interp_points), npts), dtype=float)

        out_of_range = np.where(interp_points["sunAlt"] > np.radians(-11))[0]
        if np.size(out_of_range) > 0:
            warnings.warn("Extrapolating twilight beyond a sun altitude of -11 degrees")

        good = np.where(
            (interp_points["sunAlt"] >= np.min(limits))
            & (interp_points["sunAlt"] <= np.max(limits))
            & (interp_points["airmass"] <= max_am)
            & (interp_points["airmass"] >= 1.0)
        )[0]

        for i, filterName in enumerate(filter_names):
            out_of_range = np.where(interp_points["sunAlt"] > np.max(limits))[0]
            if np.size(out_of_range) > 0:
                result[:, i] = np.nan
            else:
                result[good, i] = twilight_func(
                    interp_points[good], *self.lsst_equations[self.filter_name_dict[filterName], :].tolist()
                )

        return {"spec": result, "wave": self.lsst_eff_wave}

    def interp_spec(self, interp_points, max_am=3.0, limits=(np.radians(15.0), np.radians(-20.0))):
        """
        Parameters
        ----------
        interp_points : `np.ndarray`, (N, 3)
            Interpolation points. Should contain sunAlt, airmass and azRelSun.
        max_am : `float`, optional
            Maximum airmass to calculate twilight sky to.
        limits : `np.ndarray`, (N,), optional
            Sun altitude limits

        Returns
        -------
        spectra, wavelength : `np.ndarray`, (N, 3), `np.ndarray`, (M,)

        Note
        ----
        Originally fit the twilight with a cutoff of sun altitude of
        -11 degrees. I think it can be safely extrapolated farther,
        but be warned you may be entering a regime where it breaks down.
        """

        npts = np.size(self.solar_wave)
        result = np.zeros((np.size(interp_points), npts), dtype=float)

        out_of_range = np.where(interp_points["sunAlt"] > np.radians(-11))[0]
        if np.size(out_of_range) > 0:
            warnings.warn("Extrapolating twilight beyond a sun altitude of -11 degrees")

        good = np.where(
            (interp_points["sunAlt"] >= np.min(limits))
            & (interp_points["sunAlt"] <= np.max(limits))
            & (interp_points["airmass"] <= max_am)
            & (interp_points["airmass"] >= 1.0)
        )[0]

        # Compute the expected flux in each of the filters that
        # we have fits for
        fluxes = []
        for filter_name in self.filter_names:
            out_of_range = np.where(interp_points["sunAlt"] > np.max(limits))[0]
            if np.size(out_of_range) > 0:
                fluxes.append(np.nan)
            else:
                fluxes.append(twilight_func(interp_points[good], *self.fit_results[filter_name]))
        fluxes = np.array(fluxes)

        # ratio of model flux to raw solar flux:
        yvals = fluxes.T / (10.0 ** (-0.4 * (self.solar_mag - np.log10(3631.0))))

        # Find wavelengths bluer than cutoff
        blue_region = np.where(self.solar_wave < np.min(self.eff_wave))

        for i, yval in enumerate(yvals):
            interp_f = interp1d(self.eff_wave, yval, bounds_error=False, fill_value=yval[-1])
            ratio = interp_f(self.solar_wave)
            interp_blue = InterpolatedUnivariateSpline(self.eff_wave, yval, k=1)
            ratio[blue_region] = interp_blue(self.solar_wave[blue_region])
            result[good[i]] = self.solar_flux * ratio

        return {"spec": result, "wave": self.solar_wave}


class MoonInterp(BaseSingleInterp):
    """
    Read in the saved Lunar spectra and interpolate.
    """

    def __init__(
        self,
        comp_name="Moon",
        sorted_order=["moonSunSep", "moonAltitude", "hpid"],
        mags=False,
    ):
        super(MoonInterp, self).__init__(comp_name=comp_name, sorted_order=sorted_order, mags=mags)
        # Magic number from when the templates were generated
        self.nside = 4

    def _weighting(self, interp_points, values):
        """
        Weighting for the scattered moonlight.
        """

        result = np.zeros((interp_points.size, np.size(values[0])), dtype=float)

        # Check that moonAltitude is in range, otherwise return zero array
        if np.max(interp_points["moonAltitude"]) < np.min(self.dim_dict["moonAltitude"]):
            return result

        # Find the neighboring healpixels
        hpids, hweights = get_neighbours(
            self.nside, np.pi / 2.0 - interp_points["alt"], interp_points["azRelMoon"]
        )

        badhp = np.isin(hpids.ravel(), self.dim_dict["hpid"], invert=True).reshape(hpids.shape)
        hweights[badhp] = 0.0

        norm = np.sum(hweights, axis=0)
        good = np.where(norm != 0.0)[0]
        hweights[:, good] = hweights[:, good] / norm[good]

        # Find the neighboring moonAltitude points in the grid
        right_m_as, left_m_as, ma_right_w, ma_left_w = self._indx_and_weights(
            interp_points["moonAltitude"], self.dim_dict["moonAltitude"]
        )

        # Find the neighboring moonSunSep points in the grid
        right_mss, left_mss, mss_right_w, mss_left_w = self._indx_and_weights(
            interp_points["moonSunSep"], self.dim_dict["moonSunSep"]
        )

        nhpid = self.dim_dict["hpid"].size
        n_ma = self.dim_dict["moonAltitude"].size
        # Convert the hpid to an index.
        tmp = intid2id(hpids.ravel(), self.dim_dict["hpid"], np.arange(self.dim_dict["hpid"].size))
        hpindx = tmp.reshape(hpids.shape)
        # loop though the hweights and the moonAltitude weights

        for hpid, hweight in zip(hpindx, hweights):
            for maid, maW in zip([right_m_as, left_m_as], [ma_right_w, ma_left_w]):
                for mssid, mssW in zip([right_mss, left_mss], [mss_right_w, mss_left_w]):
                    weight = hweight * maW * mssW
                    result += weight[:, np.newaxis] * values[mssid * nhpid * n_ma + maid * nhpid + hpid]

        return result


class ZodiacalInterp(BaseSingleInterp):
    """
    Interpolate the zodiacal light based on the airmass
    and the healpix ID where the healpixels are in ecliptic
    coordinates, with the sun at ecliptic longitude zero
    """

    def __init__(self, comp_name="Zodiacal", sorted_order=["airmass", "hpid"], mags=False):
        super(ZodiacalInterp, self).__init__(comp_name=comp_name, sorted_order=sorted_order, mags=mags)
        self.nside = hp.npix2nside(
            np.size(np.where(self.spec["airmass"] == np.unique(self.spec["airmass"])[0])[0])
        )

    def _weighting(self, interp_points, values):
        """
        interp_points is a numpy array where interpolation is desired
        values are the model values.
        """
        result = np.zeros((interp_points.size, np.size(values[0])), dtype=float)

        in_range = np.where(
            (interp_points["airmass"] <= np.max(self.dim_dict["airmass"]))
            & (interp_points["airmass"] >= np.min(self.dim_dict["airmass"]))
        )
        use_points = interp_points[in_range]
        # Find the neighboring healpixels
        hpids, hweights = get_neighbours(
            self.nside,
            np.pi / 2.0 - use_points["altEclip"],
            use_points["azEclipRelSun"],
        )

        badhp = np.isin(hpids.ravel(), self.dim_dict["hpid"], invert=True).reshape(hpids.shape)
        hweights[badhp] = 0.0

        norm = np.sum(hweights, axis=0)
        good = np.where(norm != 0.0)[0]
        hweights[:, good] = hweights[:, good] / norm[good]

        am_right_index, am_left_index, am_right_w, am_left_w = self._indx_and_weights(
            use_points["airmass"], self.dim_dict["airmass"]
        )

        nhpid = self.dim_dict["hpid"].size
        # loop though the hweights and the airmass weights
        for hpid, hweight in zip(hpids, hweights):
            for am_index, amW in zip([am_right_index, am_left_index], [am_right_w, am_left_w]):
                weight = hweight * amW
                result[in_range] += weight[:, np.newaxis] * values[am_index * nhpid + hpid]

        return result
