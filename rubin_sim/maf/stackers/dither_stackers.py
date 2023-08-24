__all__ = (
    "setup_dither_stackers",
    "wrap_ra_dec",
    "wrap_ra",
    "in_hexagon",
    "polygon_coords",
    "BaseDitherStacker",
    "RandomDitherFieldPerVisitStacker",
    "RandomDitherFieldPerNightStacker",
    "SpiralDitherFieldPerVisitStacker",
    "SpiralDitherFieldPerNightStacker",
    "HexDitherFieldPerVisitStacker",
    "HexDitherFieldPerNightStacker",
    "RandomDitherPerNightStacker",
    "SpiralDitherPerNightStacker",
    "HexDitherPerNightStacker",
    "RandomRotDitherPerFilterChangeStacker",
)


import warnings

import numpy as np

from .base_stacker import BaseStacker

# Stacker naming scheme:
# [Pattern]Dither[Field]Per[Timescale].
#  Timescale indicates how often the dither offset is changed.
#  The presence of 'Field' indicates that a new offset is chosen per field, on the indicated timescale.
#  The absence of 'Field' indicates that all visits within the indicated timescale use the same dither offset.


# Original dither stackers (Random, Spiral, Hex) written by Lynne Jones (lynnej@uw.edu)
# Additional dither stackers written by Humna Awan (humna.awan@rutgers.edu), with addition of
# constraining dither offsets to be within an inscribed hexagon (code modifications for use here by LJ).


def setup_dither_stackers(ra_col, dec_col, degrees, **kwargs):
    b = BaseStacker()
    stacker_list = []
    if ra_col in b.source_dict:
        stacker_list.append(b.source_dict[ra_col](degrees=degrees, **kwargs))
    if dec_col in b.source_dict:
        if b.source_dict[ra_col] != b.source_dict[dec_col]:
            stacker_list.append(b.source_dict[dec_col](degrees=degrees, **kwargs))
    return stacker_list


def wrap_ra_dec(ra, dec):
    """
    Wrap RA into 0-2pi and Dec into +/0 pi/2.

    Parameters
    ----------
    ra : numpy.ndarray
        RA in radians
    dec : numpy.ndarray
        Dec in radians

    Returns
    -------
    numpy.ndarray, numpy.ndarray
        Wrapped RA/Dec values, in radians.
    """
    # Wrap dec.
    low = np.where(dec < -np.pi / 2.0)[0]
    dec[low] = -1 * (np.pi + dec[low])
    ra[low] = ra[low] - np.pi
    high = np.where(dec > np.pi / 2.0)[0]
    dec[high] = np.pi - dec[high]
    ra[high] = ra[high] - np.pi
    # Wrap RA.
    ra = ra % (2.0 * np.pi)
    return ra, dec


def wrap_ra(ra):
    """
    Wrap only RA values into 0-2pi (using mod).

    Parameters
    ----------
    ra : numpy.ndarray
        RA in radians

    Returns
    -------
    numpy.ndarray
        Wrapped RA values, in radians.
    """
    ra = ra % (2.0 * np.pi)
    return ra


def in_hexagon(x_off, y_off, max_dither):
    """
    Identify dither offsets which fall within the inscribed hexagon.

    Parameters
    ----------
    x_off : numpy.ndarray
        The x values of the dither offsets.
    yoff : numpy.ndarray
        The y values of the dither offsets.
    max_dither : float
        The maximum dither offset.

    Returns
    -------
    numpy.ndarray
        Indexes of the offsets which are within the hexagon inscribed inside the 'max_dither' radius circle.
    """
    # Set up the hexagon limits.
    #  y = mx + b, 2h is the height.
    m = np.sqrt(3.0)
    b = m * max_dither
    h = m / 2.0 * max_dither
    # Identify offsets inside hexagon.
    inside = np.where(
        (y_off < m * x_off + b)
        & (y_off > m * x_off - b)
        & (y_off < -m * x_off + b)
        & (y_off > -m * x_off - b)
        & (y_off < h)
        & (y_off > -h)
    )[0]
    return inside


def polygon_coords(nside, radius, rotation_angle):
    """
    Find the x,y coords of a polygon.

    This is useful for plotting dither points and showing they lie within
    a given shape.

    Parameters
    ----------
    nside : int
        The number of sides of the polygon
    radius : float
        The radius within which to plot the polygon
    rotation_angle : float
        The angle to rotate the polygon to.

    Returns
    -------
    [float, float]
        List of x/y coordinates of the points describing the polygon.
    """
    each_angle = 2 * np.pi / float(nside)
    x_coords = np.zeros(nside, float)
    y_coords = np.zeros(nside, float)
    for i in range(0, nside):
        x_coords[i] = np.sin(each_angle * i + rotation_angle) * radius
        y_coords[i] = np.cos(each_angle * i + rotation_angle) * radius
    return list(zip(x_coords, y_coords))


class BaseDitherStacker(BaseStacker):
    """Base class for dither stackers.

    The base class just adds an easy way to define a stacker as one of the 'dither' types of stackers.
    These run first, before any other stackers.

    Parameters
    ----------
    ra_col : str, optional
        The name of the RA column in the data.
        Default 'fieldRA'.
    dec_col : str, optional
        The name of the Dec column in the data.
        Default 'fieldDec'.
    degrees : bool, optional
        Flag whether RA/Dec should be treated as (and kept as) degrees.
    max_dither : float, optional
        The radius of the maximum dither offset, in degrees.
        Default 1.75 degrees.
    in_hex : bool, optional
        If True, offsets are constrained to lie within a hexagon inscribed within the max_dither circle.
        If False, offsets can lie anywhere out to the edges of the max_dither circle.
        Default True.
    """

    cols_added = []

    def __init__(
        self,
        ra_col="fieldRA",
        dec_col="fieldDec",
        degrees=True,
        max_dither=1.75,
        in_hex=True,
    ):
        # Instantiate the RandomDither object and set internal variables.
        self.ra_col = ra_col
        self.dec_col = dec_col
        self.degrees = degrees
        # Convert max_dither to radians for internal use.
        self.max_dither = np.radians(max_dither)
        self.in_hex = in_hex
        # self.units used for plot labels
        if self.degrees:
            self.units = ["deg", "deg"]
        else:
            self.units = ["rad", "rad"]
        # Values required for framework operation: this specifies the data columns required from the database.
        self.cols_req = [self.ra_col, self.dec_col]


class RandomDitherFieldPerVisitStacker(BaseDitherStacker):
    """
    Randomly dither the RA and Dec pointings up to max_dither degrees from center,
    with a different offset for each field, for each visit.

    Parameters
    ----------
    ra_col : str, optional
        The name of the RA column in the data.
        Default 'fieldRA'.
    dec_col : str, optional
        The name of the Dec column in the data.
        Default 'fieldDec'.
    degrees : bool, optional
        Flag whether RA/Dec should be treated as (and kept as) degrees.
    max_dither : float, optional
        The radius of the maximum dither offset, in degrees.
        Default 1.75 degrees.
    in_hex : bool, optional
        If True, offsets are constrained to lie within a hexagon inscribed within the max_dither circle.
        If False, offsets can lie anywhere out to the edges of the max_dither circle.
        Default True.
    random_seed : int or None, optional
        If set, then used as the random seed for the numpy random number generation for the dither offsets.
        Default None.
    """

    # Values required for framework operation: this specifies the name of the new columns.
    cols_added = ["randomDitherFieldPerVisitRa", "randomDitherFieldPerVisitDec"]

    def __init__(
        self,
        ra_col="fieldRA",
        dec_col="fieldDec",
        degrees=True,
        max_dither=1.75,
        in_hex=True,
        random_seed=None,
    ):
        """
        @ MaxDither in degrees
        """
        super().__init__(
            ra_col=ra_col,
            dec_col=dec_col,
            degrees=degrees,
            max_dither=max_dither,
            in_hex=in_hex,
        )
        self.random_seed = random_seed

    def _generate_random_offsets(self, noffsets):
        x_out = np.array([], float)
        y_out = np.array([], float)
        max_tries = 100
        tries = 0
        while (len(x_out) < noffsets) and (tries < max_tries):
            dithers_rad = np.sqrt(self._rng.rand(noffsets * 2)) * self.max_dither
            dithers_theta = self._rng.rand(noffsets * 2) * np.pi * 2.0
            x_off = dithers_rad * np.cos(dithers_theta)
            y_off = dithers_rad * np.sin(dithers_theta)
            if self.in_hex:
                # Constrain dither offsets to be within hexagon.
                idx = in_hexagon(x_off, y_off, self.max_dither)
                x_off = x_off[idx]
                y_off = y_off[idx]
            x_out = np.concatenate([x_out, x_off])
            y_out = np.concatenate([y_out, y_off])
            tries += 1
        if len(x_out) < noffsets:
            raise ValueError(
                "Could not find enough random points within the hexagon in %d tries. "
                "Try another random seed?" % (max_tries)
            )
        self.x_off = x_out[0:noffsets]
        self.y_off = y_out[0:noffsets]

    def _run(self, sim_data, cols_present=False):
        if cols_present:
            # Column already present in data; assume it is correct and does not need recalculating.
            return sim_data
        # Generate random numbers for dither, using defined seed value if desired.
        if not hasattr(self, "_rng"):
            if self.random_seed is not None:
                self._rng = np.random.RandomState(self.random_seed)
            else:
                self._rng = np.random.RandomState(2178813)

        # Generate the random dither values.
        noffsets = len(sim_data[self.ra_col])
        self._generate_random_offsets(noffsets)
        # Add to RA and dec values.
        if self.degrees:
            ra = np.radians(sim_data[self.ra_col])
            dec = np.radians(sim_data[self.dec_col])
        else:
            ra = sim_data[self.ra_col]
            dec = sim_data[self.dec_col]
        sim_data["randomDitherFieldPerVisitRa"] = ra + self.x_off / np.cos(dec)
        sim_data["randomDitherFieldPerVisitDec"] = dec + self.y_off
        # Wrap back into expected range.
        (
            sim_data["randomDitherFieldPerVisitRa"],
            sim_data["randomDitherFieldPerVisitDec"],
        ) = wrap_ra_dec(
            sim_data["randomDitherFieldPerVisitRa"],
            sim_data["randomDitherFieldPerVisitDec"],
        )
        # Convert to degrees
        if self.degrees:
            for col in self.cols_added:
                sim_data[col] = np.degrees(sim_data[col])
        return sim_data


class RandomDitherFieldPerNightStacker(RandomDitherFieldPerVisitStacker):
    """
    Randomly dither the RA and Dec pointings up to max_dither degrees from center,
    one dither offset per new night of observation of a field.
    e.g. visits within the same night, to the same field, have the same offset.

    Parameters
    ----------
    ra_col : str, optional
        The name of the RA column in the data.
        Default 'fieldRA'.
    dec_col : str, optional
        The name of the Dec column in the data.
        Default 'fieldDec'.
    degrees : bool, optional
        Flag whether RA/Dec should be treated as (and kept as) degrees.
    field_id_col : str, optional
        The name of the fieldId column in the data.
        Used to identify fields which should be identified as the 'same'.
        Default 'fieldId'.
    night_col : str, optional
        The name of the night column in the data.
        Default 'night'.
    max_dither : float, optional
        The radius of the maximum dither offset, in degrees.
        Default 1.75 degrees.
    in_hex : bool, optional
        If True, offsets are constrained to lie within a hexagon inscribed within the max_dither circle.
        If False, offsets can lie anywhere out to the edges of the max_dither circle.
        Default True.
    random_seed : int or None, optional
        If set, then used as the random seed for the numpy random number generation for the dither offsets.
        Default None.
    """

    # Values required for framework operation: this specifies the names of the new columns.
    cols_added = ["randomDitherFieldPerNightRa", "randomDitherFieldPerNightDec"]

    def __init__(
        self,
        ra_col="fieldRA",
        dec_col="fieldDec",
        degrees=True,
        field_id_col="fieldId",
        night_col="night",
        max_dither=1.75,
        in_hex=True,
        random_seed=None,
    ):
        """
        @ MaxDither in degrees
        """
        # Instantiate the RandomDither object and set internal variables.
        super().__init__(
            ra_col=ra_col,
            dec_col=dec_col,
            degrees=degrees,
            max_dither=max_dither,
            in_hex=in_hex,
            random_seed=random_seed,
        )
        self.night_col = night_col
        self.field_id_col = field_id_col
        # Values required for framework operation: this specifies the data columns required from the database.
        self.cols_req = [self.ra_col, self.dec_col, self.night_col, self.field_id_col]

    def _run(self, sim_data, cols_present=False):
        if cols_present:
            return sim_data
        # Generate random numbers for dither, using defined seed value if desired.
        if not hasattr(self, "_rng"):
            if self.random_seed is not None:
                self._rng = np.random.RandomState(self.random_seed)
            else:
                self._rng = np.random.RandomState(872453)

        # Generate the random dither values, one per night per field.
        fields = np.unique(sim_data[self.field_id_col])
        nights = np.unique(sim_data[self.night_col])
        self._generate_random_offsets(len(fields) * len(nights))
        if self.degrees:
            ra = np.radians(sim_data[self.ra_col])
            dec = np.radians(sim_data[self.dec_col])
        else:
            ra = sim_data[self.ra_col]
            dec = sim_data[self.dec_col]
        # counter to ensure new random numbers are chosen every time
        delta = 0
        for fieldid in np.unique(sim_data[self.field_id_col]):
            # Identify observations of this field.
            match = np.where(sim_data[self.field_id_col] == fieldid)[0]
            # Apply dithers, increasing each night.
            nights = sim_data[self.night_col][match]
            vertex_idxs = np.searchsorted(np.unique(nights), nights)
            vertex_idxs = vertex_idxs % len(self.x_off)
            # ensure that the same xOff/yOff entries are not chosen
            delta = delta + len(vertex_idxs)
            sim_data["randomDitherFieldPerNightRa"][match] = ra[match] + self.x_off[vertex_idxs] / np.cos(
                dec[match]
            )
            sim_data["randomDitherFieldPerNightDec"][match] = dec[match] + self.y_off[vertex_idxs]
        # Wrap into expected range.
        (
            sim_data["randomDitherFieldPerNightRa"],
            sim_data["randomDitherFieldPerNightDec"],
        ) = wrap_ra_dec(
            sim_data["randomDitherFieldPerNightRa"],
            sim_data["randomDitherFieldPerNightDec"],
        )
        if self.degrees:
            for col in self.cols_added:
                sim_data[col] = np.degrees(sim_data[col])
        return sim_data


class RandomDitherPerNightStacker(RandomDitherFieldPerVisitStacker):
    """
    Randomly dither the RA and Dec pointings up to max_dither degrees from center,
    one dither offset per night.
    All fields observed within the same night get the same offset.

    Parameters
    ----------
    ra_col : str, optional
        The name of the RA column in the data.
        Default 'fieldRA'.
    dec_col : str, optional
        The name of the Dec column in the data.
        Default 'fieldDec'.
    degrees : bool, optional
        Flag whether RA/Dec should be treated as (and kept as) degrees.
    night_col : str, optional
        The name of the night column in the data.
        Default 'night'.
    max_dither : float, optional
        The radius of the maximum dither offset, in degrees.
        Default 1.75 degrees.
    in_hex : bool, optional
        If True, offsets are constrained to lie within a hexagon inscribed within the max_dither circle.
        If False, offsets can lie anywhere out to the edges of the max_dither circle.
        Default True.
    random_seed : int or None, optional
        If set, then used as the random seed for the numpy random number generation for the dither offsets.
        Default None.
    """

    # Values required for framework operation: this specifies the names of the new columns.
    cols_added = ["randomDitherPerNightRa", "randomDitherPerNightDec"]

    def __init__(
        self,
        ra_col="fieldRA",
        dec_col="fieldDec",
        degrees=True,
        night_col="night",
        max_dither=1.75,
        in_hex=True,
        random_seed=None,
    ):
        """
        @ MaxDither in degrees
        """
        # Instantiate the RandomDither object and set internal variables.
        super().__init__(
            ra_col=ra_col,
            dec_col=dec_col,
            degrees=degrees,
            max_dither=max_dither,
            in_hex=in_hex,
            random_seed=random_seed,
        )
        self.night_col = night_col
        # Values required for framework operation: this specifies the data columns required from the database.
        self.cols_req = [self.ra_col, self.dec_col, self.night_col]

    def _run(self, sim_data, cols_present=False):
        if cols_present:
            return sim_data
        # Generate random numbers for dither, using defined seed value if desired.
        if not hasattr(self, "_rng"):
            if self.random_seed is not None:
                self._rng = np.random.RandomState(self.random_seed)
            else:
                self._rng = np.random.RandomState(66334)

        # Generate the random dither values, one per night.
        nights = np.unique(sim_data[self.night_col])
        self._generate_random_offsets(len(nights))
        if self.degrees:
            ra = np.radians(sim_data[self.ra_col])
            dec = np.radians(sim_data[self.dec_col])
        else:
            ra = sim_data[self.ra_col]
            dec = sim_data[self.dec_col]
        # Add to RA and dec values.
        for n, x, y in zip(nights, self.x_off, self.y_off):
            match = np.where(sim_data[self.night_col] == n)[0]
            sim_data["randomDitherPerNightRa"][match] = ra[match] + x / np.cos(dec[match])
            sim_data["randomDitherPerNightDec"][match] = dec[match] + y
        # Wrap RA/Dec into expected range.
        (
            sim_data["randomDitherPerNightRa"],
            sim_data["randomDitherPerNightDec"],
        ) = wrap_ra_dec(sim_data["randomDitherPerNightRa"], sim_data["randomDitherPerNightDec"])
        if self.degrees:
            for col in self.cols_added:
                sim_data[col] = np.degrees(sim_data[col])
        return sim_data


class SpiralDitherFieldPerVisitStacker(BaseDitherStacker):
    """
    Offset along an equidistant spiral with num_points, out to a maximum radius of max_dither.
    Each visit to a field receives a new, sequential offset.

    Parameters
    ----------
    ra_col : str, optional
        The name of the RA column in the data.
        Default 'fieldRA'.
    dec_col : str, optional
        The name of the Dec column in the data.
        Default 'fieldDec'.
    degrees : bool, optional
        Flag whether RA/Dec should be treated as (and kept as) degrees.
    field_id_col : str, optional
        The name of the fieldId column in the data.
        Used to identify fields which should be identified as the 'same'.
        Default 'fieldId'.
    num_points : int, optional
        The number of points in the spiral.
        Default 60.
    max_dither : float, optional
        The radius of the maximum dither offset, in degrees.
        Default 1.75 degrees.
    n_coils : int, optional
        The number of coils the spiral should have.
        Default 5.
    in_hex : bool, optional
        If True, offsets are constrained to lie within a hexagon inscribed within the max_dither circle.
        If False, offsets can lie anywhere out to the edges of the max_dither circle.
        Default True.
    """

    # Values required for framework operation: this specifies the names of the new columns.
    cols_added = ["spiralDitherFieldPerVisitRa", "spiralDitherFieldPerVisitDec"]

    def __init__(
        self,
        ra_col="fieldRA",
        dec_col="fieldDec",
        degrees=True,
        field_id_col="fieldId",
        num_points=60,
        max_dither=1.75,
        n_coils=5,
        in_hex=True,
    ):
        """
        @ MaxDither in degrees
        """
        super().__init__(
            ra_col=ra_col,
            dec_col=dec_col,
            degrees=degrees,
            max_dither=max_dither,
            in_hex=in_hex,
        )
        self.field_id_col = field_id_col
        # Convert max_dither from degrees (internal units for ra/dec are radians)
        self.num_points = num_points
        self.n_coils = n_coils
        # Values required for framework operation: this specifies the data columns required from the database.
        self.cols_req = [self.ra_col, self.dec_col, self.field_id_col]

    def _generate_spiral_offsets(self):
        # First generate a full archimedean spiral ..
        theta = np.arange(0.0001, self.n_coils * np.pi * 2.0, 0.001)
        a = self.max_dither / theta.max()
        if self.in_hex:
            a = 0.85 * a
        r = theta * a
        # Then pick out equidistant points along the spiral.
        arc = a / 2.0 * (theta * np.sqrt(1 + theta**2) + np.log(theta + np.sqrt(1 + theta**2)))
        stepsize = arc.max() / float(self.num_points)
        arcpts = np.arange(0, arc.max(), stepsize)
        arcpts = arcpts[0 : self.num_points]
        rpts = np.zeros(self.num_points, float)
        thetapts = np.zeros(self.num_points, float)
        for i, ap in enumerate(arcpts):
            diff = np.abs(arc - ap)
            match = np.where(diff == diff.min())[0][0]
            rpts[i] = r[match]
            thetapts[i] = theta[match]
        # Translate these r/theta points into x/y (ra/dec) offsets.
        self.x_off = rpts * np.cos(thetapts)
        self.y_off = rpts * np.sin(thetapts)

    def _run(self, sim_data, cols_present=False):
        if cols_present:
            return sim_data
        # Generate the spiral offset vertices.
        self._generate_spiral_offsets()
        # Now apply to observations.
        if self.degrees:
            ra = np.radians(sim_data[self.ra_col])
            dec = np.radians(sim_data[self.dec_col])
        else:
            ra = sim_data[self.ra_col]
            dec = sim_data[self.dec_col]
        for fieldid in np.unique(sim_data[self.field_id_col]):
            match = np.where(sim_data[self.field_id_col] == fieldid)[0]
            # Apply sequential dithers, increasing with each visit.
            vertex_idxs = np.arange(0, len(match), 1)
            vertex_idxs = vertex_idxs % self.num_points
            sim_data["spiralDitherFieldPerVisitRa"][match] = ra[match] + self.x_off[vertex_idxs] / np.cos(
                dec[match]
            )
            sim_data["spiralDitherFieldPerVisitDec"][match] = dec[match] + self.y_off[vertex_idxs]
        # Wrap into expected range.
        (
            sim_data["spiralDitherFieldPerVisitRa"],
            sim_data["spiralDitherFieldPerVisitDec"],
        ) = wrap_ra_dec(
            sim_data["spiralDitherFieldPerVisitRa"],
            sim_data["spiralDitherFieldPerVisitDec"],
        )
        if self.degrees:
            for col in self.cols_added:
                sim_data[col] = np.degrees(sim_data[col])
        return sim_data


class SpiralDitherFieldPerNightStacker(SpiralDitherFieldPerVisitStacker):
    """
    Offset along an equidistant spiral with num_points, out to a maximum radius of max_dither.
    Each field steps along a sequential series of offsets, each night it is observed.

    Parameters
    ----------
    ra_col : str, optional
        The name of the RA column in the data.
        Default 'fieldRA'.
    dec_col : str, optional
        The name of the Dec column in the data.
        Default 'fieldDec'.
    degrees : bool, optional
        Flag whether RA/Dec should be treated as (and kept as) degrees.
    field_id_col : str, optional
        The name of the fieldId column in the data.
        Used to identify fields which should be identified as the 'same'.
        Default 'fieldId'.
    night_col : str, optional
        The name of the night column in the data.
        Default 'night'.
    num_points : int, optional
        The number of points in the spiral.
        Default 60.
    max_dither : float, optional
        The radius of the maximum dither offset, in degrees.
        Default 1.75 degrees.
    n_coils : int, optional
        The number of coils the spiral should have.
        Default 5.
    in_hex : bool, optional
        If True, offsets are constrained to lie within a hexagon inscribed within the max_dither circle.
        If False, offsets can lie anywhere out to the edges of the max_dither circle.
        Default True.
    """

    # Values required for framework operation: this specifies the names of the new columns.
    cols_added = ["spiralDitherFieldPerNightRa", "spiralDitherFieldPerNightDec"]

    def __init__(
        self,
        ra_col="fieldRA",
        dec_col="fieldDec",
        degrees=True,
        field_id_col="fieldId",
        night_col="night",
        num_points=60,
        max_dither=1.75,
        n_coils=5,
        in_hex=True,
    ):
        """
        @ MaxDither in degrees
        """
        super().__init__(
            ra_col=ra_col,
            dec_col=dec_col,
            degrees=degrees,
            field_id_col=field_id_col,
            num_points=num_points,
            max_dither=max_dither,
            n_coils=n_coils,
            in_hex=in_hex,
        )
        self.night_col = night_col
        # Values required for framework operation: this specifies the data columns required from the database.
        self.cols_req.append(self.night_col)

    def _run(self, sim_data, cols_present=False):
        if cols_present:
            return sim_data
        self._generate_spiral_offsets()
        if self.degrees:
            ra = np.radians(sim_data[self.ra_col])
            dec = np.radians(sim_data[self.dec_col])
        else:
            ra = sim_data[self.ra_col]
            dec = sim_data[self.dec_col]
        for fieldid in np.unique(sim_data[self.field_id_col]):
            # Identify observations of this field.
            match = np.where(sim_data[self.field_id_col] == fieldid)[0]
            # Apply a sequential dither, increasing each night.
            nights = sim_data[self.night_col][match]
            vertex_idxs = np.searchsorted(np.unique(nights), nights)
            vertex_idxs = vertex_idxs % self.num_points
            sim_data["spiralDitherFieldPerNightRa"][match] = ra[match] + self.x_off[vertex_idxs] / np.cos(
                dec[match]
            )
            sim_data["spiralDitherFieldPerNightDec"][match] = dec[match] + self.y_off[vertex_idxs]
        # Wrap into expected range.
        (
            sim_data["spiralDitherFieldPerNightRa"],
            sim_data["spiralDitherFieldPerNightDec"],
        ) = wrap_ra_dec(
            sim_data["spiralDitherFieldPerNightRa"],
            sim_data["spiralDitherFieldPerNightDec"],
        )
        if self.degrees:
            for col in self.cols_added:
                sim_data[col] = np.degrees(sim_data[col])
        return sim_data


class SpiralDitherPerNightStacker(SpiralDitherFieldPerVisitStacker):
    """
    Offset along an equidistant spiral with num_points, out to a maximum radius of max_dither.
    All fields observed in the same night receive the same sequential offset, changing per night.

    Parameters
    ----------
    ra_col : str, optional
        The name of the RA column in the data.
        Default 'fieldRA'.
    dec_col : str, optional
        The name of the Dec column in the data.
        Default 'fieldDec'.
    degrees : bool, optional
        Flag whether RA/Dec should be treated as (and kept as) degrees.
    field_id_col : str, optional
        The name of the fieldId column in the data.
        Used to identify fields which should be identified as the 'same'.
        Default 'fieldId'.
    night_col : str, optional
        The name of the night column in the data.
        Default 'night'.
    num_points : int, optional
        The number of points in the spiral.
        Default 60.
    max_dither : float, optional
        The radius of the maximum dither offset, in degrees.
        Default 1.75 degrees.
    n_coils : int, optional
        The number of coils the spiral should have.
        Default 5.
    in_hex : bool, optional
        If True, offsets are constrained to lie within a hexagon inscribed within the max_dither circle.
        If False, offsets can lie anywhere out to the edges of the max_dither circle.
        Default True.
    """

    # Values required for framework operation: this specifies the names of the new columns.
    cols_added = ["spiralDitherPerNightRa", "spiralDitherPerNightDec"]

    def __init__(
        self,
        ra_col="fieldRA",
        dec_col="fieldDec",
        degrees=True,
        field_id_col="fieldId",
        night_col="night",
        num_points=60,
        max_dither=1.75,
        n_coils=5,
        in_hex=True,
    ):
        """
        @ MaxDither in degrees
        """
        super().__init__(
            ra_col=ra_col,
            dec_col=dec_col,
            degrees=degrees,
            field_id_col=field_id_col,
            num_points=num_points,
            max_dither=max_dither,
            n_coils=n_coils,
            in_hex=in_hex,
        )
        self.night_col = night_col
        # Values required for framework operation: this specifies the data columns required from the database.
        self.cols_req.append(self.night_col)

    def _run(self, sim_data, cols_present=False):
        if cols_present:
            return sim_data
        self._generate_spiral_offsets()
        nights = np.unique(sim_data[self.night_col])
        if self.degrees:
            ra = np.radians(sim_data[self.ra_col])
            dec = np.radians(sim_data[self.dec_col])
        else:
            ra = sim_data[self.ra_col]
            dec = sim_data[self.dec_col]
        # Add to RA and dec values.
        vertex_idxs = np.searchsorted(nights, sim_data[self.night_col])
        vertex_idxs = vertex_idxs % self.num_points
        sim_data["spiralDitherPerNightRa"] = ra + self.x_off[vertex_idxs] / np.cos(dec)
        sim_data["spiralDitherPerNightDec"] = dec + self.y_off[vertex_idxs]
        # Wrap RA/Dec into expected range.
        (
            sim_data["spiralDitherPerNightRa"],
            sim_data["spiralDitherPerNightDec"],
        ) = wrap_ra_dec(sim_data["spiralDitherPerNightRa"], sim_data["spiralDitherPerNightDec"])
        if self.degrees:
            for col in self.cols_added:
                sim_data[col] = np.degrees(sim_data[col])
        return sim_data


class HexDitherFieldPerVisitStacker(BaseDitherStacker):
    """
    Use offsets from the hexagonal grid of 'hexdither', but visit each vertex sequentially.
    Sequential offset for each visit.

    Parameters
    ----------
    ra_col : str, optional
        The name of the RA column in the data.
        Default 'fieldRA'.
    dec_col : str, optional
        The name of the Dec column in the data.
        Default 'fieldDec'.
    degrees : bool, optional
        Flag whether RA/Dec should be treated as (and kept as) degrees.
    field_id_col : str, optional
        The name of the fieldId column in the data.
        Used to identify fields which should be identified as the 'same'.
        Default 'fieldId'.
    max_dither : float, optional
        The radius of the maximum dither offset, in degrees.
        Default 1.75 degrees.
    in_hex : bool, optional
        If True, offsets are constrained to lie within a hexagon inscribed within the max_dither circle.
        If False, offsets can lie anywhere out to the edges of the max_dither circle.
        Default True.
    """

    # Values required for framework operation: this specifies the names of the new columns.
    cols_added = ["hexDitherFieldPerVisitRa", "hexDitherFieldPerVisitDec"]

    def __init__(
        self,
        ra_col="fieldRA",
        dec_col="fieldDec",
        degrees=True,
        field_id_col="fieldId",
        max_dither=1.75,
        in_hex=True,
    ):
        """
        @ MaxDither in degrees
        """
        super().__init__(
            ra_col=ra_col,
            dec_col=dec_col,
            degrees=degrees,
            max_dither=max_dither,
            in_hex=in_hex,
        )
        self.field_id_col = field_id_col
        # Values required for framework operation: this specifies the data columns required from the database.
        self.cols_req = [self.ra_col, self.dec_col, self.field_id_col]

    def _generate_hex_offsets(self):
        # Set up basics of dither pattern.
        dith_level = 4
        nrows = 2**dith_level
        halfrows = int(nrows / 2.0)
        # Calculate size of each offset
        dith_size_x = self.max_dither * 2.0 / float(nrows)
        dith_size_y = np.sqrt(3) * self.max_dither / float(nrows)  # sqrt 3 comes from hexagon
        if self.in_hex:
            dith_size_x = 0.95 * dith_size_x
            dith_size_y = 0.95 * dith_size_y
        # Calculate the row identification number, going from 0 at center
        nid_row = np.arange(-halfrows, halfrows + 1, 1)
        # and calculate the number of vertices in each row.
        vert_in_row = np.arange(-halfrows, halfrows + 1, 1)
        # First calculate how many vertices we will create in each row.
        total_vert = 0
        for i in range(-halfrows, halfrows + 1, 1):
            vert_in_row[i] = (nrows + 1) - abs(nid_row[i])
            total_vert += vert_in_row[i]
        self.num_points = total_vert
        self.x_off = []
        self.y_off = []
        # Calculate offsets over hexagonal grid.
        for i in range(0, nrows + 1, 1):
            for j in range(0, vert_in_row[i], 1):
                self.x_off.append(dith_size_x * (j - (vert_in_row[i] - 1) / 2.0))
                self.y_off.append(dith_size_y * nid_row[i])
        self.x_off = np.array(self.x_off)
        self.y_off = np.array(self.y_off)

    def _run(self, sim_data, cols_present=False):
        if cols_present:
            return sim_data
        self._generate_hex_offsets()
        if self.degrees:
            ra = np.radians(sim_data[self.ra_col])
            dec = np.radians(sim_data[self.dec_col])
        else:
            ra = sim_data[self.ra_col]
            dec = sim_data[self.dec_col]
        for fieldid in np.unique(sim_data[self.field_id_col]):
            # Identify observations of this field.
            match = np.where(sim_data[self.field_id_col] == fieldid)[0]
            # Apply sequential dithers, increasing with each visit.
            vertex_idxs = np.arange(0, len(match), 1)
            vertex_idxs = vertex_idxs % self.num_points
            sim_data["hexDitherFieldPerVisitRa"][match] = ra[match] + self.x_off[vertex_idxs] / np.cos(
                dec[match]
            )
            sim_data["hexDitherFieldPerVisitDec"][match] = dec[match] + self.y_off[vertex_idxs]
        # Wrap into expected range.
        (
            sim_data["hexDitherFieldPerVisitRa"],
            sim_data["hexDitherFieldPerVisitDec"],
        ) = wrap_ra_dec(sim_data["hexDitherFieldPerVisitRa"], sim_data["hexDitherFieldPerVisitDec"])
        if self.degrees:
            for col in self.cols_added:
                sim_data[col] = np.degrees(sim_data[col])
        return sim_data


class HexDitherFieldPerNightStacker(HexDitherFieldPerVisitStacker):
    """
    Use offsets from the hexagonal grid of 'hexdither', but visit each vertex sequentially.
    Sequential offset for each night of visits.

    Parameters
    ----------
    ra_col : str, optional
        The name of the RA column in the data.
        Default 'fieldRA'.
    dec_col : str, optional
        The name of the Dec column in the data.
        Default 'fieldDec'.
    degrees : bool, optional
        Flag whether RA/Dec should be treated as (and kept as) degrees.
    field_id_col : str, optional
        The name of the fieldId column in the data.
        Used to identify fields which should be identified as the 'same'.
        Default 'fieldId'.
    night_col : str, optional
        The name of the night column in the data.
        Default 'night'.
    max_dither : float, optional
        The radius of the maximum dither offset, in degrees.
        Default 1.75 degrees.
    in_hex : bool, optional
        If True, offsets are constrained to lie within a hexagon inscribed within the max_dither circle.
        If False, offsets can lie anywhere out to the edges of the max_dither circle.
        Default True.
    """

    # Values required for framework operation: this specifies the names of the new columns.
    cols_added = ["hexDitherFieldPerNightRa", "hexDitherFieldPerNightDec"]

    def __init__(
        self,
        ra_col="fieldRA",
        dec_col="fieldDec",
        degrees=True,
        field_id_col="fieldId",
        night_col="night",
        max_dither=1.75,
        in_hex=True,
    ):
        """
        @ MaxDither in degrees
        """
        super().__init__(
            ra_col=ra_col,
            dec_col=dec_col,
            field_id_col=field_id_col,
            degrees=degrees,
            max_dither=max_dither,
            in_hex=in_hex,
        )
        self.night_col = night_col
        # Values required for framework operation: this specifies the data columns required from the database.
        self.cols_req.append(self.night_col)

    def _run(self, sim_data, cols_present=False):
        if cols_present:
            return sim_data
        self._generate_hex_offsets()
        if self.degrees:
            ra = np.radians(sim_data[self.ra_col])
            dec = np.radians(sim_data[self.dec_col])
        else:
            ra = sim_data[self.ra_col]
            dec = sim_data[self.dec_col]
        for fieldid in np.unique(sim_data[self.field_id_col]):
            # Identify observations of this field.
            match = np.where(sim_data[self.field_id_col] == fieldid)[0]
            # Apply a sequential dither, increasing each night.
            vertex_idxs = np.arange(0, len(match), 1)
            nights = sim_data[self.night_col][match]
            vertex_idxs = np.searchsorted(np.unique(nights), nights)
            vertex_idxs = vertex_idxs % self.num_points
            sim_data["hexDitherFieldPerNightRa"][match] = ra[match] + self.x_off[vertex_idxs] / np.cos(
                dec[match]
            )
            sim_data["hexDitherFieldPerNightDec"][match] = dec[match] + self.y_off[vertex_idxs]
        # Wrap into expected range.
        (
            sim_data["hexDitherFieldPerNightRa"],
            sim_data["hexDitherFieldPerNightDec"],
        ) = wrap_ra_dec(sim_data["hexDitherFieldPerNightRa"], sim_data["hexDitherFieldPerNightDec"])
        if self.degrees:
            for col in self.cols_added:
                sim_data[col] = np.degrees(sim_data[col])
        return sim_data


class HexDitherPerNightStacker(HexDitherFieldPerVisitStacker):
    """
    Use offsets from the hexagonal grid of 'hexdither', but visit each vertex sequentially.
    Sequential offset per night for all fields.

    Parameters
    ----------
    ra_col : str, optional
        The name of the RA column in the data.
        Default 'fieldRA'.
    dec_col : str, optional
        The name of the Dec column in the data.
        Default 'fieldDec'.
    degrees : bool, optional
        Flag whether RA/Dec should be treated as (and kept as) degrees.
    field_id_col : str, optional
        The name of the fieldId column in the data.
        Used to identify fields which should be identified as the 'same'.
        Default 'fieldId'.
    night_col : str, optional
        The name of the night column in the data.
        Default 'night'.
    max_dither : float, optional
        The radius of the maximum dither offset, in degrees.
        Default 1.75 degrees.
    in_hex : bool, optional
        If True, offsets are constrained to lie within a hexagon inscribed within the max_dither circle.
        If False, offsets can lie anywhere out to the edges of the max_dither circle.
        Default True.
    """

    # Values required for framework operation: this specifies the names of the new columns.
    cols_added = ["hexDitherPerNightRa", "hexDitherPerNightDec"]

    def __init__(
        self,
        ra_col="fieldRA",
        dec_col="fieldDec",
        degrees=True,
        field_id_col="fieldId",
        night_col="night",
        max_dither=1.75,
        in_hex=True,
    ):
        """
        @ MaxDither in degrees
        """
        super().__init__(
            ra_col=ra_col,
            dec_col=dec_col,
            degrees=degrees,
            field_id_col=field_id_col,
            max_dither=max_dither,
            in_hex=in_hex,
        )
        self.night_col = night_col
        # Values required for framework operation: this specifies the data columns required from the database.
        self.cols_req.append(self.night_col)
        self.added_ra = self.cols_added[0]
        self.added_dec = self.cols_added[1]

    def _run(self, sim_data, cols_present=False):
        if cols_present:
            return sim_data
        # Generate the spiral dither values
        self._generate_hex_offsets()
        nights = np.unique(sim_data[self.night_col])
        if self.degrees:
            ra = np.radians(sim_data[self.ra_col])
            dec = np.radians(sim_data[self.dec_col])
        else:
            ra = sim_data[self.ra_col]
            dec = sim_data[self.dec_col]
        # Add to RA and dec values.
        vertex_id = 0
        for n in nights:
            match = np.where(sim_data[self.night_col] == n)[0]
            vertex_id = vertex_id % self.num_points
            sim_data[self.added_ra][match] = ra[match] + self.x_off[vertex_id] / np.cos(dec[match])
            sim_data[self.added_dec][match] = dec[match] + self.y_off[vertex_id]
            vertex_id += 1
        # Wrap RA/Dec into expected range.
        sim_data[self.added_ra], sim_data[self.added_dec] = wrap_ra_dec(
            sim_data[self.added_ra], sim_data[self.added_dec]
        )
        if self.degrees:
            for col in self.cols_added:
                sim_data[col] = np.degrees(sim_data[col])
        return sim_data


class RandomRotDitherPerFilterChangeStacker(BaseDitherStacker):
    """
    Randomly dither the physical angle of the telescope rotator wrt the mount,
    after every filter change. Visits (in between filter changes) that cannot
    all be assigned an offset without surpassing the rotator limit are not
    dithered.

    Parameters
    ----------
    rot_tel_col : str, optional
        The name of the column in the data specifying the physical angle
        of the telescope rotator wrt. the mount.
        Default: 'rotTelPos'.
    filter_col : str, optional
        The name of the filter column in the data.
        Default: 'filter'.
    degrees : `bool`, optional
        True if angles in the database are in degrees (default).
        If True, returned dithered values are in degrees also.
        If False, angles assumed to be in radians and returned in radians.
    max_dither : float, optional
        Abs(maximum) rotational dither, in degrees. The dithers then will be
        between -max_dither to max_dither.
        Default: 90 degrees.
    max_rot_angle : float, optional
        Maximum rotator angle possible for the camera (degrees). Default 90 degrees.
    min_rot_angle : float, optional
        Minimum rotator angle possible for the camera (degrees). Default -90 degrees.
    random_seed: int, optional
        If set, then used as the random seed for the numpy random number
        generation for the dither offsets.
        Default: None.
    debug: bool, optinal
        If True, will print intermediate steps and plots histograms of
        rotTelPos for cases when no dither is applied.
        Default: False
    """

    # Values required for framework operation: this specifies the names of the new columns.
    cols_added = ["randomDitherPerFilterChangeRotTelPos"]

    def __init__(
        self,
        rot_tel_col="rotTelPos",
        filter_col="filter",
        degrees=True,
        max_dither=90.0,
        max_rot_angle=90,
        min_rot_angle=-90,
        random_seed=None,
        debug=False,
    ):
        # Instantiate the RandomDither object and set internal variables.
        self.rot_tel_col = rot_tel_col
        self.filter_col = filter_col
        self.degrees = degrees
        self.max_dither = max_dither
        self.max_rot_angle = max_rot_angle
        self.min_rot_angle = min_rot_angle
        self.random_seed = random_seed
        # self.units used for plot labels
        if self.degrees:
            self.units = ["deg"]
        else:
            self.units = ["rad"]
            # Convert user-specified values into radians as well.
            self.max_dither = np.radians(self.max_dither)
            self.max_rot_angle = np.radians(self.max_rot_angle)
            self.min_rot_angle = np.radians(self.min_rot_angle)
        self.debug = debug

        # Values required for framework operation: specify the data columns required from the database.
        self.cols_req = [self.rot_tel_col, self.filter_col]

    def _run(self, sim_data, cols_present=False):
        if self.debug:
            import matplotlib.pyplot as plt

        # Just go ahead and return if the columns were already in place.
        if cols_present:
            return sim_data

        # Generate random numbers for dither, using defined seed value if desired.
        # Note that we must define the random state for np.random, to ensure consistency in the build system.
        if not hasattr(self, "_rng"):
            if self.random_seed is not None:
                self._rng = np.random.RandomState(self.random_seed)
            else:
                self._rng = np.random.RandomState(544320)

        if len(np.where(sim_data[self.rot_tel_col] > self.max_rot_angle)[0]) > 0:
            warnings.warn(
                "Input data does not respect the specified maxRotAngle constraint: "
                "(Re)Setting maxRotAngle to max value in the input data: %s" % max(sim_data[self.rot_tel_col])
            )
            self.max_rot_angle = max(sim_data[self.rot_tel_col])
        if len(np.where(sim_data[self.rot_tel_col] < self.min_rot_angle)[0]) > 0:
            warnings.warn(
                "Input data does not respect the specified minRotAngle constraint: "
                "(Re)Setting minRotAngle to min value in the input data: %s" % min(sim_data[self.rot_tel_col])
            )
            self.min_rot_angle = min(sim_data[self.rot_tel_col])

        # Identify points where the filter changes.
        change_idxs = np.where(sim_data[self.filter_col][1:] != sim_data[self.filter_col][:-1])[0]

        # Add the random offsets to the RotTelPos values.
        rot_dither = self.cols_added[0]

        if len(change_idxs) == 0:
            # There are no filter changes, so nothing to dither. Just use original values.
            sim_data[rot_dither] = sim_data[self.rot_tel_col]
        else:
            # For each filter change, generate a series of random values for the offsets,
            # between +/- self.max_dither. These are potential values for the rotational offset.
            # The offset actually used will be  confined to ensure that rotTelPos for all visits in
            # that set of observations (between filter changes) fall within
            # the specified min/maxRotAngle -- without truncating the rotTelPos values.

            # Generate more offsets than needed - either 2x filter changes or 2500, whichever is bigger.
            # 2500 is an arbitrary number.
            max_num = max(len(change_idxs) * 2, 2500)

            rot_offset = np.zeros(len(sim_data), float)
            # Some sets of visits will not be assigned dithers: it was too hard to find an offset.
            n_problematic_ones = 0

            # Loop over the filter change indexes (current filter change, next filter change) to identify
            # sets of visits that should have the same offset.
            for c, cn in zip(change_idxs, change_idxs[1:]):
                random_offsets = self._rng.rand(max_num + 1) * 2.0 * self.max_dither - self.max_dither
                i = 0
                potential_offset = random_offsets[i]
                # Calculate new rotTelPos values, if we used this offset.
                new_rot_tel = sim_data[self.rot_tel_col][c + 1 : cn + 1] + potential_offset
                # Does it work? Do all values fall within minRotAngle / maxRotAngle?
                good_to_go = (new_rot_tel >= self.min_rot_angle).all() and (
                    new_rot_tel <= self.max_rot_angle
                ).all()
                while (not good_to_go) and (i < max_num):
                    # break if find a good offset or hit max_num tries.
                    i += 1
                    potential_offset = random_offsets[i]
                    new_rot_tel = sim_data[self.rot_tel_col][c + 1 : cn + 1] + potential_offset
                    good_to_go = (new_rot_tel >= self.min_rot_angle).all() and (
                        new_rot_tel <= self.max_rot_angle
                    ).all()

                if not good_to_go:  # i.e. no good offset was found after max_num tries
                    n_problematic_ones += 1
                    rot_offset[c + 1 : cn + 1] = 0.0  # no dither
                else:
                    rot_offset[c + 1 : cn + 1] = random_offsets[i]  # assign the chosen offset

            # Handle the last set of observations (after the last filter change to the end of the survey).
            random_offsets = self._rng.rand(max_num + 1) * 2.0 * self.max_dither - self.max_dither
            i = 0
            potential_offset = random_offsets[i]
            new_rot_tel = sim_data[self.rot_tel_col][change_idxs[-1] + 1 :] + potential_offset
            good_to_go = (new_rot_tel >= self.min_rot_angle).all() and (
                new_rot_tel <= self.max_rot_angle
            ).all()
            while (not good_to_go) and (i < max_num):
                # break if find a good offset or cant (after max_num tries)
                i += 1
                potential_offset = random_offsets[i]
                new_rot_tel = sim_data[self.rot_tel_col][change_idxs[-1] + 1 :] + potential_offset
                good_to_go = (new_rot_tel >= self.min_rot_angle).all() and (
                    new_rot_tel <= self.max_rot_angle
                ).all()

            if not good_to_go:  # i.e. no good offset was found after max_num tries
                n_problematic_ones += 1
                rot_offset[c + 1 : cn + 1] = 0.0
            else:
                rot_offset[change_idxs[-1] + 1 :] = potential_offset

        # Assign the dithers
        sim_data[rot_dither] = sim_data[self.rot_tel_col] + rot_offset

        # Final check to make sure things are okay
        good_to_go = (sim_data[rot_dither] >= self.min_rot_angle).all() and (
            sim_data[rot_dither] <= self.max_rot_angle
        ).all()
        if not good_to_go:
            message = "Rotational offsets are not working properly:\n"
            message += " dithered rotTelPos: %s\n" % (sim_data[rot_dither])
            message += " minRotAngle: %s ; maxRotAngle: %s" % (
                self.min_rot_angle,
                self.max_rot_angle,
            )
            raise ValueError(message)
        else:
            return sim_data
