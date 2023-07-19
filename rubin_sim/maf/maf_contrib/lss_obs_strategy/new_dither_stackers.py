##############################################################################################################
# Purpose: implement new dithering strategies.

# The stackers here follow the naming scheme:  [Pattern]Dither[Field]Per[Timescale]. The absence of the
# keyword 'Field' implies dither assignment to all fields.

# Dithers are restricted to the hexagon inscribed in the circle with radius maxDither, where maxDither is the
# required dither offset (generally taken to be radius of the FOV).

# Humna Awan: humna.awan@rutgers.edu
# Last updated: 06/11/16
###############################################################################################################
__all__ = (
    "RepulsiveRandomDitherFieldPerVisitStacker",
    "RepulsiveRandomDitherFieldPerNightStacker",
    "RepulsiveRandomDitherPerNightStacker",
    "FermatSpiralDitherFieldPerVisitStacker",
    "FermatSpiralDitherFieldPerNightStacker",
    "FermatSpiralDitherPerNightStacker",
    "PentagonDiamondDitherFieldPerSeasonStacker",
    "PentagonDitherPerSeasonStacker",
    "PentagonDiamondDitherPerSeasonStacker",
    "SpiralDitherPerSeasonStacker",
)

import numpy as np

from rubin_sim.maf.stackers import BaseStacker, SpiralDitherFieldPerVisitStacker, polygon_coords, wrap_ra_dec
from rubin_sim.utils import calc_season


class RepulsiveRandomDitherFieldPerVisitStacker(BaseStacker):
    """
    Repulsive-randomly dither the RA and Dec pointings up to max_dither degrees from center,
    different offset per visit for each field.

    Note: dithers are confined to the hexagon inscribed in the circle with radius max_dither.

    Parameters
    -------------------
    ra_col: str
        name of the RA column in the data. Default: 'fieldRA'.
    dec_col : str
        name of the Dec column in the data. Default: 'fieldDec'.
    field_id_col : str
        name of the fieldID column in the data. Default: 'field_id_col'.
    max_dither: float
        radius of the maximum dither offset, in degrees. Default: 1.75
    random_seed: int
        random seed for the numpy random number generation for the dither offsets.
        Default: None.
    print_info: `bool`
        set to True to print out information about the number of squares considered,
        number of points chosen, and the filling factor. Default: False
    """

    def __init__(
        self,
        ra_col="fieldRA",
        dec_col="fieldDec",
        field_id_col="fieldID",
        max_dither=1.75,
        random_seed=None,
        print_info=False,
    ):
        # Instantiate the RandomDither object and set internal variables.
        self.ra_col = ra_col
        self.dec_col = dec_col
        self.field_id_col = field_id_col
        # Convert max_dither from degrees (internal units for ra/dec are radians)
        self.max_dither = np.radians(max_dither)
        self.random_seed = random_seed
        self.print_info = print_info
        # self.units used for plot labels
        self.units = ["rad", "rad"]
        # Values required for framework operation: this specifies the names of the new columns.
        self.cols_added = [
            "repulsiveRandomDitherFieldPerVisitRa",
            "repulsiveRandomDitherFieldPerVisitDec",
        ]
        # Values required for framework operation: this specifies the data columns required from the database.
        self.cols_req = [self.ra_col, self.dec_col, self.field_id_col]

    def _generate_rep_random_offsets(self, noffsets, num_tiles):
        # Goal: Tile the circumscribing square with squares. Discard those that fall outside the hexagon.
        # Then choose a square repulsive-randomly (i.e. choose without replacement), and choose a random
        # point from the chosen square.
        noffsets = int(noffsets)
        num_tiles = int(num_tiles)

        square_side = self.max_dither * 2  # circumscribing square. center at (0,0)
        tile_side = square_side / np.sqrt(num_tiles)

        x_center = np.zeros(num_tiles)  # x-coords of the tiles' center
        y_center = np.zeros(num_tiles)  # y-coords of the tiles' center

        # fill in x-coordinates
        k = 0
        x_center[k] = -tile_side * ((np.sqrt(num_tiles) / 2.0) - 0.5)  # far left x-coord

        temp_xarr = []
        temp_xarr.append(x_center[k])
        while k < (np.sqrt(num_tiles) - 1):
            # fill xCoords for squares right above the x-axis
            k += 1
            x_center[k] = x_center[k - 1] + tile_side
            temp_xarr.append(x_center[k])

        # fill in the rest of the x_center array
        indices = np.arange(k + 1, len(x_center))
        indices = indices % len(temp_xarr)
        temp_xarr = np.array(temp_xarr)
        x_center[k + 1 : num_tiles] = temp_xarr[indices]

        # fill in the y-coords
        i = 0
        temp = np.empty(len(temp_xarr))
        while i < num_tiles:
            # the highest y-center coord above the x-axis
            if i == 0:
                temp.fill(tile_side * ((np.sqrt(num_tiles) / 2.0) - 0.5))
            # y-centers below the top one
            else:
                temp.fill(y_center[i - 1] - tile_side)

            y_center[i : i + len(temp)] = temp
            i += len(temp)

        # set up the hexagon
        b = np.sqrt(3.0) * self.max_dither
        m = np.sqrt(3.0)
        h = self.max_dither * np.sqrt(3.0) / 2.0

        # find the points that are inside hexagon
        inside_hex = np.where(
            (y_center < m * x_center + b)
            & (y_center > m * x_center - b)
            & (y_center < -m * x_center + b)
            & (y_center > -m * x_center - b)
            & (y_center < h)
            & (y_center > -h)
        )[0]

        num_points_inside_hex = len(inside_hex)
        if self.print_info:
            print("NumPointsInsideHexagon: ", num_points_inside_hex)
            print("Total squares chosen: ", len(x_center))
            print(
                "Filling factor for repRandom (Number of points needed/Number of points in hexagon): ",
                float(noffsets) / num_points_inside_hex,
            )

        # keep only the points that are inside the hexagon
        temp_x = x_center.copy()
        temp_y = y_center.copy()
        x_center = list(temp_x[inside_hex])
        y_center = list(temp_y[inside_hex])
        x_center_copy = list(np.array(x_center).copy())  # in case need to reuse the squares
        y_center_copy = list(np.array(y_center).copy())  # in case need to reuse the squares

        # initiate the offsets' array
        x_off = np.zeros(noffsets)
        y_off = np.zeros(noffsets)
        # randomly select a point from the inside_hex points. assign a random offset from within that square and
        # then delete it from inside_hex array
        for q in range(0, noffsets):
            rand_num = np.random.rand()
            rand_index_for_squares = int(np.floor(rand_num * num_points_inside_hex))

            if rand_index_for_squares > len(x_center):
                while rand_index_for_squares > len(x_center):
                    rand_num = np.random.rand()
                    rand_index_for_squares = int(np.floor(rand_num * num_points_inside_hex))
            rand_nums = np.random.rand(2)
            rand_x_offset = (rand_nums[0] - 0.5) * (tile_side / 2.0)  # subtract 0.5 to get +/- delta
            rand_y_offset = (rand_nums[1] - 0.5) * (tile_side / 2.0)

            new_x = x_center[rand_index_for_squares] + rand_x_offset
            new_y = y_center[rand_index_for_squares] + rand_y_offset

            # make sure the offset is within the hexagon
            good_condition = (
                (new_y <= m * new_x + b)
                & (new_y >= m * new_x - b)
                & (new_y <= -m * new_x + b)
                & (new_y >= -m * new_x - b)
                & (new_y <= h)
                & (new_y >= -h)
            )
            if not (good_condition):
                while not (good_condition):
                    rand_nums = np.random.rand(2)
                    rand_x_offset = (rand_nums[0] - 0.5) * (tile_side / 2.0)  # subtract 0.5 to get +/- delta
                    rand_y_offset = (rand_nums[1] - 0.5) * (tile_side / 2.0)

                    new_x = x_center[rand_index_for_squares] + rand_x_offset
                    new_y = y_center[rand_index_for_squares] + rand_y_offset

                    good_condition = (
                        (new_y <= m * new_x + b)
                        & (new_y >= m * new_x - b)
                        & (new_y <= -m * new_x + b)
                        & (new_y >= -m * new_x - b)
                        & (new_y <= h)
                        & (new_y >= -h)
                    )

            x_off[q] = x_center[rand_index_for_squares] + rand_x_offset
            y_off[q] = y_center[rand_index_for_squares] + rand_y_offset

            if len(x_center) == 0:
                # have used all the squares ones
                print("Starting reuse of the squares inside the hexagon")
                x_center = x_center_copy.copy()
                y_center = y_center_copy.copy()
            x_center.pop(rand_index_for_squares)
            y_center.pop(rand_index_for_squares)
            num_points_inside_hex -= 1

        self.x_off = x_off
        self.y_off = y_off

    def _run(self, sim_data):
        # Generate random numbers for dither, using defined seed value if desired.
        if self.random_seed is not None:
            np.random.seed(self.random_seed)

        # analysis is simplified if deal with each field separately.
        for fieldid in np.unique(sim_data[self.field_id_col]):
            # Identify observations of this field.
            match = np.where(sim_data[self.field_id_col] == fieldid)[0]
            noffsets = len(match)
            num_tiles = np.ceil(np.sqrt(noffsets) * 1.5) ** 2  # number of tiles must be a perfect square.
            # arbitarily chosen factor of 1.5 to have more than necessary tiles inside hexagon.
            self._generate_rep_random_offsets(noffsets, num_tiles)
            # Add to RA and dec values.
            sim_data["repulsiveRandomDitherFieldPerVisitRa"][match] = sim_data[self.ra_col][
                match
            ] + self.x_off / np.cos(sim_data[self.dec_col][match])
            sim_data["repulsiveRandomDitherFieldPerVisitDec"][match] = (
                sim_data[self.dec_col][match] + self.y_off
            )

        # Wrap back into expected range.
        (
            sim_data["repulsiveRandomDitherFieldPerVisitRa"],
            sim_data["repulsiveRandomDitherFieldPerVisitDec"],
        ) = wrap_ra_dec(
            sim_data["repulsiveRandomDitherFieldPerVisitRa"],
            sim_data["repulsiveRandomDitherFieldPerVisitDec"],
        )
        return sim_data


class RepulsiveRandomDitherFieldPerNightStacker(RepulsiveRandomDitherFieldPerVisitStacker):
    """
    Repulsive-randomly dither the RA and Dec pointings up to max_dither degrees from center, one dither offset
    per new night of observation of a field.

    Note: dithers are confined to the hexagon inscribed in the circle with with radius max_dither

    Parameters
    -------------------
    ra_col: str
        name of the RA column in the data. Default: 'fieldRA'.
    dec_col : str
        name of the Dec column in the data. Default: 'fieldDec'.
    field_id_col : str
        name of the fieldID column in the data. Default: 'field_id_col'.
    night_col : str
        name of the night column in the data. Default: 'night'.
    max_dither: float
        radius of the maximum dither offset, in degrees. Default: 1.75
    random_seed: int
        random seed for the numpy random number generation for the dither offsets.
        Default: None.
    print_info: `bool`
        set to True to print out information about the number of squares considered,
        number of points chosen, and the filling factor. Default: False
    """

    def __init__(
        self,
        ra_col="fieldRA",
        dec_col="fieldDec",
        field_id_col="fieldID",
        night_col="night",
        max_dither=1.75,
        random_seed=None,
        print_info=False,
    ):
        # Instantiate the RandomDither object and set internal variables.
        super(RepulsiveRandomDitherFieldPerNightStacker, self).__init__(
            ra_col=ra_col,
            dec_col=dec_col,
            field_id_col=field_id_col,
            max_dither=max_dither,
            random_seed=random_seed,
            print_info=print_info,
        )
        self.night_col = night_col
        # Values required for framework operation: this specifies the names of the new columns.
        self.cols_added = [
            "repulsiveRandomDitherFieldPerNightRa",
            "repulsiveRandomDitherFieldPerNightDec",
        ]
        # Values required for framework operation: this specifies the data columns required from the database.
        self.cols_req.append(self.night_col)

    def _run(self, sim_data):
        # Generate random numbers for dither, using defined seed value if desired.
        if self.random_seed is not None:
            np.random.seed(self.random_seed)

        for fieldid in np.unique(sim_data[self.field_id_col]):
            # Identify observations of this field.
            match = np.where(sim_data[self.field_id_col] == fieldid)[0]

            noffsets = len(match)
            num_tiles = np.ceil(np.sqrt(noffsets) * 1.5) ** 2
            self._generate_rep_random_offsets(noffsets, num_tiles)

            # Apply dithers, increasing each night.
            vertex_idxs = np.arange(0, len(match), 1)
            nights = sim_data[self.night_col][match]
            vertex_idxs = np.searchsorted(np.unique(nights), nights)
            vertex_idxs = vertex_idxs % len(self.x_off)

            sim_data["repulsiveRandomDitherFieldPerNightRa"][match] = sim_data[self.ra_col][
                match
            ] + self.x_off[vertex_idxs] / np.cos(sim_data[self.dec_col][match])
            sim_data["repulsiveRandomDitherFieldPerNightDec"][match] = (
                sim_data[self.dec_col][match] + self.y_off[vertex_idxs]
            )
        # Wrap into expected range.
        (
            sim_data["repulsiveRandomDitherFieldPerNightRa"],
            sim_data["repulsiveRandomDitherFieldPerNightDec"],
        ) = wrap_ra_dec(
            sim_data["repulsiveRandomDitherFieldPerNightRa"],
            sim_data["repulsiveRandomDitherFieldPerNightDec"],
        )
        return sim_data


class RepulsiveRandomDitherPerNightStacker(RepulsiveRandomDitherFieldPerVisitStacker):
    """
    Repulsive-randomly dither the RA and Dec pointings up to max_dither degrees from center, one dither offset
    per night for all the fields.

    Note: dithers are confined to the hexagon inscribed in the circle with with radius max_dither

    Parameters
    -------------------
    ra_col: str
        name of the RA column in the data. Default: 'fieldRA'.
    dec_col : str
        name of the Dec column in the data. Default: 'fieldDec'.
    night_col : str
        name of the night column in the data. Default: 'night'.
    max_dither: float
        radius of the maximum dither offset, in degrees. Default: 1.75
    random_seed: int
        random seed for the numpy random number generation for the dither offsets.
        Default: None.
    print_info: `bool`
        set to True to print out information about the number of squares considered,
        number of points chosen, and the filling factor. Default: False
    """

    def __init__(
        self,
        ra_col="fieldRA",
        dec_col="fieldDec",
        night_col="night",
        max_dither=1.75,
        random_seed=None,
        print_info=False,
    ):
        # Instantiate the RepulsiveRandomDitherFieldPerVisitStacker object and set internal variables.
        super(RepulsiveRandomDitherPerNightStacker, self).__init__(
            ra_col=ra_col,
            dec_col=dec_col,
            max_dither=max_dither,
            random_seed=random_seed,
            print_info=print_info,
        )
        self.night_col = night_col
        # Values required for framework operation: this specifies the names of the new columns.
        self.cols_added = [
            "repulsiveRandomDitherPerNightRa",
            "repulsiveRandomDitherPerNightDec",
        ]
        # Values required for framework operation: this specifies the data columns required from the database.
        self.cols_req.append(self.night_col)

    def _run(self, sim_data):
        # Generate random numbers for dither, using defined seed value if desired.
        if self.random_seed is not None:
            np.random.seed(self.random_seed)

        # Generate the random dither values, one per night.
        nights = np.unique(sim_data[self.night_col])
        num_nights = len(nights)
        num_tiles = np.ceil(np.sqrt(num_nights) * 1.5) ** 2
        self._generate_rep_random_offsets(num_nights, num_tiles)

        # Add to RA and dec values.
        for n, x, y in zip(nights, self.x_off, self.y_off):
            match = np.where(sim_data[self.night_col] == n)[0]
            sim_data["repulsiveRandomDitherPerNightRa"][match] = sim_data[self.ra_col][match] + x / np.cos(
                sim_data[self.dec_col][match]
            )
            sim_data["repulsiveRandomDitherPerNightDec"][match] = sim_data[self.dec_col][match] + y
        # Wrap RA/Dec into expected range.
        (
            sim_data["repulsiveRandomDitherPerNightRa"],
            sim_data["repulsiveRandomDitherPerNightDec"],
        ) = wrap_ra_dec(
            sim_data["repulsiveRandomDitherPerNightRa"],
            sim_data["repulsiveRandomDitherPerNightDec"],
        )
        return sim_data


class FermatSpiralDitherFieldPerVisitStacker(BaseStacker):
    """
    Offset along a Fermat's spiral with num_points, out to a maximum radius of max_dither.
    Sequential offset for each visit to a field.

    Note: dithers are confined to the hexagon inscribed in the circle with with radius max_dither
    Note: Fermat's spiral is defined by r= c*sqrt(n), theta= n*angle, n= integer

    Parameters
    -------------------
    ra_col: str
        name of the RA column in the data. Default: 'fieldRA'.
    dec_col : str
        name of the Dec column in the data. Default: 'fieldDec'.
    field_id_col : str
        name of the fieldID column in the data. Default: 'fieldID'.
    num_points: int
        number of points in the spiral. Default: 60
    max_dither: float
        radius of the maximum dither offset, in degrees. Default: 1.75
    gold_angle: float
        angle in degrees defining the spiral: theta= multiple of gold_angle
        Default: 137.508

    """

    def __init__(
        self,
        ra_col="fieldRA",
        dec_col="fieldDec",
        field_id_col="fieldID",
        num_points=60,
        max_dither=1.75,
        gold_angle=137.508,
    ):
        self.ra_col = ra_col
        self.dec_col = dec_col
        self.field_id_col = field_id_col
        # Convert max_dither from degrees (internal units for ra/dec are radians)
        self.num_points = num_points
        self.gold_angle = gold_angle
        self.max_dither = np.radians(max_dither)
        # self.units used for plot labels
        self.units = ["rad", "rad"]
        # Values required for framework operation: this specifies the names of the new columns.
        self.cols_added = [
            "fermatSpiralDitherFieldPerVisitRa",
            "fermatSpiralDitherFieldPerVisitDec",
        ]
        # Values required for framework operation: this specifies the data columns required from the database.
        self.cols_req = [self.ra_col, self.dec_col, self.field_id_col]

    def _generate_fermat_spiral_offsets(self):
        # Fermat's spiral: r= c*sqrt(n), theta= n*angle
        # Golden spiral: r= c*sqrt(n), theta= n*137.508degrees
        n = np.arange(0, self.num_points)
        theta = np.radians(n * self.gold_angle)
        rmax = np.sqrt(theta.max() / np.radians(self.gold_angle))
        scaling_factor = 0.8 * self.max_dither / rmax
        r = scaling_factor * np.sqrt(n)

        self.x_off = r * np.cos(theta)
        self.y_off = r * np.sin(theta)

    def _run(self, sim_data):
        # Generate the spiral offset vertices.
        self._generate_fermat_spiral_offsets()
        # Now apply to observations.
        for fieldid in np.unique(sim_data[self.field_id_col]):
            match = np.where(sim_data[self.field_id_col] == fieldid)[0]
            # Apply sequential dithers, increasing with each visit.
            vertex_idxs = np.arange(0, len(match), 1)
            vertex_idxs = vertex_idxs % self.num_points
            sim_data["fermatSpiralDitherFieldPerVisitRa"][match] = sim_data[self.ra_col][match] + self.x_off[
                vertex_idxs
            ] / np.cos(sim_data[self.dec_col][match])
            sim_data["fermatSpiralDitherFieldPerVisitDec"][match] = (
                sim_data[self.dec_col][match] + self.y_off[vertex_idxs]
            )
        # Wrap into expected range.
        (
            sim_data["fermatSpiralDitherFieldPerVisitRa"],
            sim_data["fermatSpiralDitherFieldPerVisitDec"],
        ) = wrap_ra_dec(
            sim_data["fermatSpiralDitherFieldPerVisitRa"],
            sim_data["fermatSpiralDitherFieldPerVisitDec"],
        )
        return sim_data


class FermatSpiralDitherFieldPerNightStacker(FermatSpiralDitherFieldPerVisitStacker):
    """
    Offset along a Fermat's spiral with num_points, out to a maximum radius of max_dither.
    one dither offset  per new night of observation of a field.

    Note: dithers are confined to the hexagon inscribed in the circle with with radius max_dither
    Note: Fermat's spiral is defined by r= c*sqrt(n), theta= n*angle, n= integer

    Parameters
    -----------
    ra_col: str
        name of the RA column in the data. Default: 'fieldRA'.
    dec_col : str
        name of the Dec column in the data. Default: 'fieldDec'.
    field_id_col : str
        name of the fieldID column in the data. Default: 'fieldID'.
    night_col : str
        name of the night column in the data. Default: 'night'.
    num_points: int
        number of points in the spiral. Default: 60
    max_dither: float
        radius of the maximum dither offset, in degrees. Default: 1.75
    gold_angle: float
        angle in degrees defining the spiral: theta= multiple of gold_angle. Default: 137.508
    """

    def __init__(
        self,
        ra_col="fieldRA",
        dec_col="fieldDec",
        field_id_col="fieldID",
        night_col="night",
        num_points=60,
        max_dither=1.75,
        gold_angle=137.508,
    ):
        super(FermatSpiralDitherFieldPerNightStacker, self).__init__(
            ra_col=ra_col,
            dec_col=dec_col,
            field_id_col=field_id_col,
            num_points=num_points,
            max_dither=max_dither,
            gold_angle=gold_angle,
        )
        self.night_col = night_col
        # Values required for framework operation: this specifies the names of the new columns.
        self.cols_added = [
            "fermatSpiralDitherFieldPerNightRa",
            "fermatSpiralDitherFieldPerNightDec",
        ]
        # Values required for framework operation: this specifies the data columns required from the database.
        self.cols_req.append(self.night_col)

    def _run(self, sim_data):
        # Generate the spiral offset vertices.
        self._generate_fermat_spiral_offsets()
        # Now apply to observations.
        for fieldid in np.unique(sim_data[self.field_id_col]):
            match = np.where(sim_data[self.field_id_col] == fieldid)[0]
            # Apply sequential dithers, increasing with each visit.
            vertex_idxs = np.arange(0, len(match), 1)
            nights = sim_data[self.night_col][match]
            vertex_idxs = np.searchsorted(np.unique(nights), nights)
            vertex_idxs = vertex_idxs % self.num_points
            sim_data["fermatSpiralDitherFieldPerNightRa"][match] = sim_data[self.ra_col][match] + self.x_off[
                vertex_idxs
            ] / np.cos(sim_data[self.dec_col][match])
            sim_data["fermatSpiralDitherFieldPerNightDec"][match] = (
                sim_data[self.dec_col][match] + self.y_off[vertex_idxs]
            )
        # Wrap into expected range.
        (
            sim_data["fermatSpiralDitherFieldPerNightRa"],
            sim_data["fermatSpiralDitherFieldPerNightDec"],
        ) = wrap_ra_dec(
            sim_data["fermatSpiralDitherFieldPerNightRa"],
            sim_data["fermatSpiralDitherFieldPerNightDec"],
        )
        return sim_data


class FermatSpiralDitherPerNightStacker(FermatSpiralDitherFieldPerVisitStacker):
    """
    Offset along a Fermat's spiral with num_points, out to a maximum radius of max_dither.
    Sequential offset per night for all fields.

    Note: dithers are confined to the hexagon inscribed in the circle with with radius max_dither
    Note: Fermat's spiral is defined by r= c*sqrt(n), theta= n*angle, n= integer

    Parameters
    ----------
    ra_col: str
        name of the RA column in the data. Default: 'fieldRA'.
    dec_col : str
        name of the Dec column in the data. Default: 'fieldDec'.
    field_id_col : str
        name of the fieldID column in the data. Default: 'fieldID'.
    night_col : str
        name of the night column in the data. Default: 'night'.
    num_points: int
        number of points in the spiral. Default: 60
    max_dither: float
        radius of the maximum dither offset, in degrees. Default: 1.75
    gold_angle: float
        angle in degrees defining the spiral: theta= multiple of gold_angle
        Default: 137.508
    """

    def __init__(
        self,
        ra_col="fieldRA",
        dec_col="fieldDec",
        field_id_col="fieldID",
        night_col="night",
        num_points=60,
        max_dither=1.75,
        gold_angle=137.508,
    ):
        super(FermatSpiralDitherPerNightStacker, self).__init__(
            ra_col=ra_col,
            dec_col=dec_col,
            field_id_col=field_id_col,
            num_points=num_points,
            max_dither=max_dither,
            gold_angle=gold_angle,
        )
        self.night_col = night_col
        # Values required for framework operation: this specifies the names of the new columns.
        self.cols_added = [
            "fermatSpiralDitherPerNightRa",
            "fermatSpiralDitherPerNightDec",
        ]
        # Values required for framework operation: this specifies the data columns required from the database.
        self.cols_req.append(self.night_col)

    def _run(self, sim_data):
        # Generate the spiral offset vertices.
        self._generate_fermat_spiral_offsets()

        vertex_id = 0
        nights = np.unique(sim_data[self.night_col])
        for n in nights:
            match = np.where(sim_data[self.night_col] == n)[0]
            vertex_id = vertex_id % self.num_points

            sim_data["fermatSpiralDitherPerNightRa"][match] = sim_data[self.ra_col][match] + self.x_off[
                vertex_id
            ] / np.cos(sim_data[self.dec_col][match])
            sim_data["fermatSpiralDitherPerNightDec"][match] = (
                sim_data[self.dec_col][match] + self.y_off[vertex_id]
            )
            vertex_id += 1

        # Wrap into expected range.
        (
            sim_data["fermatSpiralDitherPerNightRa"],
            sim_data["fermatSpiralDitherPerNightDec"],
        ) = wrap_ra_dec(
            sim_data["fermatSpiralDitherPerNightRa"],
            sim_data["fermatSpiralDitherPerNightDec"],
        )
        return sim_data


class PentagonDitherFieldPerSeasonStacker(BaseStacker):
    """
    Offset along two pentagons, one inverted and inside the other.
    Sequential offset for each field on a visit in new season.

    Parameters
    -----------
    ra_col: str
        name of the RA column in the data. Default: 'fieldRA'.
    dec_col : str
        name of the Dec column in the data. Default: 'fieldDec'.
    field_id_col : str
        name of the fieldID column in the data. Default: 'fieldID'.
    exp_mjd_col : str
        name of the date/time stamp column in the data. Default: 'expMJD'.
    max_dither: float
        radius of the maximum dither offset, in degrees. Default: 1.75
    wrap_last_season: `bool`
        set to False to all consider 11 seasons independently.
        set to True to wrap 0th and 10th season, leading to a total of 10 seasons.
        Default: True
    """

    def __init__(
        self,
        ra_col="fieldRA",
        dec_col="fieldDec",
        field_id_col="fieldID",
        exp_mjd_col="expMJD",
        max_dither=1.75,
        wrap_last_season=True,
    ):
        self.ra_col = ra_col
        self.dec_col = dec_col
        self.field_id_col = field_id_col
        self.exp_mjd_col = exp_mjd_col
        # Convert max_dither from degrees (internal units for ra/dec are radians)
        self.max_dither = np.radians(max_dither)
        self.wrap_last_season = wrap_last_season
        # self.units used for plot labels
        self.units = ["rad", "rad"]
        # Values required for framework operation: this specifies the names of the new columns.
        self.cols_added = [
            "pentagonDitherFieldPerSeasonRa",
            "pentagonDitherFieldPerSeasonDec",
        ]
        # Values required for framework operation: this specifies the data columns required from the database.
        self.cols_req = [self.ra_col, self.dec_col, self.field_id_col, self.exp_mjd_col]

    def _generate_pentagon_offsets(self):
        # inner pentagon tuples
        nside = 5
        inner = polygon_coords(nside, self.max_dither / 2.5, 0.0)
        # outer pentagon tuples
        outer_temp = polygon_coords(nside, self.max_dither / 1.3, np.pi)
        # reorder outer tuples' order
        outer = []
        outer[0:3] = outer_temp[2:5]
        outer[4:6] = outer_temp[0:2]
        # join inner and outer coordiantes' array
        self.x_off = np.concatenate((zip(*inner)[0], zip(*outer)[0]), axis=0)
        self.y_off = np.concatenate((zip(*inner)[1], zip(*outer)[1]), axis=0)

    def _run(self, sim_data):
        # find the seasons associated with each visit.
        seasons = calc_season(sim_data[self.ra_col], simdata[self.exp_mjd_col])
        # check how many entries in the >10 season
        ind = np.where(seasons > 9)[0]
        # should be only 1 extra seasons ..
        if len(np.unique(seasons[ind])) > 1:
            raise ValueError("Too many seasons (more than 11). Check SeasonStacker.")

        if self.wrap_last_season:
            print("Seasons to wrap ", np.unique(seasons[ind]))
            # wrap the season around: 10th == 0th
            seasons[ind] = seasons[ind] % 10

        # Generate the spiral offset vertices.
        self._generate_pentagon_offsets()

        # Now apply to observations.
        for fieldid in np.unique(sim_data[self.field_id_col]):
            match = np.where(sim_data[self.field_id_col] == fieldid)[0]
            seasons_visited = seasons[match]
            # Apply sequential dithers, increasing with each season.
            vertex_idxs = np.searchsorted(np.unique(seasons_visited), seasons_visited)
            vertex_idxs = vertex_idxs % len(self.x_off)
            sim_data["pentagonDitherFieldPerSeasonRa"][match] = sim_data[self.ra_col][match] + self.x_off[
                vertex_idxs
            ] / np.cos(sim_data[self.dec_col][match])
            sim_data["pentagonDitherFieldPerSeasonDec"][match] = (
                sim_data[self.dec_col][match] + self.y_off[vertex_idxs]
            )
        # Wrap into expected range.
        (
            sim_data["pentagonDitherFieldPerSeasonRa"],
            sim_data["pentagonDitherFieldPerSeasonDec"],
        ) = wrap_ra_dec(
            sim_data["pentagonDitherFieldPerSeasonRa"],
            sim_data["pentagonDitherFieldPerSeasonDec"],
        )
        return sim_data


class PentagonDiamondDitherFieldPerSeasonStacker(BaseStacker):
    """
    Offset along a diamond circumscribed by a pentagon.
    Sequential offset for each field on a visit in new season.

    Parameters
    -------------------
    ra_col: str
        name of the RA column in the data. Default: 'fieldRA'.
    dec_col : str
        name of the Dec column in the data. Default: 'fieldDec'.
    field_id_col : str
        name of the fieldID column in the data. Default: 'fieldID'.
    exp_mjd_col : str
        name of the date/time stamp column in the data. Default: 'expMJD'.
    max_dither: float
        radius of the maximum dither offset, in degrees. Default: 1.75
    wrap_last_season: `bool`
        set to False to all consider 11 seasons independently.
        set to True to wrap 0th and 10th season, leading to a total of 10 seasons.
        Default: True
    """

    def __init__(
        self,
        ra_col="fieldRA",
        dec_col="fieldDec",
        field_id_col="fieldID",
        exp_mjd_col="expMJD",
        max_dither=1.75,
        wrap_last_season=True,
    ):
        self.ra_col = ra_col
        self.dec_col = dec_col
        self.field_id_col = field_id_col
        self.exp_mjd_col = exp_mjd_col
        # Convert max_dither from degrees (internal units for ra/dec are radians)
        self.max_dither = np.radians(max_dither)
        self.wrap_last_season = wrap_last_season
        # self.units used for plot labels
        self.units = ["rad", "rad"]
        # Values required for framework operation: this specifies the names of the new columns.
        self.cols_added = [
            "pentagonDiamondDitherFieldPerSeasonRa",
            "pentagonDiamondDitherFieldPerSeasonDec",
        ]
        # Values required for framework operation: this specifies the data columns required from the database.
        self.cols_req = [self.ra_col, self.dec_col, self.field_id_col, self.exp_mjd_col]

    def _generate_offsets(self):
        # outer pentagon tuples
        pent_coord = polygon_coords(5, self.max_dither / 1.3, 0)
        # inner diamond tuples
        diamond_coord = polygon_coords(4, self.max_dither / 2.5, np.pi / 2)

        # join inner and outer coordiantes' array + a point in the middle (origin)
        self.x_off = np.concatenate(([0], zip(*diamond_coord)[0], zip(*pent_coord)[0]), axis=0)
        self.y_off = np.concatenate(([0], zip(*diamond_coord)[1], zip(*pent_coord)[1]), axis=0)

    def _run(self, sim_data):
        # find the seasons associated with each visit.
        seasons = calc_season(sim_data[self.ra_col], sim_data[self.exp_mjd_col])

        # check how many entries in the >10 season
        ind = np.where(seasons > 9)[0]
        # should be only 1 extra seasons ..
        if len(np.unique(seasons[ind])) > 1:
            raise ValueError("Too many seasons (more than 11). Check SeasonStacker.")

        if self.wrap_last_season:
            print("Seasons to wrap ", np.unique(seasons[ind]))
            # wrap the season around: 10th == 0th
            seasons[ind] = seasons[ind] % 10

        # Generate the spiral offset vertices.
        self._generate_offsets()

        # Now apply to observations.
        for fieldid in np.unique(sim_data[self.field_id_col]):
            match = np.where(sim_data[self.field_id_col] == fieldid)[0]
            seasons_visited = seasons[match]
            # Apply sequential dithers, increasing with each season.
            vertex_idxs = np.searchsorted(np.unique(seasons_visited), seasons_visited)
            vertex_idxs = vertex_idxs % len(self.x_off)
            sim_data["pentagonDiamondDitherFieldPerSeasonRa"][match] = sim_data[self.ra_col][
                match
            ] + self.x_off[vertex_idxs] / np.cos(sim_data[self.dec_col][match])
            sim_data["pentagonDiamondDitherFieldPerSeasonDec"][match] = (
                sim_data[self.dec_col][match] + self.y_off[vertex_idxs]
            )
        # Wrap into expected range.
        (
            sim_data["pentagonDiamondDitherFieldPerSeasonRa"],
            sim_data["pentagonDiamondDitherFieldPerSeasonDec"],
        ) = wrap_ra_dec(
            sim_data["pentagonDiamondDitherFieldPerSeasonRa"],
            sim_data["pentagonDiamondDitherFieldPerSeasonDec"],
        )
        return sim_data


class PentagonDitherPerSeasonStacker(PentagonDitherFieldPerSeasonStacker):
    """
    Offset along two pentagons, one inverted and inside the other.
    Sequential offset for all fields every season.

    Parameters
    -------------------
    ra_col: str
        name of the RA column in the data. Default: 'fieldRA'.
    dec_col : str
        name of the Dec column in the data. Default: 'fieldDec'.
    field_id_col : str
        name of the fieldID column in the data. Default: 'fieldID'.
    exp_mjd_col : str
        name of the date/time stamp column in the data. Default: 'expMJD'.
    max_dither: float
        radius of the maximum dither offset, in degrees. Default: 1.75
    wrap_last_season: `bool`
        set to False to all consider 11 seasons independently.
        set to True to wrap 0th and 10th season, leading to a total of 10 seasons.
        Default: True
    """

    def __init__(
        self,
        ra_col="fieldRA",
        dec_col="fieldDec",
        field_id_col="fieldID",
        exp_mjd_col="expMJD",
        night_col="night",
        max_dither=1.75,
        wrap_last_season=True,
    ):
        super(PentagonDitherPerSeasonStacker, self).__init__(
            ra_col=ra_col,
            dec_col=dec_col,
            field_id_col=field_id_col,
            exp_mjd_col=exp_mjd_col,
            max_dither=max_dither,
            wrap_last_season=wrap_last_season,
        )
        # Values required for framework operation: this specifies the names of the new columns.
        self.cols_added = ["pentagonDitherPerSeasonRa", "pentagonDitherPerSeasonDec"]

    def _run(self, sim_data):
        # find the seasons associated with each visit.
        seasons = calc_season(sim_data[self.ra_col], sim_data[self.exp_mjd_col])
        years = sim_data[self.nightCol] % 365.25

        # check how many entries in the >10 season
        ind = np.where(seasons > 9)[0]
        # should be only 1 extra seasons ..
        if len(np.unique(seasons[ind])) > 1:
            raise ValueError("Too many seasons (more than 11). Check SeasonStacker.")

        if self.wrap_last_season:
            # check how many entries in the >10 season
            print(
                "Seasons to wrap ",
                np.unique(seasons[ind]),
                "with total entries: ",
                len(seasons[ind]),
            )
            seasons[ind] = seasons[ind] % 10

        # Generate the spiral offset vertices.
        self._generate_pentagon_offsets()
        # print details
        print("Total visits for all fields:", len(seasons))
        print("")

        # Add to RA and dec values.
        vertex_id = 0
        for s in np.unique(seasons):
            match = np.where(seasons == s)[0]
            # print details
            print("season", s)
            print(
                "numEntries ",
                len(match),
                "; ",
                float(len(match)) / len(seasons) * 100,
                "% of total",
            )
            match_years = np.unique(years[match])
            print("Corresponding years: ", match_years)
            for i in match_years:
                print("     Entries in year", i, ": ", len(np.where(i == years[match])[0]))
            print("")
            vertex_id = vertex_id % len(self.x_off)
            sim_data["pentagonDitherPerSeasonRa"][match] = sim_data[self.ra_col][match] + self.x_off[
                vertex_id
            ] / np.cos(sim_data[self.dec_col][match])
            sim_data["pentagonDitherPerSeasonDec"][match] = (
                sim_data[self.dec_col][match] + self.y_off[vertex_id]
            )
            vertex_id += 1

        # Wrap into expected range.
        (
            sim_data["pentagonDitherPerSeasonRa"],
            sim_data["pentagonDitherPerSeasonDec"],
        ) = wrap_ra_dec(
            sim_data["pentagonDitherPerSeasonRa"],
            sim_data["pentagonDitherPerSeasonDec"],
        )
        return sim_data


class PentagonDiamondDitherPerSeasonStacker(PentagonDiamondDitherFieldPerSeasonStacker):
    """
    Offset along a diamond circumscribed by a pentagon.
    Sequential offset for all fields every season.

    Parameters
    -------------------
    ra_col: str
        name of the RA column in the data. Default: 'fieldRA'.
    dec_col : str
        name of the Dec column in the data. Default: 'fieldDec'.
    field_id_col : str
        name of the fieldID column in the data. Default: 'fieldID'.
    exp_mjd_col : str
        name of the date/time stamp column in the data. Default: 'expMJD'.
    max_dither: float
        radius of the maximum dither offset, in degrees. Default: 1.75
    wrap_last_season: `bool`
        set to False to all consider 11 seasons independently.
        set to True to wrap 0th and 10th season, leading to a total of 10 seasons.
        Default: True
    """

    def __init__(
        self,
        ra_col="fieldRA",
        dec_col="fieldDec",
        field_id_col="fieldID",
        exp_mjd_col="expMJD",
        max_dither=1.75,
        wrap_last_season=True,
    ):
        super(PentagonDiamondDitherPerSeasonStacker, self).__init__(
            ra_col=ra_col,
            dec_col=dec_col,
            field_id_col=field_id_col,
            exp_mjd_col=exp_mjd_col,
            max_dither=max_dither,
            wrap_last_season=wrap_last_season,
        )
        # Values required for framework operation: this specifies the names of the new columns.
        self.cols_added = [
            "pentagonDiamondDitherPerSeasonRa",
            "pentagonDiamondDitherPerSeasonDec",
        ]

    def _run(self, sim_data):
        # find the seasons associated with each visit.
        seasons = calc_season(sim_data[self.ra_col], sim_data[self.exp_mjd_col])

        # check how many entries in the >10 season
        ind = np.where(seasons > 9)[0]
        # should be only 1 extra seasons ..
        if len(np.unique(seasons[ind])) > 1:
            raise ValueError("Too many seasons (more than 11). Check SeasonStacker.")

        if self.wrap_last_season:
            print("Seasons to wrap ", np.unique(seasons[ind]))
            # wrap the season around: 10th == 0th
            seasons[ind] = seasons[ind] % 10

        # Generate the spiral offset vertices.
        self._generate_offsets()

        uniq_seasons = np.unique(seasons)
        # Add to RA and dec values.
        vertex_id = 0
        for s in uniq_seasons:
            match = np.where(seasons == s)[0]
            vertex_id = vertex_id % len(self.x_off)
            sim_data["pentagonDiamondDitherPerSeasonRa"][match] = sim_data[self.ra_col][match] + self.x_off[
                vertex_id
            ] / np.cos(sim_data[self.dec_col][match])
            sim_data["pentagonDiamondDitherPerSeasonDec"][match] = (
                sim_data[self.dec_col][match] + self.y_off[vertex_id]
            )
            vertex_id += 1

        # Wrap into expected range.
        (
            sim_data["pentagonDiamondDitherPerSeasonRa"],
            sim_data["pentagonDiamondDitherPerSeasonDec"],
        ) = wrap_ra_dec(
            sim_data["pentagonDiamondDitherPerSeasonRa"],
            sim_data["pentagonDiamondDitherPerSeasonDec"],
        )
        return sim_data


class SpiralDitherPerSeasonStacker(SpiralDitherFieldPerVisitStacker):
    """
    Offsets along a 10pt spiral. Sequential offset for all fields every seaso along a 10pt spiral.

    Parameters
    -------------------
    raCol: str
        name of the RA column in the data. Default: 'fieldRA'.
    decCol : str
        name of the Dec column in the data. Default: 'fieldDec'.
    fieldIdCol : str
        name of the fieldID column in the data. Default: 'fieldID'.
    exp_mjd_col : str
        name of the date/time stamp column in the data. Default: 'expMJD'.
    maxDither: float
        radius of the maximum dither offset, in degrees. Default: 1.75
    wrap_last_season: `bool`
        set to False to all consider 11 seasons independently.
        set to True to wrap 0th and 10th season, leading to a total of 10 seasons.
        Default: True
    numPoints: int:  number of points in the spiral. Default: 10
    nCoils: int:  number of coils the spiral. Default: 3
    """

    def __init__(
        self,
        ra_col="fieldRA",
        dec_col="fieldDec",
        field_id_col="fieldID",
        exp_mjd_col="expMJD",
        max_dither=1.75,
        wrap_last_season=True,
        num_points=10,
        n_coils=3,
    ):
        super(SpiralDitherPerSeasonStacker, self).__init__(
            ra_col=ra_col,
            dec_col=dec_col,
            field_id_col=field_id_col,
            n_coils=n_coils,
            num_points=num_points,
            max_dither=max_dither,
        )
        self.exp_mjd_col = exp_mjd_col
        self.wrap_last_season = wrap_last_season
        # Values required for framework operation: this specifies the names of the new columns.
        self.cols_added = ["spiralDitherPerSeasonRa", "spiralDitherPerSeasonDec"]
        self.cols_req.append(self.exp_mjd_col)

    def _run(self, sim_data):
        # find the seasons associated with each visit.
        seasons = calc_season(sim_data[self.raCol], sim_data[self.exp_mjd_col])

        # check how many entries in the >10 season
        ind = np.where(seasons > 9)[0]
        # should be only 1 extra seasons ..
        if len(np.unique(seasons[ind])) > 1:
            raise ValueError("Too many seasons (more than 11). Check SeasonStacker.")

        if self.wrap_last_season:
            print("Seasons to wrap ", np.unique(seasons[ind]))
            # wrap the season around: 10th == 0th
            seasons[ind] = seasons[ind] % 10

        # Generate the spiral offset vertices.
        self._generateSpiralOffsets()

        # Add to RA and dec values.
        vertex_id = 0
        for s in np.unique(seasons):
            match = np.where(seasons == s)[0]
            vertex_id = vertex_id % self.numPoints
            sim_data["spiralDitherPerSeasonRa"][match] = sim_data[self.raCol][match] + self.xOff[
                vertex_id
            ] / np.cos(sim_data[self.decCol][match])
            sim_data["spiralDitherPerSeasonDec"][match] = sim_data[self.decCol][match] + self.yOff[vertex_id]
            vertex_id += 1

        # Wrap into expected range.
        (
            sim_data["spiralDitherPerSeasonRa"],
            sim_data["spiralDitherPerSeasonDec"],
        ) = wrap_ra_dec(sim_data["spiralDitherPerSeasonRa"], sim_data["spiralDitherPerSeasonDec"])
        return sim_data
