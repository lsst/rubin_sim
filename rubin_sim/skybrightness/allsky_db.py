import numpy as np
import sqlalchemy as sqla
import os
from rubin_sim.data import get_data_dir

# Tools for using an all-sky sqlite DB with cannon and photodiode data from the site.

__all__ = ["all_sky_db", "diode_sky_db"]


def all_sky_db(date_id, sql_q=None, dtypes=None, db_address=None, filt="R"):
    """
    Take in a date_id (that corresponds to a single MJD, and
    return the star and sky magnitudes in a numpy structured array.
    """
    if db_address is None:
        data_path = os.path.join(get_data_dir(), "skybrightness")
        db_address = "sqlite:///" + os.path.join(
            data_path, "photometry", "skydata.sqlite"
        )
    if sql_q is None:
        sql_q = (
            "select stars.ra, stars.dec,  obs.alt, obs.starMag, obs.sky, obs.filter from obs, "
            'stars where obs.starID = stars.ID and obs.filter = "%s" and obs.dateID = %i;'
            % (filt, date_id)
        )
    if dtypes is None:
        names = ["ra", "dec", "alt", "starMag", "sky", "filter"]
        types = [float, float, float, float, float, "|S1"]
        dtypes = list(zip(names, types))

    engine = sqla.create_engine(db_address)
    connection = engine.raw_connection()
    cursor = connection.cursor()
    cursor.execute(sql_q)
    data = cursor.fetchall()
    data = np.asarray(data, dtype=dtypes)

    q2 = "select mjd from dates where ID = %i" % date_id
    cursor.execute(q2)

    mjd = cursor.fetchall()
    if len(mjd) == 0:
        mjd = None
    else:
        mjd = mjd[0][0]
    return data, mjd


def diode_sky_db(mid_mjd, sql_q=None, dtypes=None, db_address=None, clean=True):
    if db_address is None:
        data_path = os.getenv("SIMS_SKYBRIGHTNESS_DATA_DIR")
        db_address = "sqlite:///" + os.path.join(
            data_path, "photometry", "skydata.sqlite"
        )
    if sql_q is None:
        sql_q = (
            "select mjd, R, Y, Z from photdiode where mjd > %f-1 and  mjd < %f+1"
            % (
                mid_mjd,
                mid_mjd,
            )
        )
    if dtypes is None:
        names = ["mjd", "r", "y", "z"]
        types = [float] * 4
        dtypes = list(zip(names, types))

    engine = sqla.create_engine(db_address)
    connection = engine.raw_connection()
    cursor = connection.cursor()
    cursor.execute(sql_q)
    data = cursor.fetchall()
    data = np.asarray(data, dtype=dtypes)

    if clean:
        data = data[np.where((data["r"] > 0) & (data["z"] > 0) & (data["y"] > 0))]

    return data
