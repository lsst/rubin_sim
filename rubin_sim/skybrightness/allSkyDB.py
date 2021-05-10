import numpy as np
import sqlalchemy as sqla
import os
from rubin_sim.data import get_data_dir

# Tools for using an all-sky sqlite DB with cannon and photodiode data from the site.


def allSkyDB(dateID, sqlQ=None, dtypes=None, dbAddress=None, filt='R'):
    """
    Take in a dateID (that corresponds to a single MJD, and
    return the star and sky magnitudes in a numpy structured array.
    """
    if dbAddress is None:
        dataPath = os.path.join(get_data_dir(), 'skybrightness')
        dbAddress = 'sqlite:///'+os.path.join(dataPath, 'photometry', 'skydata.sqlite')
    if sqlQ is None:
        sqlQ = 'select stars.ra, stars.dec,  obs.alt, obs.starMag, obs.sky, obs.filter from obs, stars where obs.starID = stars.ID and obs.filter = "%s" and obs.dateID = %i;' % (
            filt, dateID)
    if dtypes is None:
        names = ['ra', 'dec', 'alt', 'starMag', 'sky', 'filter']
        types = [float, float, float, float, float, '|S1']
        dtypes = list(zip(names, types))

    engine = sqla.create_engine(dbAddress)
    connection = engine.raw_connection()
    cursor = connection.cursor()
    cursor.execute(sqlQ)
    data = cursor.fetchall()
    data = np.asarray(data, dtype=dtypes)

    q2 = 'select mjd from dates where ID = %i' % dateID
    cursor.execute(q2)

    mjd = cursor.fetchall()
    if len(mjd) == 0:
        mjd = None
    else:
        mjd = mjd[0][0]
    return data, mjd


def diodeSkyDB(midMJD, sqlQ=None, dtypes=None, dbAddress=None, clean=True):
    if dbAddress is None:
        dataPath = os.getenv('SIMS_SKYBRIGHTNESS_DATA_DIR')
        dbAddress = 'sqlite:///'+os.path.join(dataPath, 'photometry', 'skydata.sqlite')
    if sqlQ is None:
        sqlQ = 'select mjd, R, Y, Z from photdiode where mjd > %f-1 and  mjd < %f+1' % (midMJD, midMJD)
    if dtypes is None:
        names = ['mjd', 'r', 'y', 'z']
        types = [float]*4
        dtypes = list(zip(names, types))

    engine = sqla.create_engine(dbAddress)
    connection = engine.raw_connection()
    cursor = connection.cursor()
    cursor.execute(sqlQ)
    data = cursor.fetchall()
    data = np.asarray(data, dtype=dtypes)

    if clean:
        data = data[np.where((data['r'] > 0) & (data['z'] > 0) &
                             (data['y'] > 0))]

    return data
