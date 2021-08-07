# Collection of utilities for MAF that relate to Opsim specifically.

import os
import shutil
import numpy as np
import pandas as pd
import sqlite3
from sqlite3 import OperationalError, IntegrityError

from .outputUtils import printDict

__all__ = ['writeConfigs', 'getFieldData', 'getSimData',
           'scaleBenchmarks', 'calcCoaddedDepth', 'labelVisits']


def writeConfigs(opsimDb, outDir):
    """
    Convenience function to get the configuration information from the opsim database and write
    this information to text files 'configSummary.txt' and 'configDetails.txt'.

    Parameters
    ----------
    opsimDb : `rubin_sim.maf.db.OpsimDatabase`
        The opsim database from which to pull the opsim configuration information.
        Opsim SQLite databases save this configuration information in their config table.
    outputDir : `str`
        The path to the output directory, where to write the config*.txt files.
    """
    configSummary, configDetails = opsimDb.fetchConfig()
    outfile = os.path.join(outDir, 'configSummary.txt')
    f = open(outfile, 'w')
    printDict(configSummary, 'Summary', f)
    f.close()
    outfile = os.path.join(outDir, 'configDetails.txt')
    f = open(outfile, 'w')
    printDict(configDetails, 'Details', f)
    f.close()


def getFieldData(opsimDb, sqlconstraint):
    """
    Find the fields (ra/dec/fieldID) relevant for a given sql constraint.
    If the opsimDb contains a Fields table, it uses
    :meth:`rubin_sim.maf.OpsimDatabase.fetchFieldsFromFieldTable()`
    to get the fields. If the opsimDb contains only a Summary, it uses
    :meth:`rubin_sim.maf.OpsimDatabase.fetchFieldsFromSummaryTable()`.

    Parameters
    ----------
    opsimDb : `rubin_sim.maf.db.OpsimDatabase`
        An opsim database to use to query for field information.
    sqlconstraint : `str`
        A SQL constraint to apply to the query (i.e. find all fields for DD proposal)

    Returns
    -------
    numpy.ndarray
        A numpy structured array containing the field information.  This data will ALWAYS be in radians.
    """
    # Get all fields used for all proposals.
    if 'proposalId' not in sqlconstraint:
        propids, propTags = opsimDb.fetchPropInfo()
        propids = list(propids.keys())
    else:
        # Parse the propID out of the sqlconstraint.
        # example: sqlconstraint: filter = r and (propid = 219 or propid = 155) and propid!= 90
        sqlconstraint = sqlconstraint.replace('=', ' = ').replace('(', '').replace(')', '')
        sqlconstraint = sqlconstraint.replace("'", '').replace('"', '')
        # Allow for choosing all but a particular proposal.
        sqlconstraint = sqlconstraint.replace('! =', ' !=')
        sqlconstraint = sqlconstraint.replace('  ', ' ')
        sqllist = sqlconstraint.split(' ')
        propids = []
        nonpropids = []
        i = 0
        while i < len(sqllist):
            if sqllist[i].lower() == 'proposalid':
                i += 1
                if sqllist[i] == "=":
                    i += 1
                    propids.append(int(sqllist[i]))
                elif sqllist[i] == '!=':
                    i += 1
                    nonpropids.append(int(sqllist[i]))
            i += 1
        if len(propids) == 0:
            propids, propTags = opsimDb.fetchPropInfo()
            propids = list(propids.keys())
        if len(nonpropids) > 0:
            for nonpropid in nonpropids:
                if nonpropid in propids:
                    propids.remove(nonpropid)
    # And query the field Table.
    if 'Field' in opsimDb.tables:
        # The field table is always in degrees.
        fieldData = opsimDb.fetchFieldsFromFieldTable(propids, degreesToRadians=True)
    # Or give up and query the summary table.
    else:
        fieldData = opsimDb.fetchFieldsFromSummaryTable(sqlconstraint)
    return fieldData


def getSimData(opsimDb, sqlconstraint, dbcols, stackers=None, groupBy='default', tableName=None):
    """Query an opsim database for the needed data columns and run any required stackers.

    Parameters
    ----------
    opsimDb : `rubin_sim.maf.db.OpsimDatabase`
    sqlconstraint : `str`
        SQL constraint to apply to query for observations.
    dbcols : `list` [`str`]
        Columns required from the database.
    stackers : `list` [`rubin_sim.maf.Stackers`], optional
        Stackers to be used to generate additional columns. Default None.
    tableName : `str`, optional
        Name of the table to query. Default None uses the opsimDb default.
    groupBy : `str`, optional
        Column name to group SQL results by.  Default uses the opsimDb default.

    Returns
    -------
    simData: `np.ndarray`
        A numpy structured array with columns resulting from dbcols + stackers, for observations matching
        the SQLconstraint.
    """
    # Get data from database.
    simData = opsimDb.fetchMetricData(dbcols, sqlconstraint, groupBy=groupBy, tableName=tableName)
    if len(simData) == 0:
        raise UserWarning('No data found matching sqlconstraint %s' % (sqlconstraint))
    # Now add the stacker columns.
    if stackers is not None:
        for s in stackers:
            simData = s.run(simData)
    return simData


def scaleBenchmarks(runLength, benchmark='design'):
    """
    Set the design and stretch values of the number of visits, area of the footprint,
    seeing values, FWHMeff values, skybrightness, and single visit depth (based on SRD values).
    Scales number of visits for the length of the run, relative to 10 years.

    Parameters
    ----------
    runLength : float
        The length (in years) of the run.
    benchmark : str
        design or stretch - which version of the SRD values to return.
        requested is another option, in which case the values of the number of visits requested
        by the OpSim run (recorded in the Config table) is returned.

    Returns
    -------
    benchmarks: `dict` of floats
       A dictionary containing the number of visits, area of footprint, seeing and FWHMeff values,
       skybrightness and single visit depth for either the design or stretch SRD values.
    """
    # Set baseline (default) numbers for the baseline survey length (10 years).
    baseline = 10.

    design = {}
    stretch = {}

    design['nvisitsTotal'] = 825
    stretch['nvisitsTotal'] = 1000
    design['Area'] = 18000
    stretch['Area'] = 20000

    design['nvisits']={'u':56,'g':80, 'r':184, 'i':184, 'z':160, 'y':160}
    stretch['nvisits']={'u':70,'g':100, 'r':230, 'i':230, 'z':200, 'y':200}

    # mag/sq arcsec
    design['skybrightness'] = {'u':21.8, 'g':22., 'r':21.3, 'i':20.0, 'z':19.1, 'y':17.5}
    stretch['skybrightness'] = {'u':21.8, 'g':22., 'r':21.3, 'i':20.0, 'z':19.1, 'y':17.5}

    # arcsec - old seeing values
    design['seeing'] = {'u':0.77, 'g':0.73, 'r':0.7, 'i':0.67, 'z':0.65, 'y':0.63}
    stretch['seeing'] = {'u':0.77, 'g':0.73, 'r':0.7, 'i':0.67, 'z':0.65, 'y':0.63}

    # arcsec - new FWHMeff values (scaled from old seeing)
    design['FWHMeff'] = {'u':0.92, 'g':0.87, 'r':0.83, 'i':0.80, 'z':0.78, 'y':0.76}
    stretch['FWHMeff'] = {'u':0.92, 'g':0.87, 'r':0.83, 'i':0.80, 'z':0.78, 'y':0.76}

    design['singleVisitDepth'] = {'u':23.9,'g':25.0, 'r':24.7, 'i':24.0, 'z':23.3, 'y':22.1}
    stretch['singleVisitDepth'] = {'u':24.0,'g':25.1, 'r':24.8, 'i':24.1, 'z':23.4, 'y':22.2}

    # Scale the number of visits.
    if runLength != baseline:
        scalefactor = float(runLength) / float(baseline)
        # Calculate scaled value for design and stretch values of nvisits, per filter.
        for f in design['nvisits']:
            design['nvisits'][f] = int(np.floor(design['nvisits'][f] * scalefactor))
            stretch['nvisits'][f] = int(np.floor(stretch['nvisits'][f] * scalefactor))

    if benchmark == 'design':
        return design
    elif benchmark == 'stretch':
        return stretch
    else:
        raise ValueError("Benchmark value %s not understood: use 'design' or 'stretch'" % (benchmark))


def calcCoaddedDepth(nvisits, singleVisitDepth):
    """
    Calculate the coadded depth expected for a given number of visits and single visit depth.

    Parameters
    ----------
    nvisits : dict of ints or floats
        Dictionary (per filter) of number of visits
    singleVisitDepth : dict of floats
        Dictionary (per filter) of the single visit depth

    Returns
    -------
    dict of floats
        Dictionary of coadded depths per filter.
    """
    coaddedDepth = {}
    for f in nvisits:
        if f not in singleVisitDepth:
            raise ValueError('Filter keys in nvisits and singleVisitDepth must match')
        coaddedDepth[f] = float(1.25 * np.log10(nvisits[f] * 10**(0.8*singleVisitDepth[f])))
        if not np.isfinite(coaddedDepth[f]):
            coaddedDepth[f] = singleVisitDepth[f]
    return coaddedDepth


def labelVisits(opsimdb_file):
    """Identify the WFD as the part of the sky with at least 750 visits per pointing and not DD,
    discount short exposures."""
    # Import these here to avoid circular dependencies
    from ..metrics import CountMetric
    from ..slicers import HealpixSlicer
    from ..metricBundles import MetricBundle, MetricBundleGroup
    from ..stackers import WFDlabelStacker
    from ..db import OpsimDatabase

    runName = os.path.split(opsimdb_file)[-1].replace('.db', '')
    # The way this is written, in order to be able to freely use the information later, we write back
    # and modify the original opsim output. This can be problematic - so make a copy first and modify that.
    newdb_file = 'wfd_' + opsimdb_file
    shutil.copy(opsimdb_file, newdb_file)

    # Generate the footprint.
    m = CountMetric(col='observationStartMJD')
    s = HealpixSlicer(nside=64)
    sqlconstraint = 'visitExposureTime > 11 and note not like "%DD%"'
    bundle = MetricBundle(m, s, sqlconstraint, runName=runName)
    opsdb = OpsimDatabase(newdb_file)
    g = MetricBundleGroup({f'{runName} footprint': bundle}, opsdb)
    g.runAll()
    wfd_footprint = bundle.metricValues.filled(0)
    wfd_footprint = np.where(wfd_footprint > 750, 1, 0)
    tablename = opsdb.defaultTable
    opsdb.close()

    # Reopen with sqlite, so we can write back to the tables later.
    conn = sqlite3.connect(newdb_file)
    cursor = conn.cursor()
    query = f'select observationId, observationStartMJD, fieldRA, fieldDec, filter, ' \
            f'visitExposureTime, note from {tablename}'
    simdata = pd.read_sql(query, conn).to_records()
    # label the visits with the visit label stacker
    wfd_stacker = WFDlabelStacker(wfd_footprint)
    simdata = wfd_stacker.run(simdata)

    # Write back proposalId/observation to the table in the new copy of the database.
    # Create some indexes
    try:
        indxObsId = f"CREATE UNIQUE INDEX idx_observationId on {tablename} (observationId)"
        cursor.execute(indxObsId)
    except OperationalError:
        print(f'Already had observationId index on {tablename}')
    try:
        indxMJD = f"CREATE UNIQUE INDEX idx_observationStartMJD on {tablename} (observationStartMJD);"
        cursor.execute(indxMJD)
    except OperationalError:
        print('Already had observationStartMJD index')
    try:
        indxFilter = f"CREATE INDEX idx_filter on {tablename} (filter)"
        cursor.execute(indxFilter)
    except OperationalError:
        print('Already had filter index')
    # Add the proposal id information.
    for obsid, pId in zip(simdata['observationId'], simdata['proposalId']):
        sql = f'UPDATE {tablename} SET proposalId = {pId} WHERE observationId = {obsid}'
        cursor.execute(sql)
    conn.commit()

    # Define dictionary of proposal tags.
    propTags = {'Other': 0, 'WFD': 1}
    propIds = np.unique(simdata['proposalId'])
    for iD in propIds:
        if iD not in propTags:
            tag = simdata['note'][np.where(simdata['proposalId'] == iD)][0]
            propTags[tag] = iD
    # Add new table to track proposal information.
    sql = 'CREATE TABLE IF NOT EXISTS "Proposal" ("proposalId" INT PRIMARY KEY, ' \
          '"proposalName" VARCHAR(20), "proposalType" VARCHAR(5))'
    cursor.execute(sql)
    # Add proposal information to Proposal table.
    for pName, pId in propTags.items():
        pType = pName.split(':')[0]
        try:
            sql = f'INSERT INTO Proposal (proposalId, proposalName, proposalType) ' \
                  f'VALUES ("{pId}", "{pName}", "{pType}")'
            cursor.execute(sql)
        except IntegrityError:
            print(f'This proposal ID is already in the proposal table {pId},{pName} (just reusing it)')
    conn.commit()
    conn.close()
    print(f'Labelled visits in new database copy {newdb_file}')
    return newdb_file
