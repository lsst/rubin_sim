from __future__ import print_function
# Collection of utilities for MAF that relate to Opsim specifically.

import os
import numpy as np
from .outputUtils import printDict

__all__ = ['writeConfigs', 'getFieldData', 'getSimData',
           'scaleBenchmarks', 'calcCoaddedDepth']


def writeConfigs(opsimDb, outDir):
    """
    Convenience function to get the configuration information from the opsim database and write
    this information to text files 'configSummary.txt' and 'configDetails.txt'.

    Parameters
    ----------
    opsimDb : OpsimDatabase
        The opsim database from which to pull the opsim configuration information.
        Opsim SQLite databases save this configuration information in their config table.
    outputDir : str
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
    :meth:`OpsimDatabase.fetchFieldsFromFieldTable()`
    to get the fields. If the opsimDb contains only a Summary, it uses
    :meth:`OpsimDatabase.fetchFieldsFromSummaryTable()`.

    Parameters
    ----------
    opsimDb : OpsimDatabase
        An opsim database to use to query for field information.
    sqlconstraint : str
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
    """
    Query an opsim database for the needed data columns and run any required stackers.

    Parameters
    ----------
    opsimDb : OpsimDatabase
    sqlconstraint : str
        SQL constraint to apply to query for observations.
    dbcols : list of str
        Columns required from the database.
    stackers : list of Stackers
        Stackers to be used to generate additional columns.
    tableName : str
        Name of the table to query.
    distinctExpMJD : bool
        Only select observations with a distinct expMJD value. This is overriden if groupBy is not expMJD.
    groupBy : str
        Column name to group SQL results by.

    Returns
    -------
    numpy.ndarray
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
    dict of floats
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

    design['skybrightness'] = {'u':21.8, 'g':22., 'r':21.3, 'i':20.0, 'z':19.1, 'y':17.5} # mag/sq arcsec
    stretch['skybrightness'] = {'u':21.8, 'g':22., 'r':21.3, 'i':20.0, 'z':19.1, 'y':17.5}

    design['seeing'] = {'u':0.77, 'g':0.73, 'r':0.7, 'i':0.67, 'z':0.65, 'y':0.63} # arcsec - old seeing values
    stretch['seeing'] = {'u':0.77, 'g':0.73, 'r':0.7, 'i':0.67, 'z':0.65, 'y':0.63}

    design['FWHMeff'] = {'u':0.92, 'g':0.87, 'r':0.83, 'i':0.80, 'z':0.78, 'y':0.76} # arcsec - new FWHMeff values (scaled from old seeing)
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
