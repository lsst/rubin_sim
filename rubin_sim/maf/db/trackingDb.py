from __future__ import print_function
from builtins import str
from builtins import object
import os
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.engine import url
from sqlalchemy.orm import sessionmaker

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.exc import DatabaseError

Base = declarative_base()

__all__ = ['TrackingDb', 'addRunToDatabase']


class RunRow(Base):
    """
    Define contents and format of run list table.

    Table to list all available MAF results, along with their opsim run and some comment info.
    """
    __tablename__ = "runs"
    # Define columns in metric list table.
    mafRunId = Column(Integer, primary_key=True)
    opsimGroup = Column(String)
    opsimRun = Column(String)
    opsimComment = Column(String)
    opsimVersion = Column(String)
    opsimDate = Column(String)
    dbFile = Column(String)
    mafComment = Column(String)
    mafVersion = Column(String)
    mafDate = Column(String)
    mafDir = Column(String)
    def __repr__(self):
        rstr = "<Run(mafRunId='%d', opsimGroup='%s', opsimRun='%s', opsimComment='%s', " \
               "opsimVersion='%s', opsimDate='%s', mafComment='%s', " \
               "mafVersion='%s', mafDate='%s', mafDir='%s', dbFile='%s'>" \
                % (self.mafRunId, self.opsimGroup, self.opsimRun, self.opsimComment,
                   self.opsimVersion, self.opsimDate, self.mafComment, self.mafVersion, self.mafDate,
                   self.mafDir, self.dbFile)
        return rstr


class TrackingDb(object):
    """Sqlite database to track MAF output runs and their locations, for showMaf.py
    """
    def __init__(self, database=None, trackingDbverbose=False):
        """
        Instantiate the results database, creating metrics, plots and summarystats tables.
        """
        self.verbose = trackingDbverbose
        self.driver = 'sqlite'
        # Connect to database
        # for sqlite, connecting to non-existent database creates it automatically
        if database is None:
            # Default is a file in the current directory.
            self.database = os.path.join(os.getcwd(), 'trackingDb_sqlite.db')
        else:
            self.database  = database
        # only sqlite
        dbAddress = url.URL.create(drivername=self.driver, database=self.database)
        engine = create_engine(dbAddress, echo=self.verbose)
        if self.verbose:
            print('Created or connected to MAF tracking %s database at %s' %(self.driver, self.database))
        self.Session = sessionmaker(bind=engine)
        self.session = self.Session()
        # Create the tables, if they don't already exist.
        try:
            Base.metadata.create_all(engine)
        except DatabaseError:
            raise DatabaseError("Cannot create a %s database at %s. Check directory exists." %(self.driver,
                                                                                               self.database))

    def close(self):
        self.session.close()

    def addRun(self, opsimGroup=None, opsimRun=None, opsimComment=None, opsimVersion=None, opsimDate=None,
               mafComment=None, mafVersion=None, mafDate=None, mafDir=None, dbFile=None, mafRunId=None):
        """Add a run to the tracking database.
        
        Parameters
        ----------
        opsimGroup : str, optional
            Set a name to group this run with (eg. "Tier 1, 2016").
        opsimRun : str, optional
            Set a name for the opsim run.
        opsimComment : str, optional
            Set a comment describing the opsim run.
        opsimVersion : str, optional
            Set the version of opsim.
        opsimDate : str, optional
            Set the date the opsim run was created.
        mafComment : str, optional
            Set a comment to describe the MAF analysis.
        mafVersion : str, optional
            Set the version of MAF used for analysis.
        mafDate : str, optional
            Set the date the MAF analysis was run.
        mafDir : str, optional
            The relative path to the MAF directory.
        dbFile : str, optional
            The relative path to the Opsim SQLite database file.
        mafRunId : int, optional
            The MafRunID to assign to this record in the database (note this is a primary key!).
            If this run (ie the mafDir) exists in the database already, this will be ignored.            
            
        Returns
        -------
        int
            The mafRunID stored in the database.
        """
        if opsimGroup is None:
            opsimGroup = 'NULL'
        if opsimRun is None:
            opsimRun = 'NULL'
        if opsimComment is None:
            opsimComment = 'NULL'
        if opsimVersion is None:
            opsimVersion = 'NULL'
        if opsimDate is None:
            opsimDate = 'NULL'
        if mafComment is None:
            mafComment = 'NULL'
        if mafVersion is None:
            mafVersion = 'NULL'
        if mafDate is None:
            mafDate = 'NULL'
        if mafDir is None:
            mafDir = 'NULL'
        if dbFile is None:
            dbFile = 'NULL'
        # Test if mafDir already exists in database.
        prevrun = self.session.query(RunRow).filter_by(mafDir=mafDir).all()
        if len(prevrun) > 0:
            runIds = []
            for run in prevrun:
                runIds.append(run.mafRunId)
            print('Updating information in tracking database - %s already present with runId %s.'
                  % (mafDir, runIds))
            for run in prevrun:
                self.session.delete(run)
            self.session.commit()
            runinfo = RunRow(mafRunId=runIds[0], opsimGroup=opsimGroup, opsimRun=opsimRun,
                             opsimComment=opsimComment, opsimVersion=opsimVersion, opsimDate=opsimDate,
                             mafComment=mafComment, mafVersion=mafVersion, mafDate=mafDate,
                             mafDir=mafDir, dbFile=dbFile)
        else:
            if mafRunId is not None:
                # Check if mafRunId exists already.
                existing = self.session.query(RunRow).filter_by(mafRunId=mafRunId).all()
                if len(existing) > 0:
                    raise ValueError('MafRunId %d already exists in database, for %s. ' \
                                     'Record must be deleted first.'
                                     % (mafRunId, existing[0].mafDir))
                runinfo = RunRow(mafRunId=mafRunId, opsimGroup=opsimGroup, opsimRun=opsimRun,
                                 opsimComment=opsimComment, opsimVersion=opsimVersion, opsimDate=opsimDate,
                                 mafComment=mafComment, mafVersion=mafVersion, mafDate=mafDate,
                                 mafDir=mafDir, dbFile=dbFile)
            else:
                runinfo = RunRow(opsimGroup=opsimGroup, opsimRun=opsimRun,
                                 opsimComment=opsimComment, opsimVersion=opsimVersion, opsimDate=opsimDate,
                                 mafComment=mafComment, mafVersion=mafVersion, mafDate=mafDate,
                                 mafDir=mafDir, dbFile=dbFile)
        self.session.add(runinfo)
        self.session.commit()
        return runinfo.mafRunId

    def delRun(self, runId):
        """
        Remove a run from the tracking database.
        """
        runinfo = self.session.query(RunRow).filter_by(mafRunId=runId).all()
        if len(runinfo) == 0:
            raise Exception('Could not find run with mafRunId %d' %(runId))
        if len(runinfo) > 1:
            raise Exception('Found more than one run with mafRunId %d' %(runId))
        print('Removing run info for runId %d ' %(runId))
        print(' ', runinfo)
        self.session.delete(runinfo[0])
        self.session.commit()


def addRunToDatabase(mafDir, trackingDbFile, opsimGroup=None,
                    opsimRun=None, opsimComment=None,
                    mafComment=None, dbFile=None):
    """Adds information about a MAF analysis run to a MAF tracking database.

    Parameters
    ----------
    mafDir : str
        Path to the directory where the MAF results are located.
    trackingDb : str or rubin_sim.maf.TrackingDb
        Full filename (+path) to the tracking database storing the MAF run information or
        a TrackingDb object.
    opsimGroup: str, optional
        Name to use to group this run with other opsim runs. Default None.
    opsimRun : str, optional
        Name of the opsim run. If not provided, will attempt to use runName from confSummary.txt.
    opsimComment : str, optional
        Comment about the opsim run. If not provided, will attempt to use runComment from confSummary.txt.
    mafComment : str, optional
        Comment about the MAF analysis. If not provided, no comment will be recorded.
    dbFile : str, optional
        Relative path + name of the opsim database file. If not provided, no location will be recorded.
    """
    mafDir = os.path.abspath(mafDir)
    if not os.path.isdir(mafDir):
        raise ValueError('There is no directory containing MAF outputs at %s.' % (mafDir))

    trackingDb = TrackingDb(database=trackingDbFile)
    autoOpsimRun = None
    autoOpsimComment = None
    opsimVersion = None
    opsimDate = None
    mafVersion = None
    mafDate = None
    if os.path.isfile(os.path.join(mafDir, 'configSummary.txt')):
        file = open(os.path.join(mafDir, 'configSummary.txt'))
        for line in file:
            tmp = line.split()
            if tmp[0].startswith('RunName'):
                autoOpsimRun = ' '.join(tmp[1:])
            if tmp[0].startswith('RunComment'):
                autoOpsimComment = ' '.join(tmp[1:])
            # MAF Date may be in a line with "MafDate" (new configs)
            #  or at the end of "MAFVersion" (old configs).
            if tmp[0].startswith('MAFDate'):
                mafDate = tmp[-1]
            if tmp[0].startswith('MAFVersion'):
                mafVersion = tmp[1]
                if len(tmp) > 2:
                    mafDate = tmp[-1]
            if tmp[0].startswith('OpsimDate'):
                opsimDate = tmp[-1]
                if len(tmp) > 2:
                    opsimDate = tmp[-2]
            if tmp[0].startswith('OpsimVersion'):
                opsimVersion = tmp[1]
                if len(tmp) > 2:
                    opsimDate = tmp[-2]
    # And convert formats to '-' (again, multiple versions of configs).
    if mafDate is not None:
        if len(mafDate.split('/')) > 1:
            t = mafDate.split('/')
            if len(t[2]) == 2:
                t[2] = '20' + t[2]
            mafDate = '-'.join([t[2], t[1], t[0]])
    if opsimDate is not None:
        if len(opsimDate.split('/')) > 1:
            t = opsimDate.split('/')
            if len(t[2]) == 2:
                t[2] = '20' + t[2]
            opsimDate = '-'.join([t[2], t[1], t[0]])

    if opsimRun is None:
        opsimRun = autoOpsimRun
    if opsimComment is None:
        opsimComment = autoOpsimComment

    print('Adding to tracking database at %s:' % (trackingDbFile))
    print(' MafDir = %s' % (mafDir))
    print(' MafComment = %s' % (mafComment))
    print(' OpsimGroup = %s' % (opsimGroup))
    print(' OpsimRun = %s' % (opsimRun))
    print(' OpsimComment = %s' % (opsimComment))
    print(' OpsimVersion = %s' % (opsimVersion))
    print(' OpsimDate = %s' % (opsimDate))
    print(' MafVersion = %s' % (mafVersion))
    print(' MafDate = %s' % (mafDate))
    print(' Opsim dbFile = %s' % (dbFile))
    runId = trackingDb.addRun(opsimGroup=opsimGroup, opsimRun=opsimRun, opsimComment=opsimComment,
                              opsimVersion=opsimVersion, opsimDate=opsimDate,
                              mafComment=mafComment, mafVersion=mafVersion, mafDate=mafDate,
                              mafDir=mafDir, dbFile=dbFile)
    print('Used MAF RunID %d' % (runId))
    trackingDb.close()
