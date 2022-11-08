import os
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.engine import url
from sqlalchemy.orm import sessionmaker

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.exc import DatabaseError
import pandas as pd
import sqlite3

Base = declarative_base()

__all__ = ["TrackingDb", "add_run_to_database"]


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
        rstr = (
            "<Run(maf_run_id='%d', run_group='%s', run_name='%s', run_comment='%s', "
            "run_version='%s', run_date='%s', maf_comment='%s', "
            "maf_version='%s', maf_date='%s', maf_dir='%s', db_file='%s'>"
            % (
                self.mafRunId,
                self.opsimGroup,
                self.opsimRun,
                self.opsimComment,
                self.opsimVersion,
                self.opsimDate,
                self.mafComment,
                self.mafVersion,
                self.mafDate,
                self.mafDir,
                self.dbFile,
            )
        )
        return rstr


class TrackingDb(object):
    """Sqlite database to track MAF output runs and their locations, for showMaf.py"""

    def __init__(self, database=None, trackingDbverbose=False):
        """
        Instantiate the results database, creating metrics, plots and summarystats tables.
        """
        self.verbose = trackingDbverbose
        self.driver = "sqlite"
        # Connect to database
        # for sqlite, connecting to non-existent database creates it automatically
        if database is None:
            # Default is a file in the current directory.
            self.database = os.path.join(os.getcwd(), "trackingDb_sqlite.db")
        else:
            self.database = database
        # only sqlite
        dbAddress = url.URL.create(drivername=self.driver, database=self.database)
        engine = create_engine(dbAddress, echo=self.verbose)
        if self.verbose:
            print(
                "Created or connected to MAF tracking %s database at %s"
                % (self.driver, self.database)
            )
        self.Session = sessionmaker(bind=engine)
        self.session = self.Session()
        # Create the tables, if they don't already exist.
        try:
            Base.metadata.create_all(engine)
        except DatabaseError:
            raise DatabaseError(
                "Cannot create a %s database at %s. Check directory exists."
                % (self.driver, self.database)
            )

    def close(self):
        self.session.close()

    def add_run(
        self,
        run_group=None,
        run_name=None,
        run_comment=None,
        run_version=None,
        run_date=None,
        maf_comment=None,
        maf_version=None,
        maf_date=None,
        maf_dir=None,
        db_file=None,
        maf_run_id=None,
    ):
        """Add a run to the tracking database.

        Parameters
        ----------
        run_group : str, optional
            Set a name to group this run with (eg. "Tier 1, 2016").
        run_name : str, optional
            Set a name for the opsim run.
        run_comment : str, optional
            Set a comment describing the opsim run.
        run_version : str, optional
            Set the version of opsim.
        run_date : str, optional
            Set the date the opsim run was created.
        maf_comment : str, optional
            Set a comment to describe the MAF analysis.
        maf_version : str, optional
            Set the version of MAF used for analysis.
        maf_date : str, optional
            Set the date the MAF analysis was run.
        maf_dir : str, optional
            The relative path to the MAF directory.
        db_file : str, optional
            The relative path to the Opsim SQLite database file.
        maf_run_id : int, optional
            The MafRunID to assign to this record in the database (note this is a primary key!).
            If this run (ie the maf_dir) exists in the database already, this will be ignored.

        Returns
        -------
        int
            The mafRunID stored in the database.
        """
        if run_group is None:
            run_group = "NULL"
        if run_name is None:
            run_name = "NULL"
        if run_comment is None:
            run_comment = "NULL"
        if run_version is None:
            run_version = "NULL"
        if run_date is None:
            run_date = "NULL"
        if maf_comment is None:
            maf_comment = "NULL"
        if maf_version is None:
            maf_version = "NULL"
        if maf_date is None:
            maf_date = "NULL"
        if maf_dir is None:
            maf_dir = "NULL"
        if db_file is None:
            db_file = "NULL"
        # Test if maf_dir already exists in database.
        prevrun = self.session.query(RunRow).filter_by(mafDir=maf_dir).all()
        if len(prevrun) > 0:
            runIds = []
            for run in prevrun:
                runIds.append(run.mafRunId)
            print(
                "Updating information in tracking database - %s already present with runId %s."
                % (maf_dir, runIds)
            )
            for run in prevrun:
                self.session.delete(run)
            self.session.commit()
            runinfo = RunRow(
                mafRunId=runIds[0],
                opsimGroup=run_group,
                opsimRun=run_name,
                opsimComment=run_comment,
                opsimVersion=run_version,
                opsimDate=run_date,
                mafComment=maf_comment,
                mafVersion=maf_version,
                mafDate=maf_date,
                mafDir=maf_dir,
                dbFile=db_file,
            )
        else:
            if maf_run_id is not None:
                # Check if maf_run_id exists already.
                existing = (
                    self.session.query(RunRow).filter_by(mafRunId=maf_run_id).all()
                )
                if len(existing) > 0:
                    raise ValueError(
                        "MafRunId %d already exists in database, for %s. "
                        "Record must be deleted first."
                        % (maf_run_id, existing[0].mafDir)
                    )
                runinfo = RunRow(
                    mafRunId=maf_run_id,
                    opsimGroup=run_group,
                    opsimRun=run_name,
                    opsimComment=run_comment,
                    opsimVersion=run_version,
                    opsimDate=run_date,
                    mafComment=maf_comment,
                    mafVersion=maf_version,
                    mafDate=maf_date,
                    mafDir=maf_dir,
                    dbFile=db_file,
                )
            else:
                runinfo = RunRow(
                    opsimGroup=run_group,
                    opsimRun=run_name,
                    opsimComment=run_comment,
                    opsimVersion=run_version,
                    opsimDate=run_date,
                    mafComment=maf_comment,
                    mafVersion=maf_version,
                    mafDate=maf_date,
                    mafDir=maf_dir,
                    dbFile=db_file,
                )
        self.session.add(runinfo)
        self.session.commit()
        return runinfo.mafRunId

    def delRun(self, runId):
        """
        Remove a run from the tracking database.
        """
        runinfo = self.session.query(RunRow).filter_by(mafRunId=runId).all()
        if len(runinfo) == 0:
            raise Exception("Could not find run with maf_run_id %d" % (runId))
        if len(runinfo) > 1:
            raise Exception("Found more than one run with maf_run_id %d" % (runId))
        print("Removing run info for runId %d " % (runId))
        print(" ", runinfo)
        self.session.delete(runinfo[0])
        self.session.commit()


def add_run_to_database(
    maf_dir,
    tracking_db_file,
    run_group=None,
    run_name=None,
    run_comment=None,
    run_version=None,
    maf_comment=None,
    db_file=None,
):
    """Adds information about a MAF analysis run to a MAF tracking database.

    Parameters
    ----------
    maf_dir : `str`
        Path to the directory where the MAF results are located.
    tracking_db_file : `str`
        Full filename (+path) to the tracking database storing the MAF run information.
    run_group: `str`, optional
        Name to use to group this run with other opsim runs. Default None.
    run_name : `str`, optional
        Name of the opsim run. If not provided, will attempt to use run_name from configSummary.txt.
    run_comment : `str`, optional
        Comment about the opsim run. If not provided, will attempt to use runComment from configSummary.txt.
    run_version : `str`, optional
        Value to use for the opsim version information. If not provided, will attempt to use the value from
        configSummary.txt
    maf_comment : `str`, optional
        Comment about the MAF analysis. If not provided, no comment will be recorded.
    db_file : `str`, optional
        Relative path + name of the opsim database file. If not provided, no location will be recorded.
    """
    maf_dir = os.path.abspath(maf_dir)
    if not os.path.isdir(maf_dir):
        raise ValueError(
            "There is no directory containing MAF outputs at %s." % (maf_dir)
        )

    trackingDb = TrackingDb(database=tracking_db_file)
    autoOpsimRun = None
    autoOpsimComment = None
    autoOpsimVersion = None
    run_date = None
    maf_version = None
    maf_date = None

    if os.path.isfile(os.path.join(maf_dir, "configSummary.txt")):
        file = open(os.path.join(maf_dir, "configSummary.txt"))
        for line in file:
            tmp = line.split()
            if tmp[0].startswith("RunName"):
                autoOpsimRun = " ".join(tmp[1:])
            if tmp[0].startswith("RunComment"):
                autoOpsimComment = " ".join(tmp[1:])
            # MAF Date may be in a line with "MafDate" (new configs)
            #  or at the end of "MAFVersion" (old configs).
            if tmp[0].startswith("MAFDate"):
                maf_date = tmp[-1]
            if tmp[0].startswith("MAFVersion"):
                maf_version = tmp[1]
                if len(tmp) > 2:
                    maf_date = tmp[-1]
            if tmp[0].startswith("OpsimDate"):
                run_date = tmp[-1]
                if len(tmp) > 2:
                    run_date = tmp[-2]
            if tmp[0].startswith("rubin_sim.__version__"):
                autoOpsimVersion = tmp[1]
                if len(tmp) > 2:
                    autoOpsimVersion = tmp[-2]
    # And convert formats to '-' (again, multiple versions of configs).
    if maf_date is not None:
        if len(maf_date.split("/")) > 1:
            t = maf_date.split("/")
            if len(t[2]) == 2:
                t[2] = "20" + t[2]
            maf_date = "-".join([t[2], t[1], t[0]])
    if run_date is not None:
        if len(run_date.split("/")) > 1:
            t = run_date.split("/")
            if len(t[2]) == 2:
                t[2] = "20" + t[2]
            run_date = "-".join([t[2], t[1], t[0]])

    if run_name is None:
        run_name = autoOpsimRun
    if run_comment is None:
        run_comment = autoOpsimComment
    if run_version is None:
        run_version = autoOpsimVersion

    # Check if version and date are in the database.
    if maf_version is None:
        try:
            conn = sqlite3.connect(os.path.join(maf_dir, "resultsDb_sqlite.db"))
            versDF = pd.read_sql("SELECT version,rundate FROM version;", conn)
            maf_version = versDF["version"].values[-1]
            maf_date = versDF["rundate"].values[-1]
            conn.close()
        except:
            pass

    print("Adding to tracking database at %s:" % (tracking_db_file))
    print(" Maf_dir = %s" % (maf_dir))
    print(" Maf_comment = %s" % (maf_comment))
    print(" run_group = %s" % (run_group))
    print(" run_name = %s" % (run_name))
    print(" run_comment = %s" % (run_comment))
    print(" run_version = %s" % (run_version))
    print(" run_date = %s" % (run_date))
    print(" maf_version = %s" % (maf_version))
    print(" maf_date = %s" % (maf_date))
    print(" db_file = %s" % (db_file))
    runId = trackingDb.add_run(
        run_group=run_group,
        run_name=run_name,
        run_comment=run_comment,
        run_version=run_version,
        run_date=run_date,
        maf_comment=maf_comment,
        maf_version=maf_version,
        maf_date=maf_date,
        maf_dir=maf_dir,
        db_file=db_file,
    )
    print("Used MAF RunID %d" % (runId))
    trackingDb.close()
