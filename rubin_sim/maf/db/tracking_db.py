__all__ = ("TrackingDb", "add_run_to_database")

import os

from sqlalchemy import Column, Integer, String, create_engine
from sqlalchemy.engine import url
from sqlalchemy.exc import DatabaseError
from sqlalchemy.orm import declarative_base, sessionmaker

from . import ResultsDb, VersionRow

Base = declarative_base()


class RunRow(Base):
    """
    Define contents and format of run list table.

    Table to list all available MAF results,
    along with their opsim run and some comment info.
    """

    __tablename__ = "runs"
    # Define columns in metric list table.
    maf_run_id = Column(Integer, primary_key=True)
    run_group = Column(String)
    run_name = Column(String)
    run_comment = Column(String)
    run_version = Column(String)
    run_date = Column(String)
    db_file = Column(String)
    maf_comment = Column(String)
    maf_version = Column(String)
    maf_date = Column(String)
    maf_dir = Column(String)

    def __repr__(self):
        rstr = (
            "<Run(maf_run_id='%d', run_group='%s', run_name='%s', run_comment='%s', "
            "run_version='%s', run_date='%s', maf_comment='%s', "
            "maf_version='%s', maf_date='%s', maf_dir='%s', db_file='%s'>"
            % (
                self.maf_run_id,
                self.run_group,
                self.run_name,
                self.maf_comment,
                self.maf_version,
                self.run_date,
                self.maf_comment,
                self.maf_version,
                self.maf_date,
                self.maf_dir,
                self.db_file,
            )
        )
        return rstr


class TrackingDb:
    """Sqlite database to track MAF output runs and their locations,
    for show_maf
    """

    def __init__(self, database=None, trackingDbverbose=False):
        """
        Set up the Tracking Database.
        """
        self.verbose = trackingDbverbose
        self.driver = "sqlite"
        # connecting to non-existent database creates it automatically
        if database is None:
            # Default is a file in the current directory.
            self.database = "trackingDb_sqlite.db"
            self.tracking_db_dir = "."
        else:
            self.database = database
            self.tracking_db_dir = os.path.dirname(database)
        # only sqlite
        dbAddress = url.URL.create(drivername=self.driver, database=self.database)
        engine = create_engine(dbAddress, echo=self.verbose)
        if self.verbose:
            print("Created or connected to MAF tracking %s database at %s" % (self.driver, self.database))
        self.Session = sessionmaker(bind=engine)
        self.session = self.Session()
        # Create the tables, if they don't already exist.
        try:
            Base.metadata.create_all(engine)
        except DatabaseError:
            raise DatabaseError(
                "Cannot create a %s database at %s. Check directory exists." % (self.driver, self.database)
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
        run_group : `str`, optional
            Set a name to group this run with (eg. "Tier 1, 2016").
        run_name : `str`, optional
            Set a name for the opsim run.
        run_comment : `str`, optional
            Set a comment describing the opsim run.
        run_version : `str`, optional
            Set the version of opsim.
        run_date : `str`, optional
            Set the date the opsim run was created.
        maf_comment : `str`, optional
            Set a comment to describe the MAF analysis.
        maf_version : `str`, optional
            Set the version of MAF used for analysis.
        maf_date : `str`, optional
            Set the date the MAF analysis was run.
        maf_dir : `str`, optional
            The relative path to the MAF directory.
            Will be converted to a relative path if absolute.
        db_file : `str`, optional
            The relative path to the Opsim SQLite database file.
        maf_run_id : `int`, optional
            The maf_run_id to assign to this record in the database
            (note this is a primary key!).
            If this run (ie the maf_dir) exists in the database already,
            this will be ignored.

        Returns
        -------
        maf_run_id : `int`
            The maf_run_id stored in the database.
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
        if maf_dir is not None:
            maf_dir = os.path.relpath(maf_dir, start=self.tracking_db_dir)
        if db_file is None:
            db_file = "NULL"
        # Test if maf_dir already exists in database.
        prevrun = self.session.query(RunRow).filter_by(maf_dir=maf_dir).all()
        if len(prevrun) > 0:
            runIds = []
            for run in prevrun:
                runIds.append(run.maf_run_id)
            print(
                "Updating information in tracking database - %s already present with runId %s."
                % (maf_dir, runIds)
            )
            for run in prevrun:
                self.session.delete(run)
            self.session.commit()
            runinfo = RunRow(
                maf_run_id=runIds[0],
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
        else:
            if maf_run_id is not None:
                # Check if maf_run_id exists already.
                existing = self.session.query(RunRow).filter_by(maf_run_id=maf_run_id).all()
                if len(existing) > 0:
                    raise ValueError(
                        "maf_run_id %d already exists in database, for %s. "
                        "Record must be deleted first." % (maf_run_id, existing[0].maf_dir)
                    )
                runinfo = RunRow(
                    maf_run_id=maf_run_id,
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
            else:
                runinfo = RunRow(
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
        self.session.add(runinfo)
        self.session.commit()
        return runinfo.maf_run_id

    def delRun(self, runId):
        """
        Remove a run from the tracking database.
        """
        runinfo = self.session.query(RunRow).filter_by(maf_run_id=runId).all()
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
    maf_comment=None,
    db_file=None,
    maf_version=None,
    maf_date=None,
    sched_version=None,
    sched_date=None,
    skip_extras=False,
):
    """Adds information about a MAF analysis run to a MAF tracking database.

    Parameters
    ----------
    maf_dir : `str`
        Path to the directory where the MAF results are located.
    tracking_db_file : `str`
        Full filename (+path) to the tracking database to use.
    run_group: `str`, optional
        Name to use to group this run with other opsim runs.
    run_name : `str`, optional
        Name of the opsim run.
    run_comment : `str`, optional
        Comment about the opsim run.
    run_version : `str`, optional
        Value to use for the opsim version information.
    maf_comment : `str`, optional
        Comment about the MAF analysis.
    db_file : `str`, optional
        Relative path + name of the opsim database file.
    """
    trackingDb = TrackingDb(database=tracking_db_file)

    maf_dir = os.path.relpath(maf_dir, start=os.path.dirname(trackingDb.tracking_db_dir))
    if not os.path.isdir(maf_dir):
        raise ValueError("There is no directory containing MAF outputs at %s." % (maf_dir))

    # Connect to resultsDb for additional information if available
    if os.path.isfile(os.path.join(maf_dir, "resultsDb_sqlite.db")) and not skip_extras:
        resdb = ResultsDb(maf_dir)
        if run_name is None:
            run_name = resdb.get_run_name()
            if len(run_name) > 1:
                run_name = 0
            else:
                run_name = run_name[0]
        if maf_version is None or maf_date is None:
            resdb.open()
            query = resdb.session.query(VersionRow).all()
            for v in query:
                if maf_version is None:
                    maf_version = v.version
                if maf_date is None:
                    maf_date = v.run_date
            resdb.close()

    print("Adding to tracking database at %s:" % (tracking_db_file))
    print(" Maf_dir = %s" % (maf_dir))
    print(" Maf_comment = %s" % (maf_comment))
    print(" run_group = %s" % (run_group))
    print(" run_name = %s" % (run_name))
    print(" run_comment = %s" % (run_comment))
    print(" run_version = %s" % (sched_version))
    print(" run_date = %s" % (sched_date))
    print(" maf_version = %s" % (maf_version))
    print(" maf_date = %s" % (maf_date))
    print(" db_file = %s" % (db_file))
    runId = trackingDb.add_run(
        run_group=run_group,
        run_name=run_name,
        run_comment=run_comment,
        run_version=sched_version,
        run_date=sched_date,
        maf_comment=maf_comment,
        maf_version=maf_version,
        maf_date=maf_date,
        maf_dir=maf_dir,
        db_file=db_file,
    )
    print("Used MAF RunID %d" % (runId))
    trackingDb.close()
