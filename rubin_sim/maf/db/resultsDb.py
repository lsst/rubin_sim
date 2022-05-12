import os, warnings
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.engine import url
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Float
from sqlalchemy import ForeignKey
from sqlalchemy.orm import relationship, backref
from sqlalchemy.exc import DatabaseError
import rubin_sim.version as rsVersion
import datetime


import numpy as np

Base = declarative_base()

__all__ = [
    "MetricRow",
    "DisplayRow",
    "PlotRow",
    "SummaryStatRow",
    "VersionRow",
    "ResultsDb",
]


class MetricRow(Base):
    """
    Define contents and format of metric list table.

    (Table to list all metrics, their info_label, and their output data files).
    """

    __tablename__ = "metrics"
    # Define columns in metric list table.
    metricId = Column(Integer, primary_key=True)
    metricName = Column(String)
    slicerName = Column(String)
    simDataName = Column(String)
    sqlConstraint = Column(String)
    metricInfoLabel = Column(String)
    metricDataFile = Column(String)

    def __repr__(self):
        return (
            "<Metric(metricId='%d', metricName='%s', slicerName='%s', "
            "simDataName='%s', sqlConstraint='%s', metricInfoLabel='%s', metricDataFile='%s')>"
        ) % (
            self.metricId,
            self.metricName,
            self.slicerName,
            self.simDataName,
            self.sqlConstraint,
            self.metricInfoLabel,
            self.metricDataFile,
        )


class VersionRow(Base):
    """"""

    __tablename__ = "version"
    verId = Column(Integer, primary_key=True)
    version = Column(String)
    rundate = Column(String)

    def __repr__(self):
        return ("<Version(version='%s', rundate='%s')>") % (self.version, self.rundate)


class DisplayRow(Base):
    """
    Define contents and format of the displays table.

    (Table to list the display properties for each metric.)
    """

    __tablename__ = "displays"
    displayId = Column(Integer, primary_key=True)
    metricId = Column(Integer, ForeignKey("metrics.metricId"))
    # Group for displaying metric (in webpages).
    displayGroup = Column(String)
    # Subgroup for displaying metric.
    displaySubgroup = Column(String)
    # Order to display metric (within subgroup).
    displayOrder = Column(Float)
    # The figure caption.
    displayCaption = Column(String)
    metric = relationship("MetricRow", backref=backref("displays", order_by=displayId))

    def __rep__(self):
        return (
            "<Display(displayGroup='%s', displaySubgroup='%s', "
            "displayOrder='%.1f', displayCaption='%s')>"
            % (
                self.displayGroup,
                self.displaySubgroup,
                self.displayOrder,
                self.displayCaption,
            )
        )


class PlotRow(Base):
    """
    Define contents and format of plot list table.

    (Table to list all plots, link them to relevant metrics in MetricList, and provide info on filename).
    """

    __tablename__ = "plots"
    # Define columns in plot list table.
    plotId = Column(Integer, primary_key=True)
    # Matches metricID in MetricList table.
    metricId = Column(Integer, ForeignKey("metrics.metricId"))
    plotType = Column(String)
    plotFile = Column(String)
    metric = relationship("MetricRow", backref=backref("plots", order_by=plotId))

    def __repr__(self):
        return "<Plot(metricId='%d', plotType='%s', plotFile='%s')>" % (
            self.metricId,
            self.plotType,
            self.plotFile,
        )


class SummaryStatRow(Base):
    """
    Define contents and format of the summary statistics table.

    (Table to list and link summary stats to relevant metrics in MetricList, and provide summary stat name,
    value and potentially a comment).
    """

    __tablename__ = "summarystats"
    # Define columns in plot list table.
    statId = Column(Integer, primary_key=True)
    # Matches metricID in MetricList table.
    metricId = Column(Integer, ForeignKey("metrics.metricId"))
    summaryName = Column(String)
    summaryValue = Column(Float)
    metric = relationship("MetricRow", backref=backref("summarystats", order_by=statId))

    def __repr__(self):
        return "<SummaryStat(metricId='%d', summaryName='%s', summaryValue='%f')>" % (
            self.metricId,
            self.summaryName,
            self.summaryValue,
        )


class ResultsDb(object):
    """The ResultsDb is a sqlite database containing information on the metrics run via MAF,
    the plots created, the display information (such as captions), and any summary statistics output.
    """

    def __init__(self, outDir=None, database=None, verbose=False):
        """
        Instantiate the results database, creating metrics, plots and summarystats tables.
        """
        # We now require resultsDb to be a sqlite file (for simplicity). Leaving as attribute though.
        self.driver = "sqlite"
        # Connect to database
        # for sqlite, connecting to non-existent database creates it automatically
        if database is None:
            # Using default value for database name, should specify directory.
            if outDir is None:
                outDir = "."
            # Check for output directory, make if needed.
            if not os.path.isdir(outDir):
                try:
                    os.makedirs(outDir)
                except OSError as msg:
                    raise OSError(
                        msg,
                        "\n  (If this was the database file (not outDir), "
                        'remember to use kwarg "database")',
                    )
            self.database = os.path.join(outDir, "resultsDb_sqlite.db")
        else:
            # Using non-default database, but may also specify directory root.
            if outDir is not None:
                database = os.path.join(outDir, database)
            self.database = database
        # If this is a new file, then we should record date and version later.
        needs_version = not os.path.isfile(self.database)

        # Connect to the specified file; this will create the database if it doesn't exist.
        already_file = os.path.isfile(self.database)
        dbAddress = url.URL.create(self.driver, database=self.database)

        engine = create_engine(dbAddress, echo=verbose)
        self.Session = sessionmaker(bind=engine)
        self.open()
        # Create the tables, if they don't already exist.
        if ~already_file:
            try:
                Base.metadata.create_all(engine)
            except DatabaseError:
                raise ValueError(
                    "Cannot create a %s database at %s. Check directory exists."
                    % (self.driver, self.database)
                )
        self.slen = 1024
        # Check if we have a database matching this schema (with metricInfoLabel)
        query = "select * from metrics limit 1"
        cols = self.session.execute(query)._metadata.keys
        if "metricInfoLabel" not in cols:
            self.updateDatabase()

        # record the version we are on
        if needs_version:
            vers = rsVersion.__version__
            rundate = datetime.datetime.now().strftime("%Y-%m-%d")
            versioninfo = VersionRow(version=vers, rundate=rundate)
            self.session.add(versioninfo)
            self.session.commit()

        self.close()

    def updateDatabase(self):
        """Update the resultsDb from 'metricMetaata' to 'metricInfoLabel'

        This updates resultsDb to work with the current version of MAF, including RunComparison and showMaf.
        There is also a 'downgradeDatabase' to revert to the older style with 'metricMetadata.
        """
        warnings.warn(
            "Updating database to match new schema with metricInfoLabel."
            "Undo with self.downgradeDatabase if necessary (for older maf versions)."
        )
        query = "alter table metrics rename column metricMetadata to metricInfoLabel"
        self.session.execute(query)
        self.session.commit()
        self.close()

    def downgradeDatabase(self):
        """Update the resultsDb from 'metricInfoLabel' to 'metricMetadata'

        This updates resultsDb to work with older versions of MAF.
        There is also a 'upgradeDatabase' to update to the newer style with 'metricInfoLabel.
        """
        warnings.warn(
            "Found a version of the resultsDb which is using metricMetadata not metricInfoLabel.\n"
            " Running an automatic update!\n"
            " Note that this can be undone by running ResultsDb.downgradeDatabase"
        )
        query = "alter table metrics rename column metricInfoLabel to metricMetadata"
        self.session.execute(query)
        self.session.commit()
        self.close()

    def open(self):
        """
        Open connection to database
        """
        self.session = self.Session()
        self.session.expire_on_commit = False

    def close(self):
        """
        Close connection to database.
        """
        self.session.close()

    def updateMetric(
        self,
        metricName,
        slicerName,
        simDataName,
        sqlConstraint,
        metricInfoLabel,
        metricDataFile,
    ):
        """
        Add a row to or update a row in the metrics table.

        Parameters
        ----------
        metricName : `str`
            Name of the Metric
        slicerName : `str`
            Name of the Slicer
        simDataName : `str`
            Name of the simulation (runName, simName, simDataName..)
        sqlConstraint : `str`
            Constraint relevant for the metric bundle
        metricInfoLabel : `str`
            Information associated with the metric. Could be derived from the sqlconstraint or could
            be a more descriptive version, specified by the user.
        metricDataFile : `str`
            The data file the metric bundle output is stored in.

        Returns
        -------
        metricId : `int`
            The Id number of this metric in the metrics table.

        If same metric (same metricName, slicerName, simDataName, sqlConstraint, infoLabel)
        already exists, it does nothing.
        """
        self.open()
        if simDataName is None:
            simDataName = "NULL"
        if sqlConstraint is None:
            sqlConstraint = "NULL"
        if metricInfoLabel is None:
            metricInfoLabel = "NULL"
        if metricDataFile is None:
            metricDataFile = "NULL"
        # Check if metric has already been added to database.
        prev = (
            self.session.query(MetricRow)
            .filter_by(
                metricName=metricName,
                slicerName=slicerName,
                simDataName=simDataName,
                metricInfoLabel=metricInfoLabel,
                sqlConstraint=sqlConstraint,
            )
            .all()
        )
        if len(prev) == 0:
            metricinfo = MetricRow(
                metricName=metricName,
                slicerName=slicerName,
                simDataName=simDataName,
                sqlConstraint=sqlConstraint,
                metricInfoLabel=metricInfoLabel,
                metricDataFile=metricDataFile,
            )
            self.session.add(metricinfo)
            self.session.commit()
        else:
            metricinfo = prev[0]
        self.close()

        return metricinfo.metricId

    def updateDisplay(self, metricId, displayDict, overwrite=True):
        """
        Add a row to or update a row in the displays table.

        Parameters
        ----------
        metricId : `int`
            The metricID for this metric bundle in the metrics table
        displayDict : `dict`
            Dictionary containing the display info (group/subgroup/order/caption)
        overwrite : `bool`, opt
            Replaces existing row with same metricId if overwrite is True (default=True).
        """
        # Because we want to maintain 1-1 relationship between metricId's and displayDict's:
        # First check if a display line is present with this metricID.
        self.open()
        displayinfo = self.session.query(DisplayRow).filter_by(metricId=metricId).all()
        if len(displayinfo) > 0:
            if overwrite:
                for d in displayinfo:
                    self.session.delete(d)
            else:
                return
        # Then go ahead and add new displayDict.
        for k in displayDict:
            if displayDict[k] is None:
                displayDict[k] = "NULL"
        keys = ["group", "subgroup", "order", "caption"]
        for k in keys:
            if k not in displayDict:
                displayDict[k] = "NULL"
        if displayDict["order"] == "NULL":
            displayDict["order"] = 0
        displayGroup = displayDict["group"]
        displaySubgroup = displayDict["subgroup"]
        displayOrder = displayDict["order"]
        displayCaption = displayDict["caption"]
        if displayCaption.endswith("(auto)"):
            displayCaption = displayCaption.replace("(auto)", "", 1)
        displayinfo = DisplayRow(
            metricId=metricId,
            displayGroup=displayGroup,
            displaySubgroup=displaySubgroup,
            displayOrder=displayOrder,
            displayCaption=displayCaption,
        )
        self.session.add(displayinfo)
        self.session.commit()
        self.close()

    def updatePlot(self, metricId, plotType, plotFile, overwrite=False):
        """
        Add a row to or update a row in the plot table.

        Parameters
        ----------
        metricId : `int`
            The metric Id of this metric bundle in the metrics table
        plotType : `str`
            The type of this plot (oneDbinned data, healpix map, etc.)
        plotFile : `str`
            The filename for this plot
        overwrite : `bool`
            Replaces existing row with the same metricId and plotType, if True.
            Default False, in which case additional plot is added to output (e.g. with different range)
        """
        self.open()
        plotinfo = (
            self.session.query(PlotRow)
            .filter_by(metricId=metricId, plotType=plotType)
            .all()
        )
        if len(plotinfo) > 0 and overwrite:
            for p in plotinfo:
                self.session.delete(p)
        plotinfo = PlotRow(metricId=metricId, plotType=plotType, plotFile=plotFile)
        self.session.add(plotinfo)
        self.session.commit()
        self.close()

    def updateSummaryStat(self, metricId, summaryName, summaryValue):
        """
        Add a row to or update a row in the summary statistic table.

        Most summary statistics will be a simple name (string) + value (float) pair.
        For special summary statistics which must return multiple values, the base name
        can be provided as 'name', together with a np recarray as 'value', where the
        recarray also has 'name' and 'value' columns (and each name/value pair is then saved
        as a summary statistic associated with this same metricId).

        Parameters
        ----------
        metricId : `int`
            The metric Id of this metric bundle
        summaryName : `str`
            The name of this summary statistic
        summaryValue: : `float` or `numpy.ndarray`
            The value for this summary statistic.
            If this is a numpy recarray, then it should also have 'name' and 'value' columns to save
            each value to rows in the summary statistic table.
        """
        # Allow for special summary statistics which return data in a np structured array with
        #   'name' and 'value' columns.  (specificially needed for TableFraction summary statistic).
        self.open()
        if isinstance(summaryValue, np.ndarray):
            if ("name" in summaryValue.dtype.names) and (
                "value" in summaryValue.dtype.names
            ):
                for value in summaryValue:
                    sSuffix = value["name"]
                    if isinstance(sSuffix, bytes):
                        sSuffix = sSuffix.decode("utf-8")
                    else:
                        sSuffix = str(sSuffix)
                    summarystat = SummaryStatRow(
                        metricId=metricId,
                        summaryName=summaryName + " " + sSuffix,
                        summaryValue=value["value"],
                    )
                    self.session.add(summarystat)
                    self.session.commit()
            else:
                warnings.warn("Warning! Cannot save non-conforming summary statistic.")
        # Most summary statistics will be simple floats.
        else:
            if isinstance(summaryValue, float) or isinstance(summaryValue, int):
                summarystat = SummaryStatRow(
                    metricId=metricId,
                    summaryName=summaryName,
                    summaryValue=summaryValue,
                )
                self.session.add(summarystat)
                self.session.commit()
            else:
                warnings.warn(
                    "Warning! Cannot save summary statistic that is not a simple float or int"
                )
        self.close()

    def getMetricId(
        self, metricName, slicerName=None, metricInfoLabel=None, simDataName=None
    ):
        """Find metric bundle Ids from the metric table.

        Parameters
        ----------
        metricName : `str`
            Name of the Metric
        slicerName : `str`, opt
            Name of the Slicer to match
        metricInfoLabel : `str`, opt
            Metadata value to match
        simDataName : `str`, opt
            Name of the simulation (simDataName) to match

        Returns
        -------
        metricId : `list` of `int`
            List of matching metricIds
        """
        self.open()
        metricId = []
        query = self.session.query(
            MetricRow.metricId,
            MetricRow.metricName,
            MetricRow.slicerName,
            MetricRow.metricInfoLabel,
            MetricRow.simDataName,
        ).filter(MetricRow.metricName == metricName)
        if slicerName is not None:
            query = query.filter(MetricRow.slicerName == slicerName)
        if metricInfoLabel is not None:
            query = query.filter(MetricRow.metricInfoLabel == metricInfoLabel)
        if simDataName is not None:
            query = query.filter(MetricRow.simDataName == simDataName)
        query = query.order_by(MetricRow.slicerName, MetricRow.metricInfoLabel)
        for m in query:
            metricId.append(m.metricId)
        self.close()
        return metricId

    def getMetricIdLike(
        self,
        metricNameLike=None,
        slicerNameLike=None,
        metricInfoLabelLike=None,
        simDataName=None,
    ):
        """Find metric bundle Ids from the metric table, but search for names 'like' the values.
        (instead of a strict match from getMetricId).

        Parameters
        ----------
        metricName : `str`
            Partial name of the Metric
        slicerName : `str`, opt
            Partial name of the Slicer to match
        metricInfoLabel : `str`, opt
            Partial info_label value to match
        simDataName : `str`, opt
            Name of the simulation (simDataName) to match (exact)

        Returns
        -------
        metricId : `list` of `int`
            List of matching metricIds
        """
        self.open()
        metricId = []
        query = self.session.query(
            MetricRow.metricId,
            MetricRow.metricName,
            MetricRow.slicerName,
            MetricRow.metricInfoLabel,
            MetricRow.simDataName,
        )
        if metricNameLike is not None:
            query = query.filter(MetricRow.metricName.like(f"%{str(metricNameLike)}%"))
        if slicerNameLike is not None:
            query = query.filter(MetricRow.slicerName.like(f"%{str(slicerNameLike)}%"))
        if metricInfoLabelLike is not None:
            query = query.filter(
                MetricRow.metricInfoLabel.like(f"%{str(metricInfoLabelLike)}%")
            )
        if simDataName is not None:
            query = query.filter(MetricRow.simDataName == simDataName)
        for m in query:
            metricId.append(m.metricId)
        self.close()
        return metricId

    def getAllMetricIds(self):
        """
        Return a list of all metricIds.
        """
        self.open()
        metricIds = []
        for m in self.session.query(MetricRow.metricId).all():
            metricIds.append(m.metricId)
        self.close()
        return metricIds

    @staticmethod
    def buildSummaryName(metricName, metricInfoLabel, slicerName, summaryStatName=None):
        """Standardize a complete summary metric name, combining the metric + slicer + summary + info_label"""
        if metricInfoLabel is None:
            metricInfoLabel = ""
        if slicerName is None:
            slicerName = ""
        sName = summaryStatName
        if sName == "Identity" or sName == "Id" or sName == "Count" or sName is None:
            sName = ""
        slName = slicerName
        if slName == "UniSlicer":
            slName = ""
        name = (
            " ".join([sName, metricName, metricInfoLabel, slName])
            .rstrip(" ")
            .lstrip(" ")
        )
        name.replace(",", "")
        return name

    def getSummaryStats(
        self,
        metricId=None,
        summaryName=None,
        summaryNameLike=None,
        summaryNameNotLike=None,
        withSimName=False,
    ):
        """
        Get the summary stats (optionally for metricId list).
        Optionally, also specify the summary metric name.
        Returns a numpy array of the metric information + summary statistic information.

        Parameters
        ----------
        metricId : `int` or `list` of `int`
            Metric bundle Ids to match from the metric table
        summaryName : `str`, opt
            Match this summary statistic name exactly.
        summaryNameLike : `str`, opt
            Partial match to this summary statistic name.
        summaryNameNotLike : `str`, opt
            Exclude summary statistics with summary names like this.
        withSimName : `bool`, opt
            If True, add the simDataName to the returned numpy recarray.

        Returns
        -------
        summarystats : `np.recarray`
            Numpy recarray containing the selected summary statistic information.
        """
        if metricId is None:
            metricId = self.getAllMetricIds()
        if not hasattr(metricId, "__iter__"):
            metricId = [
                metricId,
            ]
        summarystats = []
        self.open()
        for mid in metricId:
            # Join the metric table and the summarystat table, based on the metricID (the second filter)
            query = (
                self.session.query(MetricRow, SummaryStatRow)
                .filter(MetricRow.metricId == mid)
                .filter(MetricRow.metricId == SummaryStatRow.metricId)
            )
            if summaryName is not None:
                query = query.filter(SummaryStatRow.summaryName == str(summaryName))
            if summaryNameLike is not None:
                query = query.filter(
                    SummaryStatRow.summaryName.like(f"%{str(summaryNameLike)}%")
                )
            if summaryNameNotLike is not None:
                if isinstance(summaryNameNotLike, list):
                    for s in summaryNameNotLike:
                        query = query.filter(
                            ~SummaryStatRow.summaryName.like(f"%{str(s)}%")
                        )
                else:
                    query = query.filter(
                        ~SummaryStatRow.summaryName.like(f"%{str(summaryNameNotLike)}%")
                    )
            for m, s in query:
                long_name = self.buildSummaryName(
                    m.metricName, m.metricInfoLabel, m.slicerName, s.summaryName
                )
                vals = (
                    m.metricId,
                    long_name,
                    m.metricName,
                    m.slicerName,
                    m.metricInfoLabel,
                    s.summaryName,
                    s.summaryValue,
                )
                if withSimName:
                    vals += (m.simDataName,)
                summarystats.append(vals)
        # Convert to numpy array.
        dtype_list = [
            ("metricId", int),
            ("summaryName", str, self.slen),
            ("metricName", str, self.slen),
            ("slicerName", str, self.slen),
            ("metricInfoLabel", str, self.slen),
            ("summaryMetric", str, self.slen),
            ("summaryValue", float),
        ]
        if withSimName:
            dtype_list += [("simDataName", str, self.slen)]
        dtype = np.dtype(dtype_list)
        summarystats = np.array(summarystats, dtype)
        self.close()
        return summarystats

    def getPlotFiles(self, metricId=None, withSimName=False):
        """
        Return the metricId, name, info_label, and all plot info (optionally for metricId list).
        Returns a numpy array of the metric information + plot file names.

        Parameters
        ----------
        metricId : `int`  `list`, or `None`
            If None, plots for all metrics are returned. Otherwise, only plots
            corresponding to the supplied metric ID or IDs are returned
        withSimName : `bool`
            If True, include the run name in the fields returned

        Returns
        -------
        plotFiles : `numpy.recarray`
            ``metricId``
                The metric ID
            ``metricName``
                The metric name
            ``metricInfoLabel``
                info_label extracted from the sql constraint (usually the filter)
            ``plotType``
                The plot type
            ``plotFile``
                The full plot file (pdf by default)
            ``thumbFile``
                A plot thumbnail file name (png)
            ``simDataName``
                The name of the run plotted (if `withSimName` was `True`)
        """
        if metricId is None:
            metricId = self.getAllMetricIds()
        if not hasattr(metricId, "__iter__"):
            metricId = [
                metricId,
            ]
        self.open()
        plotFiles = []
        for mid in metricId:
            # Join the metric table and the plot table based on the metricID (the second filter does the join)
            query = (
                self.session.query(MetricRow, PlotRow)
                .filter(MetricRow.metricId == mid)
                .filter(MetricRow.metricId == PlotRow.metricId)
            )
            for m, p in query:
                # The plotFile typically ends with .pdf (but the rest of name can have '.' or '_')
                thumbfile = "thumb." + ".".join(p.plotFile.split(".")[:-1]) + ".png"
                plot_file_fields = (
                    m.metricId,
                    m.metricName,
                    m.metricInfoLabel,
                    p.plotType,
                    p.plotFile,
                    thumbfile,
                )
                if withSimName:
                    plot_file_fields += (m.simDataName,)
                plotFiles.append(plot_file_fields)

        # Convert to numpy array.
        dtype_list = [
            ("metricId", int),
            ("metricName", str, self.slen),
            ("metricInfoLabel", str, self.slen),
            ("plotType", str, self.slen),
            ("plotFile", str, self.slen),
            ("thumbFile", str, self.slen),
        ]

        if withSimName:
            dtype_list += [("simDataName", str, self.slen)]
        dtype = np.dtype(dtype_list)

        plotFiles = np.array(plotFiles, dtype)
        self.close()
        return plotFiles

    def getMetricDataFiles(self, metricId=None):
        """
        Get the metric data filenames for all or a single metric.
        Returns a list.
        """
        if metricId is None:
            metricId = self.getAllMetricIds()
        self.open()
        if not hasattr(metricId, "__iter__"):
            metricId = [
                metricId,
            ]
        dataFiles = []
        for mid in metricId:
            for m in (
                self.session.query(MetricRow).filter(MetricRow.metricId == mid).all()
            ):
                dataFiles.append(m.metricDataFile)
        self.close()
        return dataFiles

    def getMetricInfo(self, metricId=None, withSimName=False):
        """Get the simple metric info, without display information.

        Parameters
        ----------
        metricId : `int`  `list`, or `None`
            If None, data for all metrics are returned. Otherwise, only data
            corresponding to the supplied metric ID or IDs are returned
        withSimName : `bool`
            If True, include the run name in the fields returned

        Returns
        -------
        plotFiles : `numpy.recarray`
            ``metricId``
                The metric ID
            ``metricName``
                The metric name
            ``baseMetricNames``
                The base metric names
            ``slicerName``
                The name of the slicer used in the bundleGroup
            ``sqlConstraint``
                The full sql constraint used in the bundleGroup
            ``metricInfoLabel``
                Metadata extracted from the `sqlConstraint` (usually the filter)
            ``metricDataFile``
                The file name of the file with the metric data itself.
            ``simDataName``
                The name of the run plotted (if `withSimName` was `True`)

        """
        if metricId is None:
            metricId = self.getAllMetricIds()
        if not hasattr(metricId, "__iter__"):
            metricId = [
                metricId,
            ]
        self.open()
        metricInfo = []
        for mId in metricId:
            # Query for all rows in metrics and displays that match any of the metricIds.
            query = self.session.query(MetricRow).filter(MetricRow.metricId == mId)
            for m in query:
                baseMetricName = m.metricName.split("_")[0]
                mInfo = (
                    m.metricId,
                    m.metricName,
                    baseMetricName,
                    m.slicerName,
                    m.sqlConstraint,
                    m.metricInfoLabel,
                    m.metricDataFile,
                )
                if withSimName:
                    mInfo += (m.simDataName,)

                metricInfo.append(mInfo)
        # Convert to numpy array.
        dtype_list = [
            ("metricId", int),
            ("metricName", str, self.slen),
            ("baseMetricNames", str, self.slen),
            ("slicerName", str, self.slen),
            ("sqlConstraint", str, self.slen),
            ("metricInfoLabel", str, self.slen),
            ("metricDataFile", str, self.slen),
        ]
        if withSimName:
            dtype_list += [("simDataName", str, self.slen)]
        dtype = np.dtype(dtype_list)
        metricInfo = np.array(metricInfo, dtype)
        self.close()
        return metricInfo

    def getMetricDisplayInfo(self, metricId=None):
        """
        Get the contents of the metrics and displays table, together with the 'basemetricname'
        (optionally, for metricId list).
        Returns a numpy array of the metric information + display information.

        One underlying assumption here is that all metrics have some display info.
        In newer batches, this may not be the case, as the display info gets auto-generated when the
        metric is plotted.
        """
        if metricId is None:
            metricId = self.getAllMetricIds()
        if not hasattr(metricId, "__iter__"):
            metricId = [
                metricId,
            ]
        self.open()
        metricInfo = []
        for mId in metricId:
            # Query for all rows in metrics and displays that match any of the metricIds.
            query = (
                self.session.query(MetricRow, DisplayRow)
                .filter(MetricRow.metricId == mId)
                .filter(MetricRow.metricId == DisplayRow.metricId)
            )
            for m, d in query:
                baseMetricName = m.metricName.split("_")[0]
                mInfo = (
                    m.metricId,
                    m.metricName,
                    baseMetricName,
                    m.slicerName,
                    m.sqlConstraint,
                    m.metricInfoLabel,
                    m.metricDataFile,
                    d.displayGroup,
                    d.displaySubgroup,
                    d.displayOrder,
                    d.displayCaption,
                )
                metricInfo.append(mInfo)
        # Convert to numpy array.
        dtype = np.dtype(
            [
                ("metricId", int),
                ("metricName", np.str_, self.slen),
                ("baseMetricNames", np.str_, self.slen),
                ("slicerName", np.str_, self.slen),
                ("sqlConstraint", np.str_, self.slen),
                ("metricInfoLabel", np.str_, self.slen),
                ("metricDataFile", np.str_, self.slen),
                ("displayGroup", np.str_, self.slen),
                ("displaySubgroup", np.str_, self.slen),
                ("displayOrder", float),
                ("displayCaption", np.str_, self.slen * 10),
            ]
        )
        metricInfo = np.array(metricInfo, dtype)
        self.close()
        return metricInfo

    def getSimDataName(self):
        """Return a list of the simDataNames for the metric bundles in the database."""
        self.open()
        query = self.session.query(MetricRow.simDataName.distinct()).all()
        simDataName = []
        for s in query:
            simDataName.append(s[0])
        self.close()
        return simDataName
