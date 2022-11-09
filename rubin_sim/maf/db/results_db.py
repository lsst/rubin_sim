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
    metric_id = Column(Integer, primary_key=True)
    metricName = Column(String)
    slicerName = Column(String)
    simDataName = Column(String)
    sqlConstraint = Column(String)
    metricInfoLabel = Column(String)
    metricDataFile = Column(String)

    def __repr__(self):
        return (
            "<Metric(metric_id='%d', metric_name='%s', slicerName='%s', "
            "simDataName='%s', sql_constraint='%s', metricInfoLabel='%s', metricDataFile='%s')>"
        ) % (
            self.metric_id,
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
    metric_id = Column(Integer, ForeignKey("metrics.metric_id"))
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
    metric_id = Column(Integer, ForeignKey("metrics.metric_id"))
    plotType = Column(String)
    plotFile = Column(String)
    metric = relationship("MetricRow", backref=backref("plots", order_by=plotId))

    def __repr__(self):
        return "<Plot(metric_id='%d', plot_type='%s', plot_file='%s')>" % (
            self.metric_id,
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
    metric_id = Column(Integer, ForeignKey("metrics.metric_id"))
    summaryName = Column(String)
    summaryValue = Column(Float)
    metric = relationship("MetricRow", backref=backref("summarystats", order_by=statId))

    def __repr__(self):
        return "<SummaryStat(metric_id='%d', summaryName='%s', summaryValue='%f')>" % (
            self.metric_id,
            self.summaryName,
            self.summaryValue,
        )


class ResultsDb(object):
    """The ResultsDb is a sqlite database containing information on the metrics run via MAF,
    the plots created, the display information (such as captions), and any summary statistics output.
    """

    def __init__(self, out_dir=None, database=None, verbose=False):
        """
        Instantiate the results database, creating metrics, plots and summarystats tables.
        """
        # We now require results_db to be a sqlite file (for simplicity). Leaving as attribute though.
        self.driver = "sqlite"
        # Connect to database
        # for sqlite, connecting to non-existent database creates it automatically
        if database is None:
            # Using default value for database name, should specify directory.
            if out_dir is None:
                out_dir = "."
            # Check for output directory, make if needed.
            if not os.path.isdir(out_dir):
                try:
                    os.makedirs(out_dir)
                except OSError as msg:
                    raise OSError(
                        msg,
                        "\n  (If this was the database file (not out_dir), "
                        'remember to use kwarg "database")',
                    )
            self.database = os.path.join(out_dir, "resultsDb_sqlite.db")
        else:
            # Using non-default database, but may also specify directory root.
            if out_dir is not None:
                database = os.path.join(out_dir, database)
            self.database = database
        # If this is a new file, then we should record date and version later.
        needs_version = not os.path.isfile(self.database)

        # Connect to the specified file; this will create the database if it doesn't exist.
        already_file = os.path.isfile(self.database)
        db_address = url.URL.create(self.driver, database=self.database)

        engine = create_engine(db_address, echo=verbose)
        self.Session = sessionmaker(bind=engine)
        self.open()
        # Create the tables, if they don't already exist.
        if not already_file:
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
            self.update_database()

        # record the version we are on
        if needs_version:
            vers = rsVersion.__version__
            rundate = datetime.datetime.now().strftime("%Y-%m-%d")
            versioninfo = VersionRow(version=vers, rundate=rundate)
            self.session.add(versioninfo)
            self.session.commit()

        self.close()

    def update_database(self):
        """Update the results_db from 'metricMetaata' to 'metricInfoLabel'

        This updates results_db to work with the current version of MAF, including RunComparison and showMaf.
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

    def downgrade_database(self):
        """Update the results_db from 'metricInfoLabel' to 'metricMetadata'

        This updates results_db to work with older versions of MAF.
        There is also a 'upgradeDatabase' to update to the newer style with 'metricInfoLabel.
        """
        warnings.warn(
            "Found a version of the results_db which is using metricMetadata not metricInfoLabel.\n"
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

    def update_metric(
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
            Name of the simulation (run_name, simName, simDataName..)
        sqlConstraint : `str`
            Constraint relevant for the metric bundle
        metricInfoLabel : `str`
            Information associated with the metric. Could be derived from the sqlconstraint or could
            be a more descriptive version, specified by the user.
        metricDataFile : `str`
            The data file the metric bundle output is stored in.

        Returns
        -------
        metric_id : `int`
            The Id number of this metric in the metrics table.

        If same metric (same metric_name, slicerName, simDataName, sql_constraint, infoLabel)
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

        return metricinfo.metric_id

    def update_display(self, metric_id, display_dict, overwrite=True):
        """
        Add a row to or update a row in the displays table.

        Parameters
        ----------
        metric_id : `int`
            The metricID for this metric bundle in the metrics table
        display_dict : `dict`
            Dictionary containing the display info (group/subgroup/order/caption)
        overwrite : `bool`, opt
            Replaces existing row with same metric_id if overwrite is True (default=True).
        """
        # Because we want to maintain 1-1 relationship between metric_id's and display_dict's:
        # First check if a display line is present with this metricID.
        self.open()
        displayinfo = (
            self.session.query(DisplayRow).filter_by(metric_id=metric_id).all()
        )
        if len(displayinfo) > 0:
            if overwrite:
                for d in displayinfo:
                    self.session.delete(d)
            else:
                return
        # Then go ahead and add new display_dict.
        for k in display_dict:
            if display_dict[k] is None:
                display_dict[k] = "NULL"
        keys = ["group", "subgroup", "order", "caption"]
        for k in keys:
            if k not in display_dict:
                display_dict[k] = "NULL"
        if display_dict["order"] == "NULL":
            display_dict["order"] = 0
        displayGroup = display_dict["group"]
        displaySubgroup = display_dict["subgroup"]
        displayOrder = display_dict["order"]
        displayCaption = display_dict["caption"]
        if displayCaption.endswith("(auto)"):
            displayCaption = displayCaption.replace("(auto)", "", 1)
        displayinfo = DisplayRow(
            metric_id=metric_id,
            displayGroup=displayGroup,
            displaySubgroup=displaySubgroup,
            displayOrder=displayOrder,
            displayCaption=displayCaption,
        )
        self.session.add(displayinfo)
        self.session.commit()
        self.close()

    def update_plot(self, metric_id, plot_type, plot_file, overwrite=False):
        """
        Add a row to or update a row in the plot table.

        Parameters
        ----------
        metric_id : `int`
            The metric Id of this metric bundle in the metrics table
        plot_type : `str`
            The type of this plot (oneDbinned data, healpix map, etc.)
        plot_file : `str`
            The filename for this plot
        overwrite : `bool`
            Replaces existing row with the same metric_id and plot_type, if True.
            Default False, in which case additional plot is added to output (e.g. with different range)
        """
        self.open()
        plotinfo = (
            self.session.query(PlotRow)
            .filter_by(metric_id=metric_id, plotType=plot_type)
            .all()
        )
        if len(plotinfo) > 0 and overwrite:
            for p in plotinfo:
                self.session.delete(p)
        plotinfo = PlotRow(metric_id=metric_id, plotType=plot_type, plotFile=plot_file)
        self.session.add(plotinfo)
        self.session.commit()
        self.close()

    def update_summary_stat(self, metric_id, summary_name, summary_value):
        """
        Add a row to or update a row in the summary statistic table.

        Most summary statistics will be a simple name (string) + value (float) pair.
        For special summary statistics which must return multiple values, the base name
        can be provided as 'name', together with a np recarray as 'value', where the
        recarray also has 'name' and 'value' columns (and each name/value pair is then saved
        as a summary statistic associated with this same metric_id).

        Parameters
        ----------
        metric_id : `int`
            The metric Id of this metric bundle
        summary_name : `str`
            The name of this summary statistic
        summary_value: : `float` or `numpy.ndarray`
            The value for this summary statistic.
            If this is a numpy recarray, then it should also have 'name' and 'value' columns to save
            each value to rows in the summary statistic table.
        """
        # Allow for special summary statistics which return data in a np structured array with
        #   'name' and 'value' columns.  (specificially needed for TableFraction summary statistic).
        self.open()
        if isinstance(summary_value, np.ndarray):
            if ("name" in summary_value.dtype.names) and (
                "value" in summary_value.dtype.names
            ):
                for value in summary_value:
                    sSuffix = value["name"]
                    if isinstance(sSuffix, bytes):
                        sSuffix = sSuffix.decode("utf-8")
                    else:
                        sSuffix = str(sSuffix)
                    summarystat = SummaryStatRow(
                        metric_id=metric_id,
                        summaryName=summary_name + " " + sSuffix,
                        summaryValue=value["value"],
                    )
                    self.session.add(summarystat)
                    self.session.commit()
            else:
                warnings.warn("Warning! Cannot save non-conforming summary statistic.")
        # Most summary statistics will be simple floats.
        else:
            if isinstance(summary_value, float) or isinstance(summary_value, int):
                summarystat = SummaryStatRow(
                    metric_id=metric_id,
                    summaryName=summary_name,
                    summaryValue=summary_value,
                )
                self.session.add(summarystat)
                self.session.commit()
            else:
                warnings.warn(
                    "Warning! Cannot save summary statistic that is not a simple float or int"
                )
        self.close()

    def get_metric_id(
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
        metric_id : `list` of `int`
            List of matching metric_ids
        """
        self.open()
        metric_id = []
        query = self.session.query(
            MetricRow.metric_id,
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
            metric_id.append(m.metric_id)
        self.close()
        return metric_id

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
        metric_name : `str`
            Partial name of the Metric
        slicerName : `str`, opt
            Partial name of the Slicer to match
        metricInfoLabel : `str`, opt
            Partial info_label value to match
        simDataName : `str`, opt
            Name of the simulation (simDataName) to match (exact)

        Returns
        -------
        metric_id : `list` of `int`
            List of matching metric_ids
        """
        self.open()
        metric_id = []
        query = self.session.query(
            MetricRow.metric_id,
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
            metric_id.append(m.metric_id)
        self.close()
        return metric_id

    def getAllMetricIds(self):
        """
        Return a list of all metric_ids.
        """
        self.open()
        metric_ids = []
        for m in self.session.query(MetricRow.metric_id).all():
            metric_ids.append(m.metric_id)
        self.close()
        return metric_ids

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
        metric_id=None,
        summaryName=None,
        summaryNameLike=None,
        summaryNameNotLike=None,
        withSimName=False,
    ):
        """
        Get the summary stats (optionally for metric_id list).
        Optionally, also specify the summary metric name.
        Returns a numpy array of the metric information + summary statistic information.

        Parameters
        ----------
        metric_id : `int` or `list` of `int`
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
        if metric_id is None:
            metric_id = self.getAllMetricIds()
        if not hasattr(metric_id, "__iter__"):
            metric_id = [
                metric_id,
            ]
        summarystats = []
        self.open()
        for mid in metric_id:
            # Join the metric table and the summarystat table, based on the metricID (the second filter)
            query = (
                self.session.query(MetricRow, SummaryStatRow)
                .filter(MetricRow.metric_id == mid)
                .filter(MetricRow.metric_id == SummaryStatRow.metric_id)
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
                    m.metric_id,
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
            ("metric_id", int),
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

    def getPlotFiles(self, metric_id=None, withSimName=False):
        """
        Return the metric_id, name, info_label, and all plot info (optionally for metric_id list).
        Returns a numpy array of the metric information + plot file names.

        Parameters
        ----------
        metric_id : `int`  `list`, or `None`
            If None, plots for all metrics are returned. Otherwise, only plots
            corresponding to the supplied metric ID or IDs are returned
        withSimName : `bool`
            If True, include the run name in the fields returned

        Returns
        -------
        plotFiles : `numpy.recarray`
            ``metric_id``
                The metric ID
            ``metric_name``
                The metric name
            ``metricInfoLabel``
                info_label extracted from the sql constraint (usually the filter)
            ``plot_type``
                The plot type
            ``plot_file``
                The full plot file (pdf by default)
            ``thumb_file``
                A plot thumbnail file name (png)
            ``simDataName``
                The name of the run plotted (if `withSimName` was `True`)
        """
        if metric_id is None:
            metric_id = self.getAllMetricIds()
        if not hasattr(metric_id, "__iter__"):
            metric_id = [
                metric_id,
            ]
        self.open()
        plotFiles = []
        for mid in metric_id:
            # Join the metric table and the plot table based on the metricID (the second filter does the join)
            query = (
                self.session.query(MetricRow, PlotRow)
                .filter(MetricRow.metric_id == mid)
                .filter(MetricRow.metric_id == PlotRow.metric_id)
            )
            for m, p in query:
                # The plot_file typically ends with .pdf (but the rest of name can have '.' or '_')
                thumb_file = "thumb." + ".".join(p.plotFile.split(".")[:-1]) + ".png"
                plot_file_fields = (
                    m.metric_id,
                    m.metricName,
                    m.metricInfoLabel,
                    p.plotType,
                    p.plotFile,
                    thumb_file,
                )
                if withSimName:
                    plot_file_fields += (m.simDataName,)
                plotFiles.append(plot_file_fields)

        # Convert to numpy array.
        dtype_list = [
            ("metric_id", int),
            ("metric_name", str, self.slen),
            ("metricInfoLabel", str, self.slen),
            ("plot_type", str, self.slen),
            ("plot_file", str, self.slen),
            ("thumb_file", str, self.slen),
        ]

        if withSimName:
            dtype_list += [("simDataName", str, self.slen)]
        dtype = np.dtype(dtype_list)

        plotFiles = np.array(plotFiles, dtype)
        self.close()
        return plotFiles

    def getMetricDataFiles(self, metric_id=None):
        """
        Get the metric data filenames for all or a single metric.
        Returns a list.
        """
        if metric_id is None:
            metric_id = self.getAllMetricIds()
        self.open()
        if not hasattr(metric_id, "__iter__"):
            metric_id = [
                metric_id,
            ]
        dataFiles = []
        for mid in metric_id:
            for m in (
                self.session.query(MetricRow).filter(MetricRow.metric_id == mid).all()
            ):
                dataFiles.append(m.metricDataFile)
        self.close()
        return dataFiles

    def getMetricInfo(self, metric_id=None, withSimName=False):
        """Get the simple metric info, without display information.

        Parameters
        ----------
        metric_id : `int`  `list`, or `None`
            If None, data for all metrics are returned. Otherwise, only data
            corresponding to the supplied metric ID or IDs are returned
        withSimName : `bool`
            If True, include the run name in the fields returned

        Returns
        -------
        plotFiles : `numpy.recarray`
            ``metric_id``
                The metric ID
            ``metric_name``
                The metric name
            ``baseMetricNames``
                The base metric names
            ``slicerName``
                The name of the slicer used in the bundleGroup
            ``sql_constraint``
                The full sql constraint used in the bundleGroup
            ``metricInfoLabel``
                Metadata extracted from the `sql_constraint` (usually the filter)
            ``metricDataFile``
                The file name of the file with the metric data itself.
            ``simDataName``
                The name of the run plotted (if `withSimName` was `True`)

        """
        if metric_id is None:
            metric_id = self.getAllMetricIds()
        if not hasattr(metric_id, "__iter__"):
            metric_id = [
                metric_id,
            ]
        self.open()
        metricInfo = []
        for mId in metric_id:
            # Query for all rows in metrics and displays that match any of the metric_ids.
            query = self.session.query(MetricRow).filter(MetricRow.metric_id == mId)
            for m in query:
                baseMetricName = m.metricName.split("_")[0]
                mInfo = (
                    m.metric_id,
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
            ("metric_id", int),
            ("metric_name", str, self.slen),
            ("baseMetricNames", str, self.slen),
            ("slicerName", str, self.slen),
            ("sql_constraint", str, self.slen),
            ("metricInfoLabel", str, self.slen),
            ("metricDataFile", str, self.slen),
        ]
        if withSimName:
            dtype_list += [("simDataName", str, self.slen)]
        dtype = np.dtype(dtype_list)
        metricInfo = np.array(metricInfo, dtype)
        self.close()
        return metricInfo

    def getMetricDisplayInfo(self, metric_id=None):
        """
        Get the contents of the metrics and displays table, together with the 'basemetricname'
        (optionally, for metric_id list).
        Returns a numpy array of the metric information + display information.

        One underlying assumption here is that all metrics have some display info.
        In newer batches, this may not be the case, as the display info gets auto-generated when the
        metric is plotted.
        """
        if metric_id is None:
            metric_id = self.getAllMetricIds()
        if not hasattr(metric_id, "__iter__"):
            metric_id = [
                metric_id,
            ]
        self.open()
        metricInfo = []
        for mId in metric_id:
            # Query for all rows in metrics and displays that match any of the metric_ids.
            query = (
                self.session.query(MetricRow, DisplayRow)
                .filter(MetricRow.metric_id == mId)
                .filter(MetricRow.metric_id == DisplayRow.metric_id)
            )
            for m, d in query:
                baseMetricName = m.metricName.split("_")[0]
                mInfo = (
                    m.metric_id,
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
                ("metric_id", int),
                ("metric_name", np.str_, self.slen),
                ("baseMetricNames", np.str_, self.slen),
                ("slicerName", np.str_, self.slen),
                ("sql_constraint", np.str_, self.slen),
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
