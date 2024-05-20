__all__ = (
    "MetricRow",
    "DisplayRow",
    "PlotRow",
    "SummaryStatRow",
    "VersionRow",
    "ResultsDb",
)

import datetime
import os
import time
import warnings
from sqlite3 import OperationalError

import numpy as np
from sqlalchemy import Column, Float, ForeignKey, Integer, String, create_engine
from sqlalchemy.engine import url
from sqlalchemy.exc import DatabaseError
from sqlalchemy.orm import backref, declarative_base, relationship, sessionmaker
from sqlalchemy.sql import text

import rubin_sim.version as rsVersion

Base = declarative_base()


class MetricRow(Base):
    """
    Define contents and format of metric list table.

    (Table to list all metrics, their info_label, and their output data files).
    """

    __tablename__ = "metrics"
    # Define columns in metric list table.
    metric_id = Column(Integer, primary_key=True)
    metric_name = Column(String)
    slicer_name = Column(String)
    run_name = Column(String)
    sql_constraint = Column(String)
    metric_info_label = Column(String)
    metric_datafile = Column(String)

    def __repr__(self):
        return (
            "<Metric(metric_id='%d', metric_name='%s', slicer_name='%s', "
            "run_name='%s', sql_constraint='%s', metric_info_label='%s', metric_datafile='%s')>"
        ) % (
            self.metric_id,
            self.metric_name,
            self.slicer_name,
            self.run_name,
            self.sql_constraint,
            self.metric_info_label,
            self.metric_datafile,
        )


class VersionRow(Base):
    """"""

    __tablename__ = "version"
    version_id = Column(Integer, primary_key=True)
    version = Column(String)
    run_date = Column(String)

    def __repr__(self):
        return ("<Version(version='%s', run_date='%s')>") % (
            self.version,
            self.run_date,
        )


class DisplayRow(Base):
    """
    Define contents and format of the displays table.

    (Table to list the display properties for each metric.)
    """

    __tablename__ = "displays"
    display_id = Column(Integer, primary_key=True)
    metric_id = Column(Integer, ForeignKey("metrics.metric_id"))
    # Group for displaying metric (in webpages).
    display_group = Column(String)
    # Subgroup for displaying metric.
    display_subgroup = Column(String)
    # Order to display metric (within subgroup).
    display_order = Column(Float)
    # The figure caption.
    display_caption = Column(String)
    metric = relationship("MetricRow", backref=backref("displays", order_by=display_id))

    def __rep__(self):
        return (
            "<Display(display_group='%s', display_subgroup='%s', "
            "display_order='%.1f', display_caption='%s')>"
            % (
                self.display_group,
                self.display_subgroup,
                self.display_order,
                self.display_caption,
            )
        )


class PlotRow(Base):
    """
    Define contents and format of plot list table.

    (Table to list all plots,
    link them to relevant metrics in MetricList,
    and provide info on filename).
    """

    __tablename__ = "plots"
    # Define columns in plot list table.
    plot_id = Column(Integer, primary_key=True)
    # Matches metricID in MetricList table.
    metric_id = Column(Integer, ForeignKey("metrics.metric_id"))
    plot_type = Column(String)
    plot_file = Column(String)
    metric = relationship("MetricRow", backref=backref("plots", order_by=plot_id))

    def __repr__(self):
        return "<Plot(metric_id='%d', plot_type='%s', plot_file='%s')>" % (
            self.metric_id,
            self.plot_type,
            self.plot_file,
        )


class SummaryStatRow(Base):
    """
    Define contents and format of the summary statistics table.

    (Table to list and link summary stats to relevant metrics in MetricList,
    and provide summary stat name, value and potentially a comment).
    """

    __tablename__ = "summarystats"
    # Define columns in summary table.
    stat_id = Column(Integer, primary_key=True)
    # Matches metricID in MetricList table.
    metric_id = Column(Integer, ForeignKey("metrics.metric_id"))
    summary_name = Column(String)
    summary_value = Column(Float)
    metric = relationship("MetricRow", backref=backref("summarystats", order_by=stat_id))

    def __repr__(self):
        return "<SummaryStat(metric_id='%d', summary_name='%s', summary_value='%f')>" % (
            self.metric_id,
            self.summary_name,
            self.summary_value,
        )


class ResultsDb:
    """ResultsDb is a sqlite database containing information on the metrics
    run via MAF, the plots created, the display information (such as captions),
    and any summary statistics output.
    """

    def __init__(self, out_dir=None, database=None, verbose=False, timeout=180):
        """
        Set up the resultsDb database.
        """
        # We now require results_db to be a sqlite file (for simplicity).
        # Leaving as attribute though.
        self.driver = "sqlite"
        # connecting to non-existent database creates it automatically
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

        # Connect to the specified file;
        # this will create the database if it doesn't exist.
        already_file = os.path.isfile(self.database)
        db_address = url.URL.create(self.driver, database=self.database)

        engine = create_engine(db_address, echo=verbose, connect_args={"timeout": timeout})
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
        # Check if we have a database matching this schema
        # (with metric_info_label)
        query = text("select * from metrics limit 1")
        cols = self.session.execute(query)._metadata.keys
        if "sql_constraint" not in cols:
            self.update_database()

        # record the version and date MAF was run with/on
        if needs_version:
            vers = rsVersion.__version__
            run_date = datetime.datetime.now().strftime("%Y-%m-%d")
            versioninfo = VersionRow(version=vers, run_date=run_date)
            self.session.add(versioninfo)
            self.session.commit()

        self.close()

    def update_database(self):
        """Update the results_db from 'metricMetaData' to 'metric_info_label'
        and now also changing the camel case to snake case
        (metricId to metric_id, etc.).

        This updates results_db to work with the current version of MAF,
        including RunComparison and showMaf.
        There is also a 'downgrade_database' to revert to the older style
        with 'metricMetadata.
        """
        warnings.warn(
            "Updating database to match new schema."
            "Undo with self.downgrade_database if necessary (for older maf versions)."
        )
        query = text("select * from metrics limit 1")
        cols = self.session.execute(query)._metadata.keys
        if "metricMetadata" in cols:
            # if it's very old
            query = text("alter table metrics rename column metricMetadata to metric_info_label")
            self.session.execute(query)
            self.session.commit()
        # Check for metricId vs. metric_id separately,
        # as this change happened independently.
        if "metricId" in cols:
            for table in ["metrics", "summarystats", "plots", "displays"]:
                query = text(f"alter table {table} rename column metricId to metric_id")
                self.session.execute(query)
            self.session.commit()
        # Update to newest schema.
        if "run_name" not in cols:
            updates = {
                "metricName": "metric_name",
                "slicerName": "slicer_name",
                "simDataName": "run_name",
                "sqlConstraint": "sql_constraint",
                "metricInfoLabel": "metric_info_label",
                "metricDataFile": "metric_datafile",
            }
            for old, new in updates.items():
                query = text(f"alter table metrics rename column {old} to {new}")
                self.session.execute(query)
            self.session.commit()
            # update summarystat table
            updates = {
                "statId": "stat_id",
                "summaryName": "summary_name",
                "summaryValue": "summary_value",
            }
            for old, new in updates.items():
                query = text(f"alter table summarystats rename column {old} to {new}")
                self.session.execute(query)
            self.session.commit()
            # update plot table
            updates = {
                "plotId": "plot_id",
                "plotType": "plot_type",
                "plotFile": "plot_file",
            }
            for old, new in updates.items():
                query = text(f"alter table plots rename column {old} to {new}")
                self.session.execute(query)
            self.session.commit()
            # update display table
            updates = {
                "displayId": "display_id",
                "displayGroup": "display_group",
                "displaySubgroup": "display_subgroup",
                "displayOrder": "display_order",
                "displayCaption": "display_caption",
            }
            for old, new in updates.items():
                query = text(f"alter table displays rename column {old} to {new}")
                self.session.execute(query)
            self.session.commit()
            updates = {
                "verId": "version_id",
                "rundate": "run_date",
            }
            for old, new in updates.items():
                query = text(f"alter table version rename column {old} to {new}")
                self.session.execute(query)
            self.session.commit()

    def downgrade_database(self):
        """
        Downgrade resultsDb to work with v0.10<MAF< v1.0
        There is also a 'upgradeDatabase' to update to the newer
        style with 'metric_info_label.
        """
        self.open()
        warnings.warn("Downgrading MAF resultsDb to run with MAF < v1.0")
        updates = {
            "metricId": "metric_id",
            "metricName": "metric_name",
            "slicerName": "slicer_name",
            "simDataName": "run_name",
            "sqlConstraint": "sql_constraint",
            "metricInfoLabel": "metric_info_label",
            "metricDataFile": "metric_datafile",
        }
        for old, new in updates.items():
            query = text(f"alter table metrics rename column {new} to {old}")
            self.session.execute(query)
        self.session.commit()
        # update summarystat table
        updates = {
            "statId": "stat_id",
            "metricId": "metric_id",
            "summaryName": "summary_name",
            "summaryValue": "summary_value",
        }
        for old, new in updates.items():
            query = text(f"alter table summarystats rename column {new} to {old}")
            self.session.execute(query)
        self.session.commit()
        # update plot table
        updates = {
            "plotId": "plot_id",
            "metricId": "metric_id",
            "plotType": "plot_type",
            "plotFile": "plot_file",
        }
        for old, new in updates.items():
            query = text(f"alter table plots rename column {new} to {old}")
            self.session.execute(query)
        self.session.commit()
        # update display table
        updates = {
            "displayId": "display_id",
            "metricId": "metric_id",
            "displayGroup": "display_group",
            "displaySubgroup": "display_subgroup",
            "displayOrder": "display_order",
            "displayCaption": "display_caption",
        }
        for old, new in updates.items():
            query = text(f"alter table displays rename column {new} to {old}")
            self.session.execute(query)
        self.session.commit()
        updates = {
            "verId": "version_id",
            "rundate": "run_date",
        }
        for old, new in updates.items():
            query = text(f"alter table version rename column {new} to {old}")
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
        metric_name,
        slicer_name,
        run_name,
        sql_constraint,
        metric_info_label,
        metric_datafile,
    ):
        """
        Add a row to or update a row in the metrics table.

        Parameters
        ----------
        metric_name : `str`
            Name of the Metric
        slicer_name : `str`
            Name of the Slicer
        run_name : `str`
            Name of the simulation (run_name, simName, run_name..)
        sql_constraint : `str`
            sql_constraint relevant for the metric bundle
        metric_info_label : `str`
            Information associated with the metric.
            Could be derived from the sql_constraint or could
            be a more descriptive version, specified by the user.
        metric_datafile : `str`
            The data file the metric bundle output is stored in.

        Returns
        -------
        metric_id : `int`
            The Id number of this metric in the metrics table.

        If same metric (same metric_name, slicer_name, run_name,
        sql_constraint, infoLabel) already exists, it does nothing.
        """
        self.open()
        if run_name is None:
            run_name = "NULL"
        if sql_constraint is None:
            sql_constraint = "NULL"
        if metric_info_label is None:
            metric_info_label = "NULL"
        if metric_datafile is None:
            metric_datafile = "NULL"
        # Check if metric has already been added to database.
        prev = (
            self.session.query(MetricRow)
            .filter_by(
                metric_name=metric_name,
                slicer_name=slicer_name,
                run_name=run_name,
                metric_info_label=metric_info_label,
                sql_constraint=sql_constraint,
            )
            .all()
        )
        if len(prev) == 0:
            metricinfo = MetricRow(
                metric_name=metric_name,
                slicer_name=slicer_name,
                run_name=run_name,
                sql_constraint=sql_constraint,
                metric_info_label=metric_info_label,
                metric_datafile=metric_datafile,
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
            Dictionary containing the display info
            (group/subgroup/order/caption)
        overwrite : `bool`, opt
            Replaces existing row with same metric_id if overwrite is True.
        """
        # Because we want to maintain 1-1 relationship between
        # metric_id's and display_dict's: First check if a display line
        # is present with this metricID.
        self.open()
        displayinfo = self.session.query(DisplayRow).filter_by(metric_id=metric_id).all()
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
        display_group = display_dict["group"]
        display_subgroup = display_dict["subgroup"]
        display_order = display_dict["order"]
        display_caption = display_dict["caption"]
        if display_caption.endswith("(auto)"):
            display_caption = display_caption.replace("(auto)", "", 1)
        displayinfo = DisplayRow(
            metric_id=metric_id,
            display_group=display_group,
            display_subgroup=display_subgroup,
            display_order=display_order,
            display_caption=display_caption,
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
            If True, replaces existing row.
            If False, an additional plot is added to the output (e.g. with
            a different range of color values, etc).
        """
        self.open()
        plotinfo = self.session.query(PlotRow).filter_by(metric_id=metric_id, plot_type=plot_type).all()
        if len(plotinfo) > 0 and overwrite:
            for p in plotinfo:
                self.session.delete(p)
        plotinfo = PlotRow(metric_id=metric_id, plot_type=plot_type, plot_file=plot_file)
        self.session.add(plotinfo)
        self.session.commit()
        self.close()

    def update_summary_stat(self, metric_id, summary_name, summary_value, ntry=3, pause_time=100):
        """
        Add a row to or update a row in the summary statistic table.

        Most summary statistics will be a simple name (string) + value (float)
        pair. For special summary statistics which must return multiple values,
        the base name can be provided as 'name', together with a np.ndarray as
        'value', where the array also has 'name' and 'value' columns
        (and each name/value pair is then saved as a summary statistic
        associated with this same metric_id).

        Parameters
        ----------
        metric_id : `int`
            The metric Id of this metric bundle
        summary_name : `str`
            The name of this summary statistic
        summary_value: : `float` or `np.ndarray`
            The value for this summary statistic.
            If this is a np.ndarray, then it should also have 'name' and
            'value' columns to save each value to rows in the summary stats.
        ntry : `int`, opt
            The number of times to retry if database is locked.
        pause_time : `int`, opt
            Time to wait until trying again.
        """
        # Allow for special summary statistics which return data in a
        # np.ndarray with 'name' and 'value' columns.
        self.open()
        tries = 0
        if isinstance(summary_value, np.ndarray) and summary_value.dtype.names is not None:
            if ("name" in summary_value.dtype.names) and ("value" in summary_value.dtype.names):
                for value in summary_value:
                    sSuffix = value["name"]
                    if isinstance(sSuffix, bytes):
                        sSuffix = sSuffix.decode("utf-8")
                    else:
                        sSuffix = str(sSuffix)
                    summarystat = SummaryStatRow(
                        metric_id=metric_id,
                        summary_name=summary_name + " " + sSuffix,
                        summary_value=value["value"],
                    )
                    success = False
                    # This can hit a locked database if running in parallel
                    # have it try a few times before actually failing
                    # since nothing should be writing for a long time.
                    while (not success) & (tries < ntry):
                        try:
                            self.session.add(summarystat)
                            self.session.commit()
                            success = True
                        except OperationalError:
                            tries += 1
                            time.sleep(pause_time)
            else:
                warnings.warn("Warning! Cannot save non-conforming summary statistic.")
        # Most summary statistics will be simple floats.
        else:
            if isinstance(summary_value, float) or isinstance(summary_value, int):
                summarystat = SummaryStatRow(
                    metric_id=metric_id,
                    summary_name=summary_name,
                    summary_value=summary_value,
                )
                self.session.add(summarystat)
                self.session.commit()
            else:
                warnings.warn("Warning! Cannot save summary statistic that is not a simple float or int")
        self.close()

    def get_metric_id(self, metric_name, slicer_name=None, metric_info_label=None, run_name=None):
        """Find metric bundle Ids from the metric table.

        Parameters
        ----------
        metric_name : `str`
            Name of the Metric
        slicer_name : `str`, opt
            Name of the Slicer to match
        metric_info_label : `str`, opt
            Metadata value to match
        run_name : `str`, opt
            Name of the simulation (run_name) to match

        Returns
        -------
        metric_id : `list` of `int`
            List of matching metric_ids
        """
        self.open()
        metric_id = []
        query = self.session.query(
            MetricRow.metric_id,
            MetricRow.metric_name,
            MetricRow.slicer_name,
            MetricRow.metric_info_label,
            MetricRow.run_name,
        ).filter(MetricRow.metric_name == metric_name)
        if slicer_name is not None:
            query = query.filter(MetricRow.slicer_name == slicer_name)
        if metric_info_label is not None:
            query = query.filter(MetricRow.metric_info_label == metric_info_label)
        if run_name is not None:
            query = query.filter(MetricRow.run_name == run_name)
        query = query.order_by(MetricRow.slicer_name, MetricRow.metric_info_label)
        for m in query:
            metric_id.append(m.metric_id)
        self.close()
        return metric_id

    def get_metric_id_like(
        self,
        metric_name_like=None,
        slicer_name_like=None,
        metric_info_label_like=None,
        run_name=None,
    ):
        """Find metric bundle Ids from the metric table,
        but search for names 'like' the values.
        (instead of a strict match from get_metric_id).

        Parameters
        ----------
        metric_name : `str`
            Partial name of the Metric
        slicer_name : `str`, opt
            Partial name of the Slicer to match
        metric_info_label : `str`, opt
            Partial info_label value to match
        run_name : `str`, opt
            Name of the simulation (run_name) to match (exact)

        Returns
        -------
        metric_id : `list` of `int`
            List of matching metric_ids
        """
        self.open()
        metric_id = []
        query = self.session.query(
            MetricRow.metric_id,
            MetricRow.metric_name,
            MetricRow.slicer_name,
            MetricRow.metric_info_label,
            MetricRow.run_name,
        )
        if metric_name_like is not None:
            query = query.filter(MetricRow.metric_name.like(f"%{str(metric_name_like)}%"))
        if slicer_name_like is not None:
            query = query.filter(MetricRow.slicer_name.like(f"%{str(slicer_name_like)}%"))
        if metric_info_label_like is not None:
            query = query.filter(MetricRow.metric_info_label.like(f"%{str(metric_info_label_like)}%"))
        if run_name is not None:
            query = query.filter(MetricRow.run_name == run_name)
        for m in query:
            metric_id.append(m.metric_id)
        self.close()
        return metric_id

    def get_all_metric_ids(self):
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
    def build_summary_name(metric_name, metric_info_label, slicer_name, summary_stat_name=None):
        """Standardize a complete summary metric name,
        combining the metric + slicer + summary + info_label."""
        if metric_info_label is None:
            metric_info_label = ""
        if slicer_name is None:
            slicer_name = ""
        sName = summary_stat_name
        if sName == "Identity" or sName == "Id" or sName == "Count" or sName is None:
            sName = ""
        slName = slicer_name
        if slName == "UniSlicer":
            slName = ""
        name = " ".join([sName, metric_name, metric_info_label, slName]).rstrip(" ").lstrip(" ")
        name.replace(",", "")
        return name

    def get_summary_stats(
        self,
        metric_id=None,
        summary_name=None,
        summary_name_like=None,
        summary_name_notlike=None,
        with_sim_name=False,
    ):
        """
        Get the summary stats (optionally for metric_id list).
        Optionally, also specify the summary metric name.
        Returns a numpy array of the metric information +
        summary statistic information.

        Parameters
        ----------
        metric_id : `int` or `list` of `int`
            Metric bundle Ids to match from the metric table
        summary_name : `str`, opt
            Match this summary statistic name exactly.
        summary_name_like : `str`, opt
            Partial match to this summary statistic name.
        summary_name_notlike : `str`, opt
            Exclude summary statistics with summary names like this.
        with_sim_name : `bool`, opt
            If True, add the run_name to the returned numpy recarray.

        Returns
        -------
        summarystats : `np.recarray`
            Numpy recarray containing the selected summary stat information.
        """
        if metric_id is not None:
            if not hasattr(metric_id, "__iter__"):
                metric_id = [
                    metric_id,
                ]
        summarystats = []
        self.open()
        # Join the metric table and the summarystat table,
        # based on the metricID (the second filter)
        query = self.session.query(MetricRow, SummaryStatRow).filter(
            MetricRow.metric_id == SummaryStatRow.metric_id
        )
        if metric_id is not None:
            query = query.filter(MetricRow.metric_id.in_(metric_id))
        if summary_name is not None:
            query = query.filter(SummaryStatRow.summary_name == str(summary_name))
        if summary_name_like is not None:
            query = query.filter(SummaryStatRow.summary_name.like(f"%{str(summary_name_like)}%"))
        if summary_name_notlike is not None:
            if isinstance(summary_name_notlike, list):
                for s in summary_name_notlike:
                    query = query.filter(~SummaryStatRow.summary_name.like(f"%{str(s)}%"))
            else:
                query = query.filter(~SummaryStatRow.summary_name.like(f"%{str(summary_name_notlike)}%"))
        for m, s in query:
            long_name = self.build_summary_name(
                m.metric_name, m.metric_info_label, m.slicer_name, s.summary_name
            )
            vals = (
                m.metric_id,
                long_name,
                m.metric_name,
                m.slicer_name,
                m.metric_info_label,
                s.summary_name,
                s.summary_value,
            )
            if with_sim_name:
                vals += (m.run_name,)
            summarystats.append(vals)
        # Convert to numpy array.
        dtype_list = [
            ("metric_id", int),
            ("summary_name", str, self.slen),
            ("metric_name", str, self.slen),
            ("slicer_name", str, self.slen),
            ("metric_info_label", str, self.slen),
            ("summary_metric", str, self.slen),
            ("summary_value", float),
        ]
        if with_sim_name:
            dtype_list += [("run_name", str, self.slen)]
        dtype = np.dtype(dtype_list)
        summarystats = np.array(summarystats, dtype)
        self.close()
        return summarystats

    def get_plot_files(self, metric_id=None, with_sim_name=False):
        """
        Find the metric_id, name, info_label, and all plot info
        (optionally for metric_id list).

        Parameters
        ----------
        metric_id : `int`  `list`, or `None`
            If None, plots for all metrics are returned. Otherwise, only plots
            corresponding to the supplied metric ID or IDs are returned
        with_sim_name : `bool`
            If True, include the run name in the fields returned

        Returns
        -------
        plotFiles : `numpy.recarray`
            ``metric_id``
                The metric ID
            ``metric_name``
                The metric name
            ``metric_info_label``
                info_label extracted from the sql constraint
                (usually the filter)
            ``plot_type``
                The plot type
            ``plot_file``
                The full plot file (pdf by default)
            ``thumb_file``
                A plot thumbnail file name (png)
            ``run_name``
                The name of the run plotted (if `with_sim_name` was `True`)
        """
        if metric_id is None:
            metric_id = self.get_all_metric_ids()
        if not hasattr(metric_id, "__iter__"):
            metric_id = [
                metric_id,
            ]
        self.open()
        plotfiles = []
        for mid in metric_id:
            # Join the metric table and the plot table based on the metricID
            # (the second filter does the join)
            query = (
                self.session.query(MetricRow, PlotRow)
                .filter(MetricRow.metric_id == mid)
                .filter(MetricRow.metric_id == PlotRow.metric_id)
            )
            for m, p in query:
                # The plot_file typically ends with .pdf
                # (but the rest of name can have '.' or '_')
                thumb_file = "thumb." + ".".join(p.plot_file.split(".")[:-1]) + ".png"
                plot_file_fields = (
                    m.metric_id,
                    m.metric_name,
                    m.metric_info_label,
                    p.plot_type,
                    p.plot_file,
                    thumb_file,
                )
                if with_sim_name:
                    plot_file_fields += (m.run_name,)
                plotfiles.append(plot_file_fields)

        # Convert to numpy array.
        dtype_list = [
            ("metric_id", int),
            ("metric_name", str, self.slen),
            ("metric_info_label", str, self.slen),
            ("plot_type", str, self.slen),
            ("plot_file", str, self.slen),
            ("thumb_file", str, self.slen),
        ]

        if with_sim_name:
            dtype_list += [("run_name", str, self.slen)]
        dtype = np.dtype(dtype_list)

        plotfiles = np.array(plotfiles, dtype)
        self.close()
        return plotfiles

    def get_metric_data_files(self, metric_id=None):
        """
        Get the metric data filenames for all or a single metric.
        Returns a list.
        """
        if metric_id is None:
            metric_id = self.get_all_metric_ids()
        self.open()
        if not hasattr(metric_id, "__iter__"):
            metric_id = [
                metric_id,
            ]
        datafiles = []
        for mid in metric_id:
            for m in self.session.query(MetricRow).filter(MetricRow.metric_id == mid).all():
                datafiles.append(m.metric_datafile)
        self.close()
        return datafiles

    def get_metric_info(self, metric_id=None, with_sim_name=False):
        """Get the simple metric info, without display information.

        Parameters
        ----------
        metric_id : `int`  `list`, or `None`
            If None, data for all metrics are returned. Otherwise, only data
            corresponding to the supplied metric ID or IDs are returned
        with_sim_name : `bool`
            If True, include the run name in the fields returned

        Returns
        -------
        plotFiles : `numpy.recarray`
            ``metric_id``
                The metric ID
            ``metric_name``
                The metric name
            ``basemetric_names``
                The base metric names
            ``slicer_name``
                The name of the slicer used in the bundleGroup
            ``sql_constraint``
                The full sql constraint used in the bundleGroup
            ``metric_info_label``
                Metadata extracted from the `sql_constraint`
                (usually the filter)
            ``metric_datafile``
                The file name of the file with the metric data itself.
            ``run_name``
                The name of the run plotted (if `with_sim_name` was `True`)

        """
        if metric_id is None:
            metric_id = self.get_all_metric_ids()
        if not hasattr(metric_id, "__iter__"):
            metric_id = [
                metric_id,
            ]
        self.open()
        metricInfo = []
        for mId in metric_id:
            # Query for all rows in metrics and displays that match metric_ids
            query = self.session.query(MetricRow).filter(MetricRow.metric_id == mId)
            for m in query:
                base_metric_name = m.metric_name.split("_")[0]
                mInfo = (
                    m.metric_id,
                    m.metric_name,
                    base_metric_name,
                    m.slicer_name,
                    m.sql_constraint,
                    m.metric_info_label,
                    m.metric_datafile,
                )
                if with_sim_name:
                    mInfo += (m.run_name,)

                metricInfo.append(mInfo)
        # Convert to numpy array.
        dtype_list = [
            ("metric_id", int),
            ("metric_name", str, self.slen),
            ("base_metric_names", str, self.slen),
            ("slicer_name", str, self.slen),
            ("sql_constraint", str, self.slen),
            ("metric_info_label", str, self.slen),
            ("metric_datafile", str, self.slen),
        ]
        if with_sim_name:
            dtype_list += [("run_name", str, self.slen)]
        dtype = np.dtype(dtype_list)
        metricInfo = np.array(metricInfo, dtype)
        self.close()
        return metricInfo

    def get_metric_display_info(self, metric_id=None):
        """
        Get the contents of the metrics and displays table,
        together with the 'basemetric_name' (optionally, for metric_id list).
        Returns a numpy array of the metric information + display information.

        An underlying assumption here is that all metrics have some
        display info. This may not always be the case.
        """
        if metric_id is None:
            metric_id = self.get_all_metric_ids()
        if not hasattr(metric_id, "__iter__"):
            metric_id = [
                metric_id,
            ]
        self.open()
        metricInfo = []
        for mId in metric_id:
            # Query for all rows in metrics and displays that match metric_ids.
            query = (
                self.session.query(MetricRow, DisplayRow)
                .filter(MetricRow.metric_id == mId)
                .filter(MetricRow.metric_id == DisplayRow.metric_id)
            )
            for m, d in query:
                base_metric_name = m.metric_name.split("_")[0]
                mInfo = (
                    m.metric_id,
                    m.metric_name,
                    base_metric_name,
                    m.slicer_name,
                    m.sql_constraint,
                    m.metric_info_label,
                    m.metric_datafile,
                    d.display_group,
                    d.display_subgroup,
                    d.display_order,
                    d.display_caption,
                )
                metricInfo.append(mInfo)
        # Convert to numpy array.
        dtype = np.dtype(
            [
                ("metric_id", int),
                ("metric_name", np.str_, self.slen),
                ("base_metric_name", np.str_, self.slen),
                ("slicer_name", np.str_, self.slen),
                ("sql_constraint", np.str_, self.slen),
                ("metric_info_label", np.str_, self.slen),
                ("metric_datafile", np.str_, self.slen),
                ("display_group", np.str_, self.slen),
                ("display_subgroup", np.str_, self.slen),
                ("display_order", float),
                ("display_caption", np.str_, self.slen * 10),
            ]
        )
        metricInfo = np.array(metricInfo, dtype)
        self.close()
        return metricInfo

    def get_run_name(self):
        """Return a list of the run_names for the metric bundles in
        the database.
        """
        self.open()
        query = self.session.query(MetricRow.run_name.distinct()).all()
        run_name = []
        for s in query:
            run_name.append(s[0])
        self.close()
        return run_name
