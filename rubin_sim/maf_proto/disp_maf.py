__all__ = (
    "RunSelectHandler",
    "ConfigPageHandler",
    "StatPageHandler",
    "AllMetricResultsPageHandler",
    "MultiColorPageHandler",
    "make_app",
    "disp_maf",
)

import argparse
import os
import webbrowser

from jinja2 import Environment, FileSystemLoader
from tornado import ioloop, web

import rubin_sim
from rubin_sim.maf_proto.db import add_run_to_database
from rubin_sim.maf_proto.web import MafTracking


class RunSelectHandler(web.RequestHandler):
    def get(self):
        selectTempl = env.get_template("runselect.html")
        if "runId" in self.request.arguments:
            runId = int(self.request.arguments["runId"][0])
        else:
            # Set runID to a negative number, to default to first run.
            runId = startRunId
        self.write(selectTempl.render(runlist=runlist, runId=runId, jsPath=jsPath))


class ConfigPageHandler(web.RequestHandler):
    def get(self):
        configTempl = env.get_template("configs.html")
        runId = int(self.request.arguments["runId"][0])
        self.write(configTempl.render(runlist=runlist, runId=runId))


class StatPageHandler(web.RequestHandler):
    def get(self):
        statTempl = env.get_template("stats.html")
        runId = int(self.request.arguments["runId"][0])
        self.write(statTempl.render(runlist=runlist, runId=runId))


class AllMetricResultsPageHandler(web.RequestHandler):
    def get(self):
        """Load up the files and display"""
        allresultsTempl = env.get_template("allmetricresults.html")
        runId = int(self.request.arguments["runId"][0])
        self.write(allresultsTempl.render(runlist=runlist, runId=runId))


class MultiColorPageHandler(web.RequestHandler):
    def get(self):
        """Display sky maps."""
        multiColorTempl = env.get_template("multicolor.html")
        runId = int(self.request.arguments["runId"][0])
        self.write(multiColorTempl.render(runlist=runlist, runId=runId))


def make_app():
    """The tornado global configuration"""
    application = web.Application(
        [
            ("/", RunSelectHandler),
            ("/configParams", ConfigPageHandler),
            ("/summaryStats", StatPageHandler),
            ("/allMetricResults", AllMetricResultsPageHandler),
            ("/multiColor", MultiColorPageHandler),
            (r"/(favicon.ico)", web.StaticFileHandler, {"path": faviconPath}),
            (r"/(sorttable.js)", web.StaticFileHandler, {"path": jsPath}),
            (r"/*/(.*)", web.StaticFileHandler, {"path": staticpath}),
        ]
    )
    return application


def disp_maf():
    """Display MAF output in a web browser. After launching, point your browser
    to 'http://localhost:8888/'.
    """

    parser = argparse.ArgumentParser(
        description="Python script to display MAF output in a web browser."
        + "  After launching, point your browser to 'http://localhost:8888/'"
    )
    defaultdb = os.path.join(os.getcwd(), "disp_maf.sqlite")
    parser.add_argument(
        "-t",
        "--tracking_db",
        type=str,
        default=defaultdb,
        help="Tracking database filename.",
    )
    parser.add_argument(
        "-d",
        "--maf_dir",
        type=str,
        default=None,
        help="Add this directory to the tracking_db and open immediately.",
    )
    parser.add_argument(
        "-c",
        "--maf_comment",
        type=str,
        default=None,
        help="Add a comment to the trackingDB describing the "
        + " MAF analysis of this directory (paired with maf_dir argument).",
    )
    parser.add_argument("-p", "--port", type=int, default=8888, help="Port for connecting to showMaf.")
    parser.add_argument(
        "--no_browser",
        dest="no_browser",
        default=False,
        action="store_true",
        help="Do not open a new browser tab",
    )

    args = parser.parse_args()

    # Check tracking DB is sqlite (and add as convenience if forgotten).
    tracking_db = args.tracking_db
    print("Using tracking database at %s" % (tracking_db))

    global startRunId
    startRunId = -666
    # If given a directory argument:
    if args.maf_dir is not None:
        maf_dir = os.path.realpath(args.maf_dir)
        if not os.path.isdir(maf_dir):
            print("There is no directory containing MAF outputs at %s." % (maf_dir))
            print("Just opening using tracking db at %s." % (tracking_db))
        else:
            add_run_to_database(maf_dir, tracking_db, maf_comment=args.maf_comment)

    # Open tracking database and start visualization.
    global runlist
    runlist = MafTracking(tracking_db)
    if startRunId < 0:
        startRunId = runlist.runs[0]["maf_run_id"]
    # Set up path to template and favicon paths, and load templates.
    maf_dir = os.path.join(rubin_sim.__path__[0], "maf_proto")
    templateDir = os.path.join(maf_dir, "web/templates/")
    global faviconPath
    faviconPath = os.path.join(maf_dir, "web/")
    global jsPath
    jsPath = os.path.join(maf_dir, "web/")
    global env
    env = Environment(loader=FileSystemLoader(templateDir))
    # Add 'zip' to jinja templates.
    env.globals.update(zip=zip)

    global staticpath
    staticpath = "."
    # Start up tornado app.
    application = make_app()
    application.listen(args.port)
    print("Tornado Starting: \nPoint your web browser to http://localhost:%d/ \nCtrl-C to stop" % (args.port))
    if not args.no_browser:
        webbrowser.open_new_tab("http://localhost:%d" % (args.port))
    ioloop.IOLoop.instance().start()
