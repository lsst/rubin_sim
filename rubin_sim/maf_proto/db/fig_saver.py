__all__ = ("FigSaver",)

import copy
import os
import sqlite3
from pathlib import Path

import matplotlib.pylab as plt
import pandas as pd


class FigSaver(object):
    """Class to save figures and store info about them in a database.
    Uses info dictionary to construct reasonable filenames.

    Parameters
    ----------
    tracking_file : `str`
        Path to tracking database. If directory does not exist,
        it will be created.
    png_dpi : `int`
        DPI to use for pngs. Default 72 (top for web pages).
        Set to None to skip png generation.
    pdf_dpi : `int`
        DPI to use for pdf generation. Default 600.
        Set to None to skip pdf generation
    close_figs : `bool`
        Set to True to close figure after saving. Default True.
    bbox_inches : `str`
        Passed to matplotlib.Figure.savefig. Default "tight".
    """

    def __init__(
        self,
        tracking_file="maf_figs/maf_tracking.db",
        png_dpi=72,
        close_figs=True,
        pdf_dpi=600,
        bbox_inches="tight",
    ):
        self.outdir = os.path.dirname(tracking_file)
        Path(self.outdir).mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(tracking_file)
        self.close_figs = close_figs
        self.pdf_dpi = pdf_dpi
        self.png_dpi = png_dpi
        self.bbox_inches = bbox_inches

        # This eventually goes somewhere else so can be documented
        schema_dict = {
            "metric: name": "",
            "metric: col": "",
            "slicer: nside": 0,
            "observations_subset": "",
            "caption": "",
        }
        self.schema_row = pd.Series(schema_dict).to_frame().T

    def _construct_fileroot(self, info):
        filename = ""

        for key in ["metric: name", "metric: col", "observations_subset"]:
            if key in info.keys():
                filename += info[key] + "_"

        if "slicer: nside" in info.keys():
            filename += "nside%i_" % info["slicer: nside"]

        # Maybe a more extensive clean here
        filename = filename.replace("=", "_").replace(" ", "_")

        if filename == "":
            filename = "default_filename"

        self.filename = filename

    def __call__(self, fig, info):

        row = copy.copy(self.schema_row)
        for key in info:
            row[key] = info[key]

        self._construct_fileroot(info)

        if self.pdf_dpi is not None:
            filename = os.path.join(self.outdir, self.filename + ".pdf")
            fig.savefig(filename, dpi=self.pdf_dpi, bbox_inches=self.bbox_inches)
            row["filename"] = filename
            row.to_sql("plots", self.conn, index=False, if_exists="append")

        if self.png_dpi is not None:
            filename = os.path.join(self.outdir, "thumb_" + self.filename + ".png")
            fig.savefig(filename, dpi=self.png_dpi, bbox_inches="tight")
            row["filename"] = filename
            row.to_sql("plots", self.conn, index=False, if_exists="append")

        if self.close_figs:
            plt.close(fig)
