__all__ = ("FigSaver",)

import os
import sqlite3
from pathlib import Path

import matplotlib.pylab as plt
import pandas as pd

from .schemas import empty_info


class FigSaver:
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

    def _construct_fileroot(self, info):
        """Construct a reasonable filename from the info dict

        Parameters
        ----------
        info : `dict`
            Dictionary to use when constructing filename.
        """
        filename = ""

        for key in ["metric: name", "metric: col", "observations_subset"]:
            if key in info.keys():
                filename += info[key] + "_"

        if "slicer: nside" in info.keys():
            filename += "nside%i" % info["slicer: nside"]

        # Maybe a more extensive clean here
        swaps = {"=": "_", " ": "_", "<": "lt", ">": "gt"}
        for key in swaps:
            filename = filename.replace(key, swaps[key])

        while "__" in filename:
            filename = filename.replace("__", "_")

        while filename[-1] == "_":
            filename = filename[0:-1]

        # Could throw a warning here, or even an error
        if filename == "":
            raise ValueError("Unable to generate output filename from info dict")

        return filename

    def save_stats(self, stats):
        """Save summary statistics to the output DB."""
        df = pd.DataFrame(stats)
        df.to_sql("stats", self.conn, index=False, if_exists="append")

    def __call__(self, fig, info, filename=None):
        """Save a figure

        Parameters
        ----------
        fig : `matplotlib.Figure`
            The figure object to save.
        info : `dict`
            Dict with information about the figure.
            Used to generate filename and fill info in tracking
            database.
        filename : `str`
            Base filename for the output. Default of None will
            result in auto-generated filename.
        """

        row = empty_info(as_df_row=True)

        for key in row.columns:
            if key in info.keys():
                row[key] = info[key]

        if filename is None:
            filename = self._construct_fileroot(info)

        if self.pdf_dpi is not None:
            pdf_filename = filename + ".pdf"
            output_file = os.path.join(self.outdir, pdf_filename)
            fig.savefig(output_file, dpi=self.pdf_dpi, bbox_inches=self.bbox_inches)
            row["filename"] = pdf_filename
            row.to_sql("plots", self.conn, index=False, if_exists="append")

        if self.png_dpi is not None:
            png_filename = "thumb_" + filename + ".png"
            output_filename = os.path.join(self.outdir, png_filename)
            fig.savefig(output_filename, dpi=self.png_dpi, bbox_inches=self.bbox_inches)
            row["filename"] = png_filename
            row.to_sql("plots", self.conn, index=False, if_exists="append")

        if self.close_figs:
            plt.close(fig)
