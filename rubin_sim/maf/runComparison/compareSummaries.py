import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display_markdown

__all__ = ['SummaryComparison']


class SummaryComparison():
    """
    This class provides some utility methods to handle the CSV file which can be created with RunComparison.
    RunComparison creates a file with all of the summary metrics from multiple runs, pulled from the
    original resultsDb files -- after saving this pandas dataframe, we can read it back in here.

    This class provides some utilities to more easily pull out particular columns (metrics) from
    the dataframe, pull out subsets of runs (families), normalize subsets of the metrics/runs appropriately,
    and print some potentially useful family run information to screen (such as in a jupyter notebook).

    See the JSON file describing the current set of available runs for more information on families.
    A CSV file containing the current set of summary statistics outputs for these runs should be
    matched with another JSON file containing information on the (likely) most-relevant metrics.
    These files can be downloaded at XXX.
    """
    def __init__(self, summary_file, metric_file, run_file):
        self.read_summary_stats(summary_file)
        self.read_metric_info(metric_file)
        self.read_run_info(run_file)

    def read_summary_stats(self, filename):
        """This is expecting a summary CSV file similar to that created by running RunComparison to pull
        summary statistics out of the resultsDb files.

        Parameters
        ----------
        filename : `str`
            The filename for the CSV file.
        """
        # Read in the CSV file containing the summary stat information (
        self.summary_stats = pd.read_csv(filename, index_col=0)

    def read_metric_info(self, metric_file):
        """Read the file containing shortcuts to current metric information.

        This includes: sets of metrics which together form a family, short names for these metrics,
        and whether particular metrics deal with magnitudes or must be inverted.
        """
        self.metric_info = None

        # This should come from the metric_file instead
        self.tablemetrics = ['fOArea fO All visits HealpixSlicer',
                        'Effective Area (deg) ExgalM5_with_cuts i band non-DD year 10 HealpixSlicer',
                        'Nvisits All',
                        'fONv MedianNvis fO All visits HealpixSlicer',
                        'Median NVisits u band HealpixSlicer',
                        'Median NVisits g band HealpixSlicer',
                        'Median NVisits r band HealpixSlicer',
                        'Median NVisits i band HealpixSlicer',
                        'Median NVisits z band HealpixSlicer',
                        'Median NVisits y band HealpixSlicer', ]
        self.tablenames = ['Area with >825 visits/pointing (fO_Area)',
                      'Unextincted area i>25.9',
                      'Nvisits total',
                      'Median Nvis over top 18k (fO_Nv Med)',
                      'Median Nvis u band',
                      'Median Nvis g band',
                      'Median Nvis r band',
                      'Median Nvis i band',
                      'Median Nvis z band',
                      'Median Nvis y band']
        # useful bits of metric info - fill in from JSON file
        self.metrics = {}
        self.metric_short_names = {}
        self.metric_short_names_norm = {}
        self.metric_invert_cols = {}
        self.metric_mag_cols = {}
        self.metric_plot_styles = {}

    def read_run_info(self, run_file):
        """"Read the file containing information about current recommended simulations,
        including how they are grouped into families.

        This includes: simulations which are currently available, which families they are grouped
        into, which run is considered the most-relevant baseline for that family, short descriptions
        for the overall family and individual runs in the family, and where each simulation is
        available for download.
        """
        self.run_info = None
        # The simulations included in the survey strategy grouping
        self.run_families = {}
        # An overall comment about the grouping
        self.run_comments = {}
        # Potentially useful nicknames or overwhelmingly brief descriptors -- USE THESE SPARINGLY
        # By using the nickname instead of the full name, you lose traceability for which run is really which.
        self.run_nicknames = {}
        # What is the most-useful comparison run for this grouping
        self.run_family_baseline = {}
        # The release number for this grouping
        self.run_family_version = {}

    def list_of_families(self):
        """Print a pretty list of the simulation groups under consideration, as of this time. """
        # The families
        displaystring = ''
        family_list = []
        simlist = []
        for k in self.run_families:
            if k == 'version_baselines':
                continue
            family_list.append(k)
            displaystring+= f"**{k}**, with {len(self.run_families[k])} simulations.<br>"
            simlist += self.run_families[k]
        display_markdown(displaystring, raw=True)
        simlist = set(simlist)
        print(f'For {len(simlist)} unique simulations in all.')
        return family_list

    def family_info(self, f, normalized=False):
        """Print some summary information about the family and return a high-level set of metrics."""
        d = pd.DataFrame(self.summary_stats[self.tablemetrics].loc[self.run_families[f]])
        if normalized:
            d = d/self.summary_stats[self.tablemetrics].loc[self.run_family_baseline[f]]
        d.columns = self.tablenames
        d['Briefly'] = self.run_nicknames[f]
        display_markdown(self.run_comments[f], raw=True)
        print(f"Comparison run: {self.run_family_baseline[f]}")
        return d

    def plot_areaNvis(self, f):
        """Make a commonly-used plot showing high-level metrics only about number of visits and survey area
        """
        metrics = self.tablemetrics[0:4]
        names = self.tablenames[0:4]
        d = self.summary_stats[metrics].loc[self.run_families[f]]
        nd = self.norm_df(d, self.run_family_baseline[f])
        nd.columns = names
        self.plot(nd, normed=True, figsize=(10, 6), style=['k-', 'k:', 'r-', 'r:'])
        plt.xlim(0, len(nd)-1)
        xlims = plt.xlim()
        if plt.ylim()[0] < 0.50:
            plt.ylim(bottom=0.50)
        if plt.ylim()[1] > 1.5:
            plt.ylim(top=1.5)
        plt.fill_between(xlims, 1.05, plt.ylim()[1], color='g', alpha=0.1)
        plt.fill_between(xlims, 0.95, plt.ylim()[0], color='r', alpha=0.1)

    @staticmethod
    def norm_df(df, norm_run, invert_cols=None, reverse_cols=None, mag_cols=None):
        """
        Normalize values in a DataFrame, based on the values in a given run.
        Can normalize some columns (metric values) differently (invert_cols, reverse_cols, mag_cols)
        if those columns are specified; this lets the final normalized dataframe 'look' the same way
        in a plot (i.e. "up" is better (reverse_cols), they center on 1 (mag_cols), and the magnitude scales
        as expected (invert_cols)).

        The names to use for invert_cols, reverse_cols, mag_cols, etc. are specified by the user, but
        may be obtained from self.metric_info.

        Parameters
        ----------
        df : `pd.DataFrame`
            The data frame containing the metric values to compare
        norm_run: `str`
            The name of the simulation to normalize to (typically family_baseline)
        invert_cols: `list`, opt
            Columns (metric values) to convert to 1 / value
        reverse_cols: `list`, opt
            Columns (metric values) to invert (-1 * value)
        mag_cols: `list`, opt
            Columns (metrics values) to treat as magnitudes (1 + (difference from norm_run))

        Returns
        -------
        out_df : `pd.DataFrame`
            Normalized data frame
        """
        # Copy the dataframe but drop the columns containing only strings
        out_df = df.copy()
        if reverse_cols is not None:
            out_df[reverse_cols] = -out_df[reverse_cols]
        if invert_cols is not None:
            out_df[invert_cols] = 1 / out_df[invert_cols]
        if mag_cols is not None:
            out_df[mag_cols] = 1 + out_df[mag_cols] - out_df[mag_cols].loc[norm_run]
        else:
            mag_cols = []
        # which columns are strings?
        string_cols = [c for c, t in zip(df.columns, df.dtypes) if t == 'object']
        cols = [c for c in out_df.columns.values if not (c in mag_cols or c in string_cols)]
        out_df[cols] = 1 + (out_df[cols] - out_df[cols].loc[norm_run]) / out_df[cols].loc[norm_run]
        return out_df

    @staticmethod
    def plot(df, normed=True, style=None, figsize=(10, 6), run_nicknames=None):
        """Plot a DataFrame of metric values.

        Parameters
        ---------
        df: `pd.DataFrame`
            The dataframe of metric values to plot
        normed: `bool`, opt
            Is the dataframe normalized or not? (default True)
            If true, adds +/- 5% lines to output
        style: `list`, opt
            Optional list of line color/style values to use for the plotted metric values
        figsize: `tuple`, opt
            Figure size
        run_nicknames: `list`, opt
            Replace the run names in the dataframe with these nicknames
        """
        df.plot(figsize=figsize, style=style)
        plt.legend(loc=(1.01, 0))
        if normed:
            plt.axhline(0.95, alpha=0.3, linestyle=':')
            plt.axhline(1.0, alpha=0.3, linestyle='--')
            plt.axhline(1.05, alpha=0.3, linestyle=':')
        if run_nicknames is not None:
            xnames = run_nicknames
        else:
            xnames = df.index.values
        xi = np.arange(len(xnames))
        plt.xticks(xi, xnames, rotation=90, fontsize='large')
        plt.xlim(0, len(xnames) - 1)
        plt.grid('k:', alpha=0.3)
        plt.tight_layout()

    @staticmethod
    def fO_cutoff(df, norm_run):
        """Calculate the Y value for a cutoff line where the fO metric fails SRD req.
        This location changes on the plot according to the scaling from the baseline for the family,
        so is useful to mark on plots of SRD quantities.
        """
        srd_fO_cutoff = df['fONv MedianNvis fO All visits HealpixSlicer'].loc[norm_run]
        srd_fO_cutoff = 825 / srd_fO_cutoff
        return srd_fO_cutoff