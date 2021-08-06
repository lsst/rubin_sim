# Plot the summary statistics (values stored in this class).
def plotSummaryStats(self, output=None, totalVisits=True):
    """
    Plot the normalized metric values as a function of opsim run.

    output: str, optional
        Name of figure to save to disk. If this is left as None the figure
        is not saved.
    totalVisits: bool
        If True the total number of visits is included in the metrics plotted.
        When comparing runs a very different lengths it is recommended to
        set this flag to False.
    """
    ylabel = '(run - ' + self.baselineRun + ')/' + self.baselineRun
    dataframe = self.noramlizeStatsdf
    magcols = [col for col in dataframe.columns if 'M5' in col]
    HAcols = [col for col in dataframe.columns if 'HA' in col]
    propMocols = [col for col in dataframe.columns if 'Prop. Mo.' in col]
    seeingcols = [col for col in dataframe.columns if 'seeing' in col]
    parallaxCols = [col for col in dataframe.columns if 'Parallax' in col]
    if totalVisits is True:
        othercols = ['Mean Slew Time', 'Median Slew Time', 'Median NVists Per Night',
                     'Median Open Shutter Fraction', 'Nights with Observations',
                     'Total Eff Time', 'Total Visits']
    else:
        othercols = ['Mean Slew Time', 'Median Slew Time',
                     'Median NVists Per Night',
                     'Median Open Shutter Fraction']
    colsets = [othercols, magcols, HAcols, propMocols, parallaxCols, seeingcols]
    fig, axs = plt.subplots(len(colsets), 1, figsize=(8, 33))
    fig.subplots_adjust(hspace=.4)
    axs = axs.ravel()
    for i, c in enumerate(colsets):
        x = np.arange(len(dataframe))
        for metric in dataframe[c].columns:
            axs[i].plot(x, dataframe[metric], marker='.', ms=10, label=metric)
        axs[i].grid(True)
        axs[i].set_ylabel(ylabel)
        lgd = axs[i].legend(loc=(1.02, 0.2), ncol=1)
        plt.setp(axs[i].xaxis.get_majorticklabels(), rotation=90)
        plt.setp(axs[i], xticks=x, xticklabels=[x.strip('') for x in dataframe.index.values])
    if output:
        plt.savefig(output, bbox_extra_artists=(lgd,), bbox_inches='tight')
