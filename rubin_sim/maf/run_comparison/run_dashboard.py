"""Dashboard for showing run metrics from the archive.
"""

__all__ = ["run_metric_dashboard"]

import panel as pn
import pandas as pd

from .archive import (
    get_metric_summaries,
    get_metric_sets,
    get_family_runs,
)

from .summary_plots import bokeh_plot_run_metric_mesh


def run_metric_dashboard(summaries=None, metric_sets=None, family_runs=None):
    """Create an interactive metric exploration dashboard.

    Parameters
    ----------
    summaries : `pandas.DataFrame`
        The summary metrics to normalize (as returned by `get_metric_summaries`)
    metric_sets : `pandas.DataFrame`
        Metric metadata as returned by `archive.get_metric_sets`
    family_sets : `pandas.DataFrame`
        Run family metadata as returned by `archive.get_family_runs`

    Returns
    -------
    app : `panel.layout.base.Panel`
    """

    all_summaries = get_metric_summaries() if summaries is None else summaries
    metric_sets = get_metric_sets() if metric_sets is None else metric_sets
    family_runs = get_family_runs() if family_runs is None else family_runs

    def make_run_metric_mesh(
        baseline_run, these_metric_sets, these_families, height, width
    ):
        metric_label_map = metric_sets.loc[these_metric_sets, "short_name"].droplevel(
            "metric set"
        )
        run_label_map = family_runs.loc[these_families, ["run", "brief"]].set_index(
            "run"
        )["brief"]

        # Some runs have "brief" descriptions too long to be useful labels, so use run names instead
        long_run_labels = run_label_map.str.len() > 40
        run_label_map[long_run_labels] = run_label_map.index[long_run_labels]

        summary = get_metric_summaries(
            these_families, these_metric_sets, summary_source=all_summaries
        )
        if baseline_run not in summary.index:
            summary = pd.concat(
                [
                    get_metric_summaries(
                        runs=[baseline_run],
                        metric_sets=these_metric_sets,
                        summary_source=all_summaries,
                    ),
                    summary,
                ]
            )

        p = bokeh_plot_run_metric_mesh(
            summary,
            baseline_run=baseline_run,
            metric_set=metric_sets.loc[these_metric_sets],
            run_label_map=run_label_map,
            metric_label_map=metric_label_map,
        )

        p.plot_height = height
        p.plot_width = width
        return p

    baseline_options = family_runs.loc["baseline"].run.unique().tolist()
    family_options = family_runs.index.unique().tolist()
    metric_set_options = (
        metric_sets.index.get_level_values("metric set").unique().tolist()
    )
    baseline_run = pn.widgets.Select(
        name="baseline",
        value=baseline_options[0],
        options=baseline_options,
    )
    these_metric_sets = pn.widgets.MultiSelect(
        name="metric sets", value=[metric_set_options[0]], options=metric_set_options
    )
    these_families = pn.widgets.MultiSelect(
        name="run families", value=[family_options[0]], options=family_options
    )

    height = pn.widgets.IntSlider(name="Figure height", start=128, end=4096, value=768)
    width = pn.widgets.IntSlider(name="Figure width", start=128, end=2048, value=768)

    app = pn.Column(
        baseline_run,
        these_families,
        these_metric_sets,
        pn.Row(height, width),
        pn.bind(
            make_run_metric_mesh,
            baseline_run,
            these_metric_sets,
            these_families,
            height,
            width,
        ),
    )
    return app


if __name__ == "__main__":
    app = run_metric_dashboard()
    app.show()
