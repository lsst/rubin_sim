__all__ = (
    "radar",
    "normalize_for_radar",
)

import matplotlib.pylab as plt
import numpy as np
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine

# Starting with example at
# https://matplotlib.org/examples/api/radar_chart.html


def normalize_for_radar(
    summary,
    norm_run="baseline",
    invert_cols=None,
    reverse_cols=None,
    mag_cols=[],
):
    """Normalize values in a dataframe to a given run,
    return output in a dataframe.

    This provides a similar functionality as the
    normalize_metric_summaries method, and returns a similar
    dataframe. The options for specifying which columns to invert,
    reverse, or identify as 'magnitudes'
    are slightly different, instead of using a 'metric_set'.

    Parameters
    ----------
    summary : `pandas.DataFrame`
        The data frame containing the metric summary stats to normalize
        (such as from `get_metric_summaries`).
        Note that this should contain only the runs and metrics to be
        normalized -- e.g.
        `summary.loc[[list of runs], [list of metrics]]`
        summary should be indexed by the run name.
    norm_run : `str`
        The name of the run to use to define the normalization.
    invert_cols : `list` [`str`]
        A list of column names that should be inverted (e.g., columns that
        are uncertainties and are better with a smaller value)
    reverse_cols : `list` [`str]
        Columns to reverse (e.g., magnitudes)
    mag_cols : `list` [`str`]
        Columns that are in magnitudes
    """
    out_df = summary.copy()
    if reverse_cols is not None:
        for colname in reverse_cols:
            out_df[colname] = -out_df[colname]
    if invert_cols is not None:
        for colname in invert_cols:
            out_df[colname] = 1.0 / out_df[colname]
    if norm_run is not None:
        indx = np.max(np.where(out_df.index == norm_run)[0])
        for col in out_df.columns:
            if col != "run_name":
                if (col in mag_cols) | (mag_cols == "all"):
                    out_df[col] = 1.0 + (out_df[col] - out_df[col].iloc[indx])
                else:
                    out_df[col] = 1.0 + (out_df[col] - out_df[col].iloc[indx]) / out_df[col].iloc[indx]
    return out_df


def _radar_factory(num_vars, frame="circle"):
    """Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : `int`
        Number of variables for radar chart.
    frame : {'circle' | 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)
    # rotate theta such that the first axis is at the top,
    # then make sure we don't go past 360
    theta += np.pi / 2
    theta = theta % (2.0 * np.pi)

    def draw_poly_patch(self):
        verts = _unit_poly_verts(theta)
        return plt.Polygon(verts, closed=True, edgecolor="k")

    def draw_circle_patch(self):
        # unit circle centered on (0.5, 0.5)
        return plt.Circle((0.5, 0.5), 0.5)

    patch_dict = {"polygon": draw_poly_patch, "circle": draw_circle_patch}
    if frame not in patch_dict:
        raise ValueError("unknown value for `frame`: %s" % frame)

    class RadarAxes(PolarAxes):
        name = "radar"
        # use 1 line segment to connect specified points
        RESOLUTION = 1
        # define draw_frame method
        draw_patch = patch_dict[frame]

        def fill(self, *args, **kwargs):
            """Override fill so that line is closed by default"""
            closed = kwargs.pop("closed", True)
            return super(RadarAxes, self).fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super(RadarAxes, self).plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            return self.draw_patch()

        def _gen_axes_spines(self):
            if frame == "circle":
                return PolarAxes._gen_axes_spines(self)
            # The following is a hack to get the spines (i.e. the axes frame)
            # to draw correctly for a polygon frame.

            # spine_type must be 'left', 'right', 'top', 'bottom', or `circle`.
            spine_type = "circle"
            verts = _unit_poly_verts(theta)
            # close off polygon by repeating first vertex
            verts.append(verts[0])
            path = Path(verts)

            spine = Spine(self, spine_type, path)
            spine.set_transform(self.transAxes)
            return {"polar": spine}

    register_projection(RadarAxes)
    return theta


def _unit_poly_verts(theta):
    """Return vertices of polygon for subplot axes.

    This polygon is circumscribed by a unit circle centered at (0.5, 0.5)
    """
    x0, y0, r = [0.5] * 3
    verts = [(r * np.cos(t) + x0, r * np.sin(t) + y0) for t in theta]
    return verts


def radar(
    df,
    rgrids=[0.7, 1.0, 1.3, 1.6],
    colors=None,
    alpha=0.1,
    legend=True,
    figsize=(8.5, 5),
    fill=False,
    bbox_to_anchor=(1.6, 0.5),
    legend_font_size=None,
):
    """make a radar plot!"""
    theta = _radar_factory(np.size(df.columns), frame="polygon")
    fig, axes = plt.subplots(figsize=figsize, subplot_kw=dict(projection="radar"))
    axes.set_rgrids(rgrids, fontsize="x-large")

    if colors is None:
        colors = [None for i in range(len(df))]
    ix = 0
    for i, row in df.iterrows():
        axes.plot(theta, row.values, "o-", label=i, color=colors[ix])
        if fill:
            axes.fill(theta, row.values, alpha=alpha)
        ix += 1

    variables = df.columns.values

    axes.set_varlabels(variables)
    if legend:
        axes.legend(
            bbox_to_anchor=bbox_to_anchor,
            borderaxespad=0,
            loc="lower right",
            fontsize=legend_font_size,
        )
    axes.set_ylim([np.min(rgrids), np.max(rgrids)])

    return fig, axes
