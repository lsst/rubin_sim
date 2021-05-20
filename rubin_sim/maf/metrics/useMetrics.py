# imports
import numpy as np

from rubin_sim.maf.metrics.baseMetric import BaseMetric

# constants

__all__ = ["UseMetric"]

# exception classes

# interface functions

# classes


class UseMetric(BaseMetric):  # pylint: disable=too-few-public-methods
    """Metric to classify visits by type of visits"""

    def __init__(self, noteCol="note", **kwargs):
        self.noteCol = noteCol
        super().__init__(col=[noteCol], metricDtype="object", **kwargs)

    def run(self, dataSlice, slicePoint=None):  # pylint: disable=invalid-name
        """Run the metric.

        Parameters
        ----------
        dataSlice : numpy.NDarray
           Values passed to metric by the slicer, which the metric will use to calculate
           metric values at each slicePoint.
        slicePoint : Dict
           Dictionary of slicePoint metadata passed to each metric.
           E.g. the ra/dec of the healpix pixel or opsim fieldId.

        Returns
        -------
        str
            use at each slicePoint.
        """
        use_name = None
        visible_bands = ("u", "g", "r")
        notes = dataSlice[self.noteCol]
        if len(notes.shape) == 0:
            note = notes
        else:
            note = notes[0]
            assert np.all(notes == note)

        note_elems = note.replace(":", ", ").split(", ")
        if note_elems[0] == "greedy":
            use_name = note_elems[0]
        if note_elems[0] == "DD":
            use_name = note_elems[1]
        if note_elems[0] == "blob":
            use_name = "wide with only IR"
            for band in visible_bands:
                if band in note_elems[1]:
                    use_name = "wide with u, g, or r"

        assert use_name is not None, f"Unrecognized note: {note}"
        return use_name


# internal functions & classes
