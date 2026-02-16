__all__ = ("UseMetric",)

from rubin_sim.maf.metrics.base_metric import BaseMetric


class UseMetric(BaseMetric):  # pylint: disable=too-few-public-methods
    """Metric to classify visits by type of visits"""

    start_matches = [
        "singles",
        "pairs",
        "triplet",
        "template",
        "too",
        "twilight",
        "ddf_cosmos",
        "ddf_ecdfs",
        "ddf_edfs",
        "ddf_elaiss1",
        "ddf_xmm_lss",
    ]

    def __init__(
        self, note_col="observation_reason", prog_col="science_program", science_programs=tuple(), **kwargs
    ):
        self.note_col = note_col
        self.prog_col = prog_col
        self.science_programs = list(science_programs)
        super().__init__(col=[note_col, prog_col], metric_dtype="object", **kwargs)

    def run(self, data_slice, slice_point=None):  # pylint: disable=invalid-name
        """Run the metric.

        Parameters
        ----------
        data_slice : `np.ndarray`, (N,)`
        slice_point : `dict`
           Dictionary of slice_point metadata passed to each metric.
           E.g. the ra/dec of the healpix pixel.

        Returns
        -------
        use_name : `str`
            use at each slice_point.
        """
        if len(self.science_programs) > 0 and data_slice[self.prog_col] not in self.science_programs:
            return "not science"

        for start_match in self.start_matches:
            if data_slice[self.note_col].startswith(start_match):
                return start_match

        return "other"


# internal functions & classes
