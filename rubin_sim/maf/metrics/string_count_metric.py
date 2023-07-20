__all__ = ("StringCountMetric",)

from collections import Counter

import numpy as np

from .base_metric import BaseMetric


class Keylookerupper:
    """Helper object to unpack dictionary values as reduceFunction results."""

    def __init__(self, key="blank", name=None):
        self.key = key
        self.__name__ = name

    def __call__(self, indict):
        return np.max(indict[self.key])


class StringCountMetric(BaseMetric):
    """Count up the number of times each string appears in a column.

    Dynamically builds reduce functions for each unique string value, so summary sats can be
    named the same as strings in the simData array without knowing the values of those trings ahead of time.
    """

    def __init__(self, metric_name="stringCountMetric", col="filter", percent=False, **kwargs):
        """
        Parameters
        ----------

        col: str ('filter')
            Column name that has strings to look at
        percent : bool (False)
            Normalize and return results as percents ranther than raw count
        """
        if percent:
            units = "percent"
        else:
            units = "count"
        self.percent = percent
        cols = [col]
        super(StringCountMetric, self).__init__(cols, metric_name, units=units, metric_dtype=object, **kwargs)
        self.col = col

    def run(self, data_slice, slice_point=None):
        counter = Counter(data_slice[self.col])
        # convert to a numpy array
        lables = list(counter.keys())
        # Numpy can't handle empty string as a dtype
        lables = [x if x != "" else "blank" for x in lables]
        metric_value = np.zeros(1, dtype=list(zip(lables, [float] * len(counter.keys()))))
        for key in counter:
            if key == "":
                metric_value["blank"] = counter[key]
            else:
                metric_value[key] = counter[key]
        if self.percent:
            norm = sum(metric_value[0]) / 100.0
            # Not sure I really like having to loop here, but the dtype is inflexible
            for key in metric_value.dtype.names:
                metric_value[key] = metric_value[key] / norm

        # Now to dynamically set up the reduce functions
        for i, key in enumerate(metric_value.dtype.names):
            name = key
            self.reduce_funcs[name] = Keylookerupper(key=key, name=name)
            self.reduce_order[name] = i

        return metric_value
