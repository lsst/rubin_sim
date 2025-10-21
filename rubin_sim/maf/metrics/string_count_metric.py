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

    Dynamically builds reduce functions for each unique string value,
    so summary stats can be named the same as strings in the
    simData array without knowing the values of those strings ahead of time.


    Parameters
    ----------
    metric_name : `str`, opt
        Name of the metric.
    col : `str`, opt
        Column name that has strings to look at.
    percent : `bool`, opt
        Normalize and return results as percents rather than raw count.
    clip_end : `bool`
        Clip if the end of a string if it ends with a comma and number.
    """

    def __init__(
        self, metric_name="stringCountMetric", col="filter", percent=False, clip_end=False, **kwargs
    ):
        if percent:
            units = "percent"
        else:
            units = "count"
        self.percent = percent
        cols = [col]
        super().__init__(cols, metric_name, units=units, metric_dtype=object, **kwargs)
        self.col = col
        self.clip_end = clip_end

    def run(self, data_slice, slice_point=None):

        # If we need to clip off trailing integer
        if self.clip_end:
            replace_col = []
            for val in data_slice[self.col]:
                if ", " in val:
                    chunks = val.split(", ")
                    if chunks[-1].isdigit():
                        new_val = ", ".join(chunks[0:-1])
                        replace_col.append(new_val)
                    else:
                        replace_col.append(val)
                else:
                    replace_col.append(val)
            data_slice[self.col] = replace_col

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
            # Not sure I really like having to loop here,
            # but the dtype is inflexible
            for key in metric_value.dtype.names:
                metric_value[key] = metric_value[key] / norm

        # Now to dynamically set up the reduce functions
        for i, key in enumerate(metric_value.dtype.names):
            name = key
            self.reduce_funcs[name] = Keylookerupper(key=key, name=name)
            self.reduce_order[name] = i

        return metric_value
