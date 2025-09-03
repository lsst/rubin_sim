__all__ = ("metric_parallel",)

from itertools import repeat
from multiprocessing import Manager, Pool, set_start_method

import numpy as np
from rubin_scheduler.utils import SharedNumpyArray


def call_single_indx(shared_data, shared_metric, shared_slicer, indx):
    """Call a slicer and return a single value"""

    result = shared_slicer(shared_data.read(), shared_metric, indx=[indx], skip_setup=True)
    return result


def launch_jobs(shared_data, metric, slicer, processes=6):
    """Launch the slicer calls in parallel."""
    with Manager() as manager:
        manager.shared_metric = metric
        manager.shared_slicer = slicer
        # make the args iterable
        args = zip(
            repeat(shared_data),
            repeat(manager.shared_metric),
            repeat(manager.shared_slicer),
            range(len(slicer)),
        )
        with Pool(processes=processes) as pool:
            result = pool.starmap(call_single_indx, args)
    result = np.concatenate(result)
    return result


def metric_parallel(visits, metric, slicer, info=None, processes=6, set_multi_method=True):
    """Run a metric and slicer in parallel.

    Parameters
    ----------
    visits : `np.array`
        The numpy array with visit info
    metric : `rubin_sim.maf_proto.metric`
        The metric to run
    slicer : `rubin_sim.maf_proto.slicer`
        The slicer object
    info : `dict`
        An info dict for tracking how metric was run.
        Default None
    processes : `int`
        Number of processes to launch. Default 6.
    set_multi_method : `bool`
        Use multiprocessing.set_start_method to ensure
        things work on mac OS. Default True.
    """

    # This seems to be required for things to
    # work on mac OS.
    if set_multi_method:
        set_start_method("fork", force=True)

    shared_array = SharedNumpyArray(visits)
    slicer.setup_slicer(visits)

    result = launch_jobs(shared_array, metric, slicer, processes=processes)

    shared_array.unlink()
    if info is not None:
        info = slicer.add_info(metric, info)
        return result, info
    else:
        return result
