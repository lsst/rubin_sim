__all__ = ("make_metric_list",)

import inspect

import rubin_sim.maf.maf_contrib as maf_contrib
import rubin_sim.maf.metrics as metrics


def make_metric_list(outfile):
    f = open(outfile, "w")

    # Print header
    print(".. py:currentmodule:: rubin_sim.maf", file=f)
    print("", file=f)
    print(".. _maf-metric-list:", file=f)
    print("", file=f)
    print("################################", file=f)
    print("rubin_sim MAF: Available metrics", file=f)
    print("################################", file=f)

    print(" ", file=f)

    print("Core LSST MAF metrics", file=f)
    print("^^^^^^^^^^^^^^^^^^^^^", file=f)
    print(" ", file=f)
    for name, obj in inspect.getmembers(metrics):
        if inspect.isclass(obj):
            modname = inspect.getmodule(obj).__name__
            if modname.startswith("rubin_sim.maf.metrics"):
                link = f":py:class:`~rubin_sim.maf.metrics.{name}` "
                simpledoc = inspect.getdoc(obj).split("\n")[0]
                print(f"- {link} \n \t {simpledoc}", file=f)
    print(" ", file=f)

    print("Contributed maf_contrib metrics", file=f)
    print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^", file=f)
    print(" ", file=f)
    for name, obj in inspect.getmembers(maf_contrib):
        if inspect.isclass(obj):
            modname = inspect.getmodule(obj).__name__
            if modname.startswith("rubin_sim.maf.maf_contrib") and name.endswith("Metric"):
                link = f":py:class:`~rubin_sim.maf.maf_contrib.{name}` "
                simpledoc = inspect.getdoc(obj).split("\n")[0]
                print(f"- {link} \n \t {simpledoc}", file=f)
    print(" ", file=f)


if __name__ == "__main__":
    make_metric_list("maf-metric-list.rst")
