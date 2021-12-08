import inspect
import rubin_sim.maf.metrics as metrics
import rubin_sim.maf.mafContrib as mafContrib

__all__ = ["makeMetricList"]


def makeMetricList(outfile):

    f = open(outfile, "w")

    # Print header
    print(".. py:currentmodule:: rubin_sim.maf", file=f)
    print("", file=f)
    print(".. _rubin_sim.maf_metricList:", file=f)
    print("", file=f)
    print("================================", file=f)
    print("rubin_sim MAF: Available metrics", file=f)
    print("================================", file=f)

    print("Core LSST MAF metrics", file=f)
    print("=====================", file=f)
    print(" ", file=f)
    for name, obj in inspect.getmembers(metrics):
        if inspect.isclass(obj):
            modname = inspect.getmodule(obj).__name__
            if modname.startswith("rubin_sim.maf.metrics"):
                link = f":py:class:`~rubin_sim.maf.metrics.{name}` "
                simpledoc = inspect.getdoc(obj).split("\n")[0]
                print(f"- {link} \n \t {simpledoc}", file=f)
    print(" ", file=f)

    print("Contributed mafContrib metrics", file=f)
    print("==============================", file=f)
    print(" ", file=f)
    for name, obj in inspect.getmembers(mafContrib):
        if inspect.isclass(obj):
            modname = inspect.getmodule(obj).__name__
            if modname.startswith("rubin_sim.maf.mafContrib") and name.endswith(
                "Metric"
            ):
                link = f":py:class:`~rubin_sim.maf.mafContrib.{name}` "
                simpledoc = inspect.getdoc(obj).split("\n")[0]
                print(f"- {link} \n \t {simpledoc}", file=f)
    print(" ", file=f)


if __name__ == "__main__":

    makeMetricList("rs_maf/metricList.rst")
