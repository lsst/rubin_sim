import inspect
import rubin_sim.maf.metrics as metrics
import rubin_sim.maf.mafContrib as mafContrib

__all__ = ['makeMetricList']

def makeMetricList(outfile):

    f = open(outfile, 'w')

    print("=================", file=f)
    print("Available metrics", file=f)
    print("=================", file=f)


    print("Core LSST MAF metrics", file=f)
    print("=====================", file=f)
    print(" ", file=f)
    for name, obj in inspect.getmembers(metrics):
        if inspect.isclass(obj):
            modname = inspect.getmodule(obj).__name__
            if modname.startswith('rubin_sim.maf.metrics'):
                link = "source/rubin_sim.maf.metrics.html#%s.%s" % (modname, obj.__name__)
                simpledoc = inspect.getdoc(obj).split('\n')[0]
                print("- `%s <%s>`_ \n \t %s" % (name, link, simpledoc), file=f)
    print(" ", file=f)

    print("Contributed mafContrib metrics", file=f)
    print("==============================", file=f)
    print(" ", file=f)
    for name, obj in inspect.getmembers(mafContrib):
        if inspect.isclass(obj):
            modname = inspect.getmodule(obj).__name__
            if modname.startswith('rubin_sim.maf.mafContrib')  and name.endswith('Metric'):
                link = "source/rubin_sim.maf.mafContrib.html#%s.%s" % (modname, obj.__name__)
                simpledoc = inspect.getdoc(obj).split('\n')[0]
                print("- `%s <%s>`_ \n \t %s" % (name, link, simpledoc), file=f)
    print(" ", file=f)


if __name__ == '__main__':

    makeMetricList('metricList.rst')
