from . import db as db
from . import metricBundles as metricBundles
from . import slicers
from .metrics import LCLObsCountMetric


def run_lightcurvelynx(sim_data):
    out_dir = "/Users/maxwest/maf_runs/"
    slicer = slicers.LynxBaseSlicer()
    metric = LCLObsCountMetric(col="lightcurve")

    bundle = metricBundles.MetricBundle(metric, slicer, constraint="")

    results_db = db.ResultsDb(out_dir=out_dir)

    group = metricBundles.MetricBundleGroup(
        [bundle],
        sim_data,
        out_dir=out_dir,
        results_db=results_db,
        db_table="lightcurves",
    )

    group.run_all()
