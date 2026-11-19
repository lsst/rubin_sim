from . import db as db
from . import metricBundles as metricBundles
from .metrics import LynxObsCountMetric
from .slicers import LynxResultsSlicer


def run_lightcurvelynx(sim_data, out_dir):
    slicer = LynxResultsSlicer()
    metric = LynxObsCountMetric(col="lightcurve", from_results=True)

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
