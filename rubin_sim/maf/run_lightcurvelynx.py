from . import db as db
from . import metricBundles as metricBundles
from . import slicers
from .metrics import LCLObsCountMetric


def run_lightcurvelynx(sim_data="/Users/maxwest/rubin_sim_data/lcl_data/lightcurves.db"):
    out_dir = "/Users/maxwest/maf_runs/"
    # data = read_parquet(sim_data)
    slicer = slicers.LynxSpatialSlicer()
    slicer.setup_slicer(sim_data)
    # slicer.sim_data = data
    metric = LCLObsCountMetric()

    # constraint = "select * from lightcurves"

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
