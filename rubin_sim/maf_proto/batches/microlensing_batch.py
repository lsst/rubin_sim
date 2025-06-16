__all__ = ("microlensing_batch",)

import numpy as np

import rubin_sim.maf_proto as maf


class DoNothing:
    def __call__(*args, **kwargs):
        pass


def microlensing_batch(observations=None, run_name=None, quick_test=False, fig_saver=None):
    visits_array, df, run_name, subset, fig_saver = maf.batch_preamble(
        observations=observations,
        run_name=run_name,
        quick_test=quick_test,
        fig_saver=fig_saver,
    )

    mjd0 = visits_array["observationStartMJD"].min()

    summary_stats = []

    crossing_times = [
        [1, 5],
        [5, 10],
        [10, 20],
        [20, 30],
        [30, 60],
        [60, 90],
        [100, 200],
        [200, 500],
        [500, 1000],
    ]

    for ct in crossing_times:
        info = maf.empty_info()
        info["run_name"] = run_name
        info["observations_subset"] = subset
        mjd0 = np.min(visits_array["observationStartMJD"])
        metric = maf.MicrolensingMetric(mjd0=mjd0, name="Microlensing, crossing %i-%i days" % (ct[0], ct[1]))
        if quick_test:
            metric.generate_microlensing_events(n_events=20, min_crossing_time=ct[0], max_crossing_time=ct[1])
        else:
            metric.generate_microlensing_events(
                n_events=10000, min_crossing_time=ct[0], max_crossing_time=ct[1]
            )
        sl = maf.Slicer(nside=None, missing=0, ra=np.degrees(metric.ra), dec=np.degrees(metric.dec))

        mic_array, info = sl(visits_array, metric, info=info)

        ph = maf.PlotHealbin(info=info)

        summary_stats.append(maf.gen_summary_row(info, "Mean Microlensing ", np.mean(mic_array)))
        summary_stats.append(
            maf.gen_summary_row(
                info, "Uncert in Mean Microlensing ", np.mean(mic_array) / np.sqrt(np.sum(mic_array))
            )
        )

        fig = ph(
            np.degrees(metric.ra),
            np.degrees(metric.dec),
            mic_array,
            unit="Microlensing Crossing %i-%i days Detected (1,0)" % (ct[0], ct[1]),
        )
        fig_saver(fig, info=info)

    return summary_stats
