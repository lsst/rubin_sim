.. py:currentmodule:: rubin_sim.data

.. _prenight:

============================================================
Running pre-night simulations and adding them to the archive
============================================================

Introduction
============

Prenight simulation tools generate simulations of a few nights of observing, and add them to an archive.
These simulations can then be seen by `schedview`'s reporting tools, and used to generate pre-night brifing reports.
Normally, these simulation are run by submitting the `batch/run_prenight_sims.sh` script as a batch job, either by hand or using a cron job.
For more fine-grained control, the simulations and be run and added to the archive using lower-level tools,

Running a standard set of pre-night simulations
===============================================

The standard set of pre-night simulatios can be run by calling the `batch/run_prenight_sims.sh` shell (for SV) or `batch/run_auxtel_prenight_sims.sh` (for auxtel) scripts.

If `RUBIN_SIM_DIR` is the `rubin_sim` root director, it can be called thus::

    ${RUBIN_SIM_DIR}/batch/run_prenight_sims.sh

or thus::

    ${RUBIN_SIM_DIR}/batch/run_auxtel_prenight_sims.sh auxtel

or they can be submitted as a batch job to slurm.
At the USDF, there is an installation of `rubin_sim` at `/sdf/data/rubin/shared/scheduler/packages/rubin_sim` such that the commands become::

    /sdf/data/rubin/shared/scheduler/packages/rubin_sim/batch/run_auxtel_prenight_sims.sh
    sdf/data/rubin/shared/scheduler/packages/rubin_sim/batch/run_prenight_sims.sh


These batch jobs will record their output in:

* `/sdf/data/rubin/shared/scheduler/prenight/sbatch/run_prenight_sims_%A_%a.out` and
* `/sdf/data/rubin/shared/scheduler/prenight/sbatch/run_prenight_sims_%A_%a.err`

The working files can be found in `/sdf/data/rubin/shared/scheduler/prenight/work/run_prenight_sims/%Y-%m-%dT%H%M%`.

Automated runs of prenight simulations
======================================

Batch jobs are submitted automatically each morning following the `sbatch` commands shown above.
The `crontab` entries are::

    15 6 * * * /opt/slurm/slurm-curr/bin/sbatch /sdf/data/rubin/shared/scheduler/packages/rubin_sim/batch/run_auxtel_prenight_sims.sh 2>&1 >> /sdf/data/rubin/shared/scheduler/prenight/daily/daily_auxtel_cron.out
    30 6 * * * /opt/slurm/slurm-curr/bin/sbatch /sdf/data/rubin/shared/scheduler/packages/rubin_sim/batch/run_prenight_sims.sh 2>&1 >> /sdf/data/rubin/shared/scheduler/prenight/daily/daily_simonyi_cron.out


Custom runs of prenight simulations
===================================

Much of the `run_prenight_sims.sh` script is dedicated to figuring out what environment to use and configuring it.
If these defaults are not desired, a user can configure an environment to their liking, and run the simulation with the following steps.
This environment must have the following installed:

* `rubin_scheduler` (`github.com/lsst/rubin_sim`)
* `schedview`  (`github.com/lsst/schedview`)
* `ts_fbs_utils` (`github.com/lsst-ts/ts_fbs_utils.git`)
* `sims_sv_survey` (`github.com/lsst-sims/sims_sv_survey`)
* `rubin_nights` (`github.com/lsst-sims/rubin_night`)
* `lsst_resources` (`github.com/lsst/resources`)

First, retrieve the set of completed visits at the start of the nights to be simulated.
For example::

    DAYOBS="20250819"
    COMPLETED_OPSIM_DB="completed_visits.db"
    TOKEN_FILE="~/.usdf_access_token"
    fetch_sv_visits ${DAYOBS} ${COMPLETED_OPSIM_DB} ${TOKEN_FILE}

Second, create a pickle of the scheduler you want to run::

    SCHEDULER_CONFIG_SCRIPT="ts_config_ocs/Scheduler/feature_scheduler/maintel/fbs_config_sv_survey.py"
    SCHEDULER_PICKLE="scheduler.p"
    COMPLETED_OPSIM_DB="completed_visits.db"
    make_sv_scheduler ${SCHEDULER_PICKLE} --opsim ${COMPLETED_OPSIM_DB} --config-script ${SCHEDULER_CONFIG_SCRIPT}

Third, create a model observatory pickle::

    make_model_observatory observatory.p

Fourch, create an band scheduler pickle::

    make_band_scheduler band_scheduler.p

Finally, run the sim (and add it to the archive)::

    OPSIMRUN="prenight_nominal_$(date --iso=s)"
    LABEL="Nominal start and overhead, ideal conditions, run at $(date --iso=s)"
    NUM_NIGHTS=3
    ARCHIVE="s3://rubin:rubin-scheduler-prenight/opsim/"
    DELAY=0
    ANOM_SCALE=0
    ANOM_SEED=1
    run_sv_sim \
        ${SCHEDULER_PICKLE} \
        observatory.p \
        "" \
        ${DAYOBS} \
        ${NUM_NIGHTS} \
        ${OPSIMRUN} \
        --keep-rewards \
        --no-downtime \
        --label ${LABEL} \
        --archive ${ARCHIVE} \
        --capture-env \
        --delay ${DELAY} \
        --anom-overhead-scale ${ANOM_SCALE} \
        --anom-overhead-seed ${ANOM_SEED} \
        --tags mytag1 mytag2 mytag3

See the output of `run_sv_sim --help` for the meanings of the options.
