.. py:currentmodule:: rubin_sim.data

.. _prenight:

============================================================
Running pre-night simulations and adding them to the archive
============================================================

Introduction
============

Prenight simulation tools generate simulations of a few nights of observing, and add them to an archive.
These simulations can then be seen by `schedview`'s reporting tools, and used to generate pre-night brifing reports.
Normally, these simulation are run by submitting the `batch/run_prenight_sims.sh script <https://github.com/lsst-sims/lsst_survey_sim/blob/main/batch/run_prenight_sims.sh>` in the `lsst_survey_sim repository <https://github.com/lsst-sims/lsst_survey_sim/>`_ as a batch job, either by hand or using a cron job.
For more fine-grained control, the simulations and be run and added to the archive using lower-level tools,

Running a standard set of pre-night simulations
===============================================

The standard set of pre-night simulatios can be run by calling the `batch/run_prenight_sims.sh` shell (for SV) or `batch/run_auxtel_prenight_sims.sh` (for auxtel) scripts.

If `LSST_SURVEY_SIM_DIR` is the `lsst_survey_sim` root directory, it can be called thus::

    ${LSST_SURVEY_SIM_DIR}/batch/run_prenight_sims.sh

or thus::

    ${LSST_SURVEY_SIM_DIR}/batch/run_auxtel_prenight_sims.sh auxtel

or they can be submitted as a batch job to slurm.
At the USDF, there is an installation of `lsst_survey_sim` at `/sdf/data/rubin/shared/scheduler/packages/lsst_survey_sim` such that the commands become::

    /sdf/data/rubin/shared/scheduler/packages/lsst_survey_sim/batch/run_auxtel_prenight_sims.sh
    /sdf/data/rubin/shared/scheduler/packages/lsst_survey_sim/batch/run_prenight_sims.sh


These batch jobs will record their output in:

* `/sdf/data/rubin/shared/scheduler/prenight/sbatch/run_prenight_sims_%A_%a.out` and
* `/sdf/data/rubin/shared/scheduler/prenight/sbatch/run_prenight_sims_%A_%a.err`

The working files can be found in `/sdf/data/rubin/shared/scheduler/prenight/work/run_prenight_sims/%Y-%m-%dT%H%M%`.

Automated runs of prenight simulations
======================================

Batch jobs are submitted automatically each morning following the `sbatch` commands shown above.
The `crontab` entries are::

    15 6 * * * /opt/slurm/slurm-curr/bin/sbatch /sdf/data/rubin/shared/scheduler/packages/lsst_survey_sim/batch/run_auxtel_prenight_sims.sh 2>&1 >> /sdf/data/rubin/shared/scheduler/prenight/daily/daily_auxtel_cr on.out
    55 6 * * * /opt/slurm/slurm-curr/bin/sbatch /sdf/data/rubin/shared/scheduler/packages/lsst_survey_sim/batch/run_prenight_sims.sh 2>&1 >> /sdf/data/rubin/shared/scheduler/prenight/daily/daily_simonyi_cron.out

Custom runs of prenight simulations
===================================

Preparation of the environment
------------------------------

Much of the `run_prenight_sims.sh` script is dedicated to figuring out what environment to use and configuring it.
If these defaults are not desired, a user can configure an environment to their liking, and run the simulation with the following steps.
The dependencies of `lsst_survey_sim` provide most of what the preniight simulations require to run.
The additional dependencies beyond what `lsst_survey_sim` requires are:

- `click`
- `psycopg2`
- `botocore`
- `boto3`
- `lsst-resources`

You will also need to configure environment variables for access to the prenight S3 bucket and metadata database::

    export ARCHIVE="s3://rubin:rubin-scheduler-prenight/opsim/vseq/"
    export VSARCHIVE_PGDATABASE="opsim_log"
    export VSARCHIVE_PGHOST="usdf-maf-visit-seq-archive-tx.sdf.slac.stanford.edu"
    export VSARCHIVE_PGUSER="writer"
    export VSARCHIVE_PGSCHEMA="vsmd"

Get completed visits
--------------------

First, retrieve the set of completed visits at the start of the nights to be simulated.
For example::

    DAYOBS="$(date -u --date='-12 hours' +'%Y%m%d')"
    TOKEN_FILE="~/.usdf_access_token"
    fetch_lsst_visits ${DAYOBS} completed_visits.db ${TOKEN_FILE}

You can register this visit list in the prenight simulation metadata database::

    LASTNIGHTISO="$(date --date='-36 hours' -u +'%F')"
    COMPLETED=$(vseqarchive record-visitseq-metadata \
        completed \
        completed_visits.db \
        "Consdb query through ${LASTNIGHTISO}" \
        --first_day_obs 20250620 \
        --last_day_obs ${LASTNIGHTISO})

Note that the `fetch_lsst_visits` command required the `dayobs` of the night you are about to simulate in `YYYYMMDD` format,
while the `vseqarchive record-visitseq-metadata` command takes the `dayobs` of the last night in the completed visits
(probably the night before the one used in `fetch_lsst_visits`) in either `YYYYMMDD` or `YYYY-MM-DD` format.

The `vseqarchive record-visitseq-metadata` command does not actually save the visits themselves in the S3 bucket:
it only records the checksum of the set of visits in the metadata database and assigns the set a UUID.
Above, this UUID is saved in the `COMPLETED` environment variable.

Create pickles of the objects you need to run the simulation
------------------------------------------------------------

If you do not already have a pickle, begin by getting the configuration you want from `ts_config_scheduler`::

    TS_CONFIG_SCHEDULER_REFERENCE="develop"
    SCHED_CONFIG_FNAME="ts_config_scheduler/Scheduler/feature_scheduler/maintel/fbs_config_lsst_survey.py"
    git clone --depth 1 https://github.com/lsst-ts/ts_config_scheduler
    cd ts_config_scheduler
    git fetch --depth 1 origin "${TS_CONFIG_SCHEDULER_REFERENCE}"
    git checkout FETCH_HEAD

Create a pickle of the scheduler you want to run::

    make_lsst_scheduler scheduler.p --opsim completed_visits.db --config_script ${SCHED_CONFIG_FNAME}

Create a model observatory pickle::

    make_model_observatory observatory.p

Run the simulation
------------------

Set a directory in which to save the results of the simulations::

    OPSIM_RESULTS_DIR=/my/results/directory
    mkdir ${OPSIM_RESULTS_DIR}

Finally, run the sim (and add it to the archive)::

    OPSIMRUN="prenight_nominal_$(date --iso=s)"
    NUM_NIGHTS="3"
    LABEL="Nominal start and overhead, ideal conditions, run at $(date --iso=s)"
    run_lsst_sim scheduler.p observatory.p "" ${DAYOBS} ${NUM_NIGHTS} "${OPSIMRUN}" \
    --keep_rewards --label "${LABEL}" \
    --delay 0 --anom_overhead_scale 0 \
    --results ${OPSIM_RESULT_DIR}

Add the simulation to the S3 bucket and metadata database
---------------------------------------------------------

Set `LAST_DAYOBS` to whatever the last dayobs in the simualtion was.
In the above example, that's three days ofter tonight::

    LAST_DAYOBS="$(date -u --date='+36 hours' +'%Y%m%d')"

Register the simulation in the database::

    SIM_UUID=$(vseqarchive record-visitseq-metadata \
        simulations \
        ${OPSIM_RESULT_DIR}/opsim.db \
        "${LABEL}" \
        --first_day_obs ${DAYOBS} \
        --last_day_obs ${LAST_DAYOBS}
        )

Add additional metadata for this entry recording where the pre-simulation visits came from::

    vseqarchive update-visitseq-metadata ${SIM_UUID} parent_visitseq_uuid ${COMPLETED}
    vseqarchive update-visitseq-metadata ${SIM_UUID} parent_last_day_obs ${LASTNIGHTISO}

Set appropriate tags.
The following set of tags make the visit sequence recognized as a standard prenight by tools like `obsloctap` and the prenight brifing report::

    vseqarchive tag ${SIM_UUID} prenight ideal nominal

Save the visits themselves to the S3 bucket, simultaneously updating the metadata database with references for their locations::

    vseqarchive archive-file ${SIM_UUID} ${OPSIM_RESULT_DIR}/opsim.db visits --archive-base ${ARCHIVE}

If you used the `--keep-rewords` option in `run_lsst_sim` (as was done above), save the rewards file you generated as well::

    vseqarchive archive-file ${SIM_UUID} ${OPSIM_RESULT_DIR}/rewards.h5 rewards --archive-base ${ARCHIVE}

Update the indexes of prenight simulations
------------------------------------------

Some services (e.g. `obsloctap`) need to query for prenight simulation of a given night, but do not have access to the prenight simulation database.
To support them, we maintain the results of queries to the metadata database for prenight sims of a given night in a pre-determined location in the S3 bucket.
When we create a new simulation, we need to update the objects with the results in the S3 bucket.
We need to do this for each night simulated.
In this example, the `DAYOBS` environment variable contains the first night (tonight).
We specified that we were simulating 3 nights above, so set variables for tomorrow and the day after, and iterate over them::

    NEXT_DAYOBS="$(date -u --date='+12 hours' +'%Y%m%d')"
    LAST_DAYOBS="$(date -u --date='+36 hours' +'%Y%m%d')"
    DAYOBS_SIMULATED="$DAYOBS $NEXT_DAYOBS $LAST_DAYOBS"
    for DAYOBS_TO_INDEX in ${DAYOBS_SIMULATED}; do
        vseqarchive make-prenight-index ${DAYOBS_TO_INDEX} simonyi
    done
