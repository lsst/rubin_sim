.. py:currentmodule:: rubin_sim.sim_archive.vseqarchive

.. _vseqarchivedemo:

===========================================================================
Example of running a simulation and adding it to the visit sequence archive
===========================================================================

Introduction
============

The precess used in this demonstration closely follows the bash script used to run prenight simuluations.
It has three major sections:

1. Prepare the environment.
2. Run the simulation, saving files to local disk as we go.
3. Add any date we want to keep to the archive.

Preparing the environment
=========================

In this demonstration, we will start with the lates tagged LSST stack available from ``cvmfs``.
(It is tested at the USDF, but should work anywhere ``cvmfs`` is installed.)
Find the version we want, and activate the ``conda`` environment::

    LATEST_TAGGED_STACK=$(
        find /cvmfs/sw.lsst.eu/almalinux-x86_64/lsst_distrib \
            -maxdepth 1 \
            -regex '.*/v[0-9]+\.[0-9]+\.[0-9]+' \
            -printf "%f\n" |
        sort -V |
        tail -1
    )
    source /cvmfs/sw.lsst.eu/almalinux-x86_64/lsst_distrib/${LATEST_TAGGED_STACK}/loadLSST-ext.sh

Next we install specific versions of packages in a separate location so we have full control for just this execution::

    SIM_PACKAGE_DIR=$(mktemp -d --suffix=.opsim_packages)
    export PYTHONPATH=${SIM_PACKAGE_DIR}:${PYTHONPATH}
    export PATH=${SIM_PACKAGE_DIR}/bin:${PATH}

Set the actual versions we want. These will usually be tags, but can be any github reference (e.g. tags, branches, commits)::


    RUBIN_SCHEDULER_REFERENCE="v3.18.1"
    RUBIN_SIM_REFERENCE="tickets/SP-2709"
    SCHEDVIEW_REFERENCE="tickets/SP-2167"
    TS_FBS_UTILS_REFERENCE="v0.18.0"
    LSST_SURVEY_SIM_REFERENCE="v0.2.0"
    RUBIN_NIGHTS_REFERENCE="v0.7.0"

Note that the ``obs_version_at_time`` command provided by ``schedview`` will query the EFD for the latest version being used (for some of these packages).
You can also use the github API and bash tools to get the highest tag for a repository, for example::

    SOMEREPO_LATEST_REFERENCE=$(
        curl -s https://api.github.com/repos/lsst/somerepo/tags |
        jq -r '.[].name' |
        egrep '^v[0-9]+\.[0-9]+\.[0-9]+$' |
        sort -V |
        tail -1
    )

Now actually do the install of all requested versions.
Pick up the current version of ``lsst-resources`` along the way.::

    pip install --no-deps --target=${SIM_PACKAGE_DIR} \
        git+https://github.com/lsst/rubin_sim.git@${RUBIN_SIM_REFERENCE} \
        git+https://github.com/lsst/schedview.git@${SCHEDVIEW_REFERENCE} \
        git+https://github.com/lsst-ts/ts_fbs_utils.git@${TS_FBS_UTILS_REFERENCE} \
        git+https://github.com/lsst-sims/lsst_survey_sim.git@${LSST_SURVEY_SIM_REFERENCE} \
        git+https://github.com/lsst-sims/rubin_nights.git@${RUBIN_NIGHTS_REFERENCE} \
        git+https://github.com/lsst/rubin_scheduler.git@${RUBIN_SCHEDULER_REFERENCE} \
        lsst-resources

Running the simulation
======================

Begin by creating a directory in which to work::

    WORK_DIR=${HOME}/devel/my_sample_opsim
    mkdir ${WORK_DIR}
    cd $WORK_DIR

Next we need is the scheduler configuration we're going to use.
We want the archive to track this, so we want it to be at a URL somewhere that it would be useful for the archive to reference.
Let's use something from github::

    TS_CONFIG_SCHEDULER_REFERENCE="develop"
    SCHED_CONFIG_FNAME="ts_config_scheduler/Scheduler/feature_scheduler/maintel/fbs_config_lsst_survey.py"
    echo "Using ts_config_scheduler ${SCHED_CONFIG_FNAME} from ${TS_CONFIG_SCHEDULER_REFERENCE}"
    git clone https://github.com/lsst-ts/ts_config_scheduler
    cd ts_config_scheduler
    git checkout "${TS_CONFIG_SCHEDULER_REFERENCE}"
    cd ${WORK_DIR}

We need to set the ``dayobs`` on which the simulation should start::

    DAYOBS="20260101"

Get the pre-existing visits from consdb::

    USDF_ACCESS_TOKEN_PATH=~/.lsst/usdf_access_token
    fetch_lsst_visits ${DAYOBS} completed_visits.db ${USDF_ACCESS_TOKEN_PATH}

Note that you will need a USDF access token.

Create a pickle of the scheduler to run, with completed visits pre-loaded, and write it to a pickle, ``scheduler.p``::

    make_lsst_scheduler scheduler.p --opsim completed_visits.db --config-script ${SCHED_CONFIG_FNAME}

Create a model observatory and write it to pickle file, ``observatory.p``::

    make_model_observatory observatory.p

Run the simulation, writing the result to files in the local directory::

    RESULTS_DIR="."
    OPSIMRUN="prenight_nominal_$(date --iso=s)"
    run_lsst_sim scheduler.p observatory.p "" ${DAYOBS} 1 "${OPSIMRUN}" --keep_rewards --results ${RESULTS_DIR}

There will now be an assortment of output files in the current working directory.

Adding entries to the visit sequence archive
============================================

Setting up the environment for the archive
------------------------------------------

The visit sequence archive has two components: a ``postgresql`` dataabase that tracks metadata, and a resource (directory or S3 bucket) in which the visits and other file content can be saved.

Begin by configuring the environment variables that the tools use to find the metadata database::

    export VSARCHIVE_PGDATABASE="opsim_log"
    export VSARCHIVE_PGHOST="usdf-maf-visit-seq-archive-tx.sdf.slac.stanford.edu"
    export VSARCHIVE_PGUSER="tester"
    export VSARCHIVE_PGSCHEMA="test"

Note that we have set ``VSARCHIVE_PGSCHEMA`` to ``test``, so metadata will be saved in a test schema.
The production schema is ``vsmd``.

Now, create a root for a demonstration resource in which to save the data itself::

    mkdir ${HOME}/devel/test_visitseq_archive
    export ARCHIVE_URL="file:///${HOME}/devel/test_visitseq_archive"

Make a simple utility shell function
------------------------------------

For the demonstration, it will be useful to query the metadata databas, but specifying all the connection parameters will be inconvenient.
One option would be to set the environment variables ``psql`` uses (``PGDATABASE``, ``PGHOST``, ``PGUSER``), but setting these might confuse other utilities that query different postgresql databases (like ``consdb``), so lets make a utility that just sets them for one command::

    vseq-psql() {
        PGDATABASE=${VSARCHIVE_PGDATABASE} \
        PGHOST=${VSARCHIVE_PGHOST} \
        PGUSER=${VSARCHIVE_PGUSER} \
        psql "$@"
    }

This results in a now shell command, ``vseq-psql``, that works just like ``psql``, but where the user does not need to specify connection parameters.

Adding an entry for pre-existing visits to the archive
------------------------------------------------------

We need to add two entries to the visit sequence archive, one for the pre-existing sequences of visits queried from consdb, and the other for the sequence generated by the simulation.

Begin by creating an entry for the pre-existing visits::

    COMPLETED=$(vseqarchive record-visitseq-metadata \
        completed \
        completed_visits.db \
        "Consdb query through 2025-10-31" \
        --first_day_obs 20250620 \
        --last_day_obs 20251031)

The ``COMPLETED`` UUID will now contain a reference for the sequence of visits returned from the consdb.
This command only adds an entry to the metadata, it does not save the visits themselves in the archive.
We can skip saving the visits themselves, if we are okay with relying on using consdb to recreate it.
(If you want to be sure, you can save them in the same way as simulated visits are saved below.)

Adding the simulation
---------------------

Now create an entry for the simulated visits::

    SIM_UUID=$(vseqarchive record-visitseq-metadata \
        simulations \
        opsim.db \
        "Test pre-night simulation 1" \
        --first_day_obs 20261201 \
        --last_day_obs 20261202
        )

This command only stored the bare minimum of metadata, and did not save the visits or any of the files in the archive.
We can now add additional metadata to the database::

    vseqarchive update-visitseq-metadata ${SIM_UUID} parent_visitseq_uuid ${COMPLETED}
    vseqarchive update-visitseq-metadata ${SIM_UUID} parent_last_day_obs 2025-12-31

    SCHEDULER_VERSION=$(python -c "import rubin_scheduler; print(rubin_scheduler.__version__)")
    vseqarchive update-visitseq-metadata ${SIM_UUID} scheduler_version "${SCHEDULER_VERSION}"

and the visits and rewards to the resource::

    vseqarchive archive-file ${SIM_UUID} opsim.db visits --archive-base ${ARCHIVE_URL}
    vseqarchive archive-file ${SIM_UUID} rewards.h5 rewards --archive-base ${ARCHIVE_URL}

We can alse add tags and comments to the metadata database::

    vseqarchive tag ${SIM_UUID} test prenight nominal
    vseqarchive comment ${SIM_UUID} "Just a test prenight"

Another option is to save specification for the ``conda`` environment::

    CONDA_HASH=$(vseqarchive record-conda-env)
    vseqarchive update-visitseq-metadata ${SIM_UUID} conda_env_sha256 ${CONDA_HASH}

Finaly, we can save statistics.
For the basic statistics tools currently available, the visits are needed in an HDF5 file, but in the above instructions we just have an sqlite3 file.
We can get the HDF5 by asking for the visits from the archive and giving it a destination filename with an ``.h5`` extension::

    vseqarchive get-file ${SIM_UUID} visits visits.h5

and then we can compute the statistics on our columns of interest and add them to the metadata database::

    vseqarchive add-nightly-stats ${SIM_UUID} visits.h5 fieldRA fieldDec azimuth altitude
