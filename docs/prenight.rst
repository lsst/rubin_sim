.. py:currentmodule:: rubin_sim.data

.. _prenight:

####################
Prenight Simulations
####################

===================================================================
Generating new pre-night simulations and adding them to the archive
===================================================================

Introduction
============

Prenight simulation tools generate simulations of a few nights of observing, and add them to an archive.
These simulations can then be seen by ``schedview``'s reporting tools, and used to generate pre-night brifing reports.
Normally, these simulation are run by submitting the `batch/run_prenight_sims.sh script <https://github.com/lsst-sims/lsst_survey_sim/blob/main/batch/run_prenight_sims.sh>`_ in the `lsst_survey_sim repository <https://github.com/lsst-sims/lsst_survey_sim/>`_ as a batch job, either by hand or using a cron job.
For more fine-grained control, the simulations and be run and added to the archive using lower-level tools,

Running a standard set of pre-night simulations
===============================================

The standard set of pre-night simulations can be run by calling the ``batch/run_prenight_sims.sh`` shell (for SV) or ``batch/run_auxtel_prenight_sims.sh`` (for auxtel) scripts.
These scripts are "protected" by gate files: for user ``${USER}`` to run ``batch/run_prenight_sims.sh``, the file ``/sdf/data/rubin/shared/scheduler/cron_gates/run_prenight_sims/${USER}`` must exist,
and for that user to run ``batch/run_auxtel_prenight_sims.sh``, the file ``/sdf/data/rubin/shared/scheduler/cron_gates/run_auxtel_prenight_sims/${USER}`` must exist.
(This was done so that any user with write access to ``/sdf/data/rubin/shared/scheduler/cron_gates/${SCRIPT}`` can stop a cron job that runs the corresponding script from doing anything, even if that cron job is not owned by that user.)

If ``LSST_SURVEY_SIM_DIR`` is the ``lsst_survey_sim`` root directory, it can be called thus::

    ${LSST_SURVEY_SIM_DIR}/batch/run_prenight_sims.sh

or thus::

    ${LSST_SURVEY_SIM_DIR}/batch/run_auxtel_prenight_sims.sh auxtel

or they can be submitted as a batch job to slurm.
At the USDF, there is an installation of ``lsst_survey_sim`` at ``/sdf/data/rubin/shared/scheduler/packages/lsst_survey_sim`` such that the commands become::

    /sdf/data/rubin/shared/scheduler/packages/lsst_survey_sim/batch/run_auxtel_prenight_sims.sh
    /sdf/data/rubin/shared/scheduler/packages/lsst_survey_sim/batch/run_prenight_sims.sh


These scripts build python environments in the USDF scratch disks of the user running the script, in directories of the form::

  /sdf/scratch/users/${USER:0:1}/${USER}/prenight_venvs/prenight-YYYY-MM-DDTHHMMSS-XXXXXX

Where ``${USER:0:1}`` is the first letter of a username, and the ``XXXXXX`` is replaced by random characters.
For example, a recent environment created was: ``/sdf/scratch/users/n/neilsen/prenight_venvs/prenight-2025-12-16T141515-gCnXfu``.

These batch jobs will record their logs (stderr and stdout) in:

* ``/sdf/data/rubin/shared/scheduler/prenight/sbatch/run_prenight_sims_%A_%a.out`` and
* ``/sdf/data/rubin/shared/scheduler/prenight/sbatch/run_prenight_sims_%A_%a.err``

The working files can be found in ``/sdf/data/rubin/shared/scheduler/prenight/work/run_prenight_sims/%Y-%m-%dT%H%M%``.

Automated runs of prenight simulations
======================================

Batch jobs are submitted automatically each morning following the ``sbatch`` commands shown above.
The ``crontab`` entries are::

    15 6 * * * /opt/slurm/slurm-curr/bin/sbatch /sdf/data/rubin/shared/scheduler/packages/lsst_survey_sim/batch/run_auxtel_prenight_sims.sh 2>&1 >> /sdf/data/rubin/shared/scheduler/prenight/daily/daily_auxtel_cr on.out
    55 6 * * * /opt/slurm/slurm-curr/bin/sbatch /sdf/data/rubin/shared/scheduler/packages/lsst_survey_sim/batch/run_prenight_sims.sh 2>&1 >> /sdf/data/rubin/shared/scheduler/prenight/daily/daily_simonyi_cron.out

If necessary, these cron jobs can be stopped from doing anything, even a user that does not own the cron job, if they have write access to ``/sdf/data/rubin/shared/scheduler/cron_gates/${SCRIPT}``.
This is accomplished using gate files: early in each script, the script checks for the existence of a file with name ``/sdf/data/rubin/shared/scheduler/cron_gates/${SCRIPT_NAME}/${USER}`` and aborts if it does not exist.
Any user with write access to ``/sdf/data/rubin/shared/scheduler/cron_gates/${SCRIPT_NAME}`` can create or remove files in that directory, so a user can cause these scripts to immediately abort
when started by a cron job owned by a different user by removing ``/sdf/data/rubin/shared/scheduler/cron_gates/${SCRIPT_NAME}/${CRON_JOB_USER}``.

So, to stop the ``run_prenight_sims.sh`` cron job owned by user ``neilsen``, remove the file ``/sdf/data/rubin/shared/scheduler/cron_gates/run_prenight_sims/neilsen``,
and to stop the ``run_auxtel_prenight_sims.sh`` cron job owned by user ``neilsen``, remove the file ``/sdf/data/rubin/shared/scheduler/cron_gates/run_auxtel_prenight_sims/neilsen``,

The logs of the cron jobs (and any other executions of these scripts submitted using ``sbatch``) can be found in ``/sdf/data/rubin/shared/scheduler/prenight/sbatch/run_prenight_sims_%A_%a.out``,
where ``%A`` is the slurm "Job array's master job allocation number" and ``%a`` is the slum "Job array ID (index) number".

Updating the versions of the scripts
====================================

The batch jobs run versions of ``lsst_survey_sim/batch/run_prenight_sims.sh`` and ``lsst_survey_sim/run_auxtel_prenight_sims.sh`` installed in ``/sdf/data/rubin/shared/scheduler/packages``.
Versions of ``lsst_survey_sim`` installed in ``/sdf/data/rubin/shared/scheduler/packages`` can be updated as follows.

First, if one does not exist already, tag ``lsst_survey_sim`` at the commit you want.
To find the next available tag::

  curl -s https://api.github.com/repos/lsst-sims/lsst_survey_sims/tags \
      | jq -r '.[].name' \
      | egrep '^v[0-9]+.[0-9]+.[0-9]+.*$' \
      | sort -V

Make and push a new tag (with the base of the repository at the commit you want as the current working directory)::

  NEWVERSION="0.1.0.dev2"
  NEWTAG=v${NEWVERSION}
  echo "New version is ${NEWVERSION} with tag ${NEWTAG}"
  echo ""
  git tag ${NEWTAG}
  git push origin tag ${NEWTAG}

Then install it in ``/sdf/data/rubin/shared/scheduler/packages``::

  PACKAGEDIR="/sdf/data/rubin/shared/scheduler/packages"
  TARGETDIR="${PACKAGEDIR}/lsst_survey_sim-${NEWVERSION}"
  PIPORIGIN="git+https://github.com/lsst-sims/lsst_survey_sim.git@${NEWTAG}"
  echo "Installing from ${PIPORIGIN} to ${TARGETDIR}"
  echo ""
  pip install \
      --no-deps \
      --target=${TARGETDIR} \
      ${PIPORIGIN}

If you want to make it the default version, replace the link::

  if test -L ${PACKAGEDIR}/lsst_survey_sim ; then
    rm ${PACKAGEDIR}/lsst_survey_sim
  fi
  ln -s ${TARGETDIR} ${PACKAGEDIR}/lsst_survey_sim

Custom runs of prenight simulations
===================================

Preparation of the environment
------------------------------

Much of the ``run_prenight_sims.sh`` script is dedicated to figuring out what environment to use and configuring it.
If these defaults are not desired, a user can configure an environment to their liking, and run the simulation with the following steps.
The dependencies of ``lsst_survey_sim`` provide most of what the preniight simulations require to run.
The additional dependencies beyond what ``lsst_survey_sim`` requires are:

- ``click``
- ``psycopg2``
- ``botocore``
- ``boto3``
- ``lsst-resources``

You will also need to configure environment variables for access to the prenight S3 bucket and metadata database::

    export VSARCHIVE_PGDATABASE="opsim_log"
    export VSARCHIVE_PGHOST="usdf-maf-visit-seq-archive-tx.sdf.slac.stanford.edu"
    export VSARCHIVE_PGUSER="writer"
    export VSARCHIVE_PGSCHEMA="vsmd"

Get completed visits
--------------------

First, retrieve the set of completed visits at the start of the nights to be simulated.
For example::

    DAYOBS="$(date -u --date='-12 hours' +'%Y%m%d')"
    TOKEN_FILE="~/.lsst/usdf_access_token"
    fetch_lsst_visits ${DAYOBS} completed_visits.db ${TOKEN_FILE}

You can register this visit list in the prenight simulation metadata database::

    PREVIOUS_NIGHT_ISO="$(date --date='-36 hours' -u +'%F')"
    COMPLETED=$(vseqarchive record-visitseq-metadata \
        completed \
        completed_visits.db \
        "Consdb query through ${PREVIOUS_NIGHT_ISO}" \
        --first_day_obs 20250620 \
        --last_day_obs ${PREVIOUS_NIGHT_ISO})

The ``vseqarchive record-visitseq-metadata`` command creates a row in the ``completed`` table in the visit sequence archive metadata database,
automatically assigning it a UUID and recording a checksum of the table of visits and the label and date range provided as arguments in the above command.
It **does not** save the visits themselves in the S3 bucket:
it only records the checksum of the set of visits in the metadata database and assigns the set a UUID.
Above, this automatically generated UUID is saved in the ``COMPLETED`` environment variable.

Note that the ``fetch_lsst_visits`` command required the ``dayobs`` of the night you are about to simulate in ``YYYYMMDD`` format,
while the ``vseqarchive record-visitseq-metadata`` command takes the ``dayobs`` of the last night in the completed visits
(probably the night before the one used in ``fetch_lsst_visits``) in either ``YYYYMMDD`` or ``YYYY-MM-DD`` format.

Create pickles of the objects you need to run the simulation
------------------------------------------------------------

If you do not already have a pickle, begin by getting the configuration you want from ``ts_config_scheduler``::

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

Set ``LAST_DAYOBS`` to whatever the last dayobs in the simualtion was.
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
    vseqarchive update-visitseq-metadata ${SIM_UUID} parent_last_day_obs ${PREVIOUS_NIGHT_ISO}

Set appropriate tags.
The following set of tags make the visit sequence recognized as a standard prenight by tools like ``obsloctap`` and the prenight brifing report::

    vseqarchive tag ${SIM_UUID} prenight ideal nominal

Save the visits themselves to the S3 bucket, simultaneously updating the metadata database with references for their locations::

    ARCHIVE="s3://rubin:rubin-scheduler-prenight/opsim/vseq/"
    vseqarchive archive-file ${SIM_UUID} ${OPSIM_RESULT_DIR}/opsim.db visits --archive-base ${ARCHIVE}

When the opsim visits file has an extension that indicates that it is an ``sqlite3`` database (as it will be in this example),
it will be automatically converted to an HDF5 file for archiving.

If you used the ``--keep-rewords`` option in ``run_lsst_sim`` (as was done above), save the rewards file you generated as well::

    vseqarchive archive-file ${SIM_UUID} ${OPSIM_RESULT_DIR}/rewards.h5 rewards --archive-base ${ARCHIVE}

Update the indexes of prenight simulations
------------------------------------------

Some services (e.g. ``obsloctap``) need to query for prenight simulation of a given night, but do not have access to the prenight simulation database.
To support them, we maintain the results of queries to the metadata database for prenight sims of a given night in a pre-determined location in the S3 bucket.
When we create a new simulation, we need to update the objects with the results in the S3 bucket.
We need to do this for each night simulated.
In this example, the ``DAYOBS`` environment variable contains the first night (tonight).
We specified that we were simulating 3 nights above, so set variables for tomorrow and the day after, and iterate over them::

    NEXT_DAYOBS="$(date -u --date='+12 hours' +'%Y%m%d')"
    LAST_DAYOBS="$(date -u --date='+36 hours' +'%Y%m%d')"
    DAYOBS_SIMULATED="$DAYOBS $NEXT_DAYOBS $LAST_DAYOBS"
    for DAYOBS_TO_INDEX in ${DAYOBS_SIMULATED}; do
        vseqarchive make-prenight-index ${DAYOBS_TO_INDEX} simonyi
    done

==========================================
Retrieving indexes of prenight simulations
==========================================

The typical user of the prenight simulations should use the ``rubin_sim.sim_archive.prenightindex.get_prenight_index`` function.

``get_prenight_index`` first attempts to query the metadata database for the data necessary to create the index.
If this query fails (e.g. due to access restrictions), it falls back on reading the index in the S3 bucket created by the ``vseqarchive make-prenight-index`` command.
For example::

    >>> from rubin_sim.sim_archive.prenightindex import get_prenight_index
    >>> pn_index_df = get_prenight_index('2025-12-10', 'simonyi')
    >>> pn_index_df.tail()
                                        sim_creation_day_obs  daily_id                                     visitseq_label  ...                                   tags comments                                              files
    visitseq_uuid                                                                                                           ...
    7468dd4a-54ef-44fb-8bc0-8091ad49e9ab           2025-12-10         2  Nominal start and overhead, ideal conditions, ...  ...             [ideal, nominal, prenight]       {}  {'rewards': 's3://rubin:rubin-scheduler-prenig...
    c24d9a7a-995e-4b3a-8da6-40ac7ed3688d           2025-12-10         3  Start time delayed by 60 minutes, nominal slew...  ...            [delay_60, ideal, prenight]       {}  {'rewards': 's3://rubin:rubin-scheduler-prenig...
    ff6df943-8a5b-4871-bbe2-5729f8aa0820           2025-12-10         4  Start time delayed by 240 minutes, nominal sle...  ...           [delay_240, ideal, prenight]       {}  {'rewards': 's3://rubin:rubin-scheduler-prenig...
    6ee29551-5326-4d3b-9b2e-ba50c9e292d6           2025-12-10         5  Anomalous overhead (101, 0.1), nominal start, ...  ...  [anomalous_overhead, ideal, prenight]       {}  {'rewards': 's3://rubin:rubin-scheduler-prenig...
    d08fb912-d57b-46ab-a911-ee97fd1a013f           2025-12-10         6  Anomalous overhead (102, 0.1), nominal start, ...  ...  [anomalous_overhead, ideal, prenight]       {}  {'rewards': 's3://rubin:rubin-scheduler-prenig...

    [5 rows x 17 columns]
    >>> pn_index_df.columns
    Index(['sim_creation_day_obs', 'daily_id', 'visitseq_label', 'visitseq_url',
        'telescope', 'first_day_obs', 'last_day_obs', 'creation_time',
        'scheduler_version', 'config_url', 'sim_runner_kwargs',
        'conda_env_sha256', 'parent_visitseq_uuid', 'parent_last_day_obs',
        'tags', 'comments', 'files'],
        dtype='object')

The columns of this `pandas.DataFrame` are as follows:

.. list-table:: VisitSequence Metadata Fields
   :widths: 25 20 55
   :header-rows: 1

   * - Field
     - Type
     - Description
   * - visitseq_label
     - ``str``
     - Human-readable label for the simulation
   * - visitseq_url
     - ``str`` or ``None``
     - URL of the visits file (HDF5)
   * - telescope
     - ``str``
     - Telescope identifier (``simonyi`` or ``auxtel``)
   * - first_day_obs
     - ``datetime.date`` or ``None``
     - Local evening of the first night covered by the simulation
   * - last_day_obs
     - ``datetime.date`` or ``None``
     - Local evening of the last night covered
   * - creation_time
     - ``pd.Timestamp`` or ``None``
     - When the simulation was run (UTC)
   * - scheduler_version
     - ``str`` or ``None``
     - Version of **rubin_scheduler** used
   * - config_url
     - ``str`` or ``None``
     - URL of the configuration script (if recorded)
   * - sim_runner_kwargs
     - ``dict`` or ``None``
     - Arguments passed to ``sim_runner`` (JSON-serialisable)
   * - conda_env_sha256
     - ``str`` (hex) or ``None``
     - SHA-256 hash of the Conda environment used
   * - parent_visitseq_uuid
     - ``uuid.UUID`` or ``None``
     - UUID of the parent simulation (if this is a continuation)
   * - parent_last_day_obs
     - ``datetime.date`` or ``None``
     - Last night of the parent simulation
   * - sim_creation_day_obs
     - ``datetime.date`` or ``None``
     - Derived from ``creation_time`` (UTC-12 day)
   * - daily_id
     - ``int``
     - Sequential ID for simulations created on the same day (1-based)
   * - stats
     - ``dict`` or ``None``
     - Aggregated nightly statistics for the simulation. Structure follows the output of ``VisitSequenceArchiveMetadata.sims_on_night_with_stats`` – a mapping from metric name to a sub-dictionary of statistic values (``count``, ``mean``, ``p05``, ``q1``, ``median``, ``q3``, ``p95``, ``min``, ``max``, etc.)
   * - tags
     - ``list[str]``
     - List of tags attached to the simulation (e.g., ``["prenight","ideal","nominal"]``)
   * - files
     - ``dict``
     - Mapping from file type (e.g., ``"scheduler"``, ``"observatory"``, ``"rewards"``) to the URL where that file is stored
   * - comments
     - ``dict``
     - Mapping from comment timestamp (ISO string) to comment text


This data may also be read directly from the S3 bucket for the desired night (dayobs).
The URL for the json data is of the form::

    s3://rubin:rubin-scheduler-prenight/opsim/prenight_index/{telescope}/{year}/{month}/{telescope}_prenights_for_{YYYY-MM-DD}.json

The content is a json file with (hex representations of) simulation UUIDs as keys and dictionaries with metadata as the values of those keys.
The start of a sample index looks like this::

    {
    "e572d2b3-d07b-496b-a0c3-a69c915a5ad4":{
        "sim_creation_day_obs":"2025-12-08",
        "daily_id":2,
        "visitseq_label":"Nominal start and overhead, ideal conditions, run at 2025-12-08T06:59:18-08:00",
        "visitseq_url":"s3:\/\/rubin:rubin-scheduler-prenight\/opsim\/vseq\/simonyi\/2025-12-08\/e572d2b3-d07b-496b-a0c3-a69c915a5ad4\/visits.h5",
        "telescope":"simonyi",
        "first_day_obs":"2025-12-08",
        "last_day_obs":"2025-12-10",
        "creation_time":"2025-12-08T15:09:51.399Z",
        "scheduler_version":"",
        "config_url":null,
        "sim_runner_kwargs":null,
        "conda_env_sha256":"52333af3b43db7704318f35d4fe7b9ddf770ca49c892f9681933c20de36cb0b3",
        "parent_visitseq_uuid":"93b058db-c0f4-4395-b856-4064454299f3",
        "parent_last_day_obs":"2025-12-07",
        "tags":[
            "ideal",
            "nominal",
            "prenight"
        ],
        "comments":{

        },
        "files":{
            "rewards":"s3:\/\/rubin:rubin-scheduler-prenight\/opsim\/vseq\/simonyi\/2025-12-08\/e572d2b3-d07b-496b-a0c3-a69c915a5ad4\/rewards.h5"
        },
        "stats":{
            "azimuth":{
                "q1":174.9977611669,
                "q3":295.2438484415,
                "max":359.7925184257,
                "min":0.0004341731,
                "p05":20.4236498145,
                "p95":344.7673586831,
                "std":82.3488406074,
                "mean":226.8318565587,
                "count":562,
                "median":222.7917334351,
                "accumulated":false
            },

It can be read using `lsst.resource.ResourcePath`::

    >>> from lsst.resources import ResourcePath
    >>> import json
    >>> prenight_index_rp = ResourcePath('s3://rubin:rubin-scheduler-prenight/opsim/prenight_index/simonyi/2025/12/simonyi_prenights_for_2025-12-10.json')
    >>> prenight_index_json = prenight_index_rp.read().decode()
    >>> prenight_index = json.loads(prenight_index_json)

===============================================
Retrieving the table of visits for a simulation
===============================================

The indexed of simulations above include the ``visitseq_url`` referece to the HDF5 file with table of simulated visits themselves.
The simplest way to read these is with the ``rubin_sim.sim_archive.vseqarchive.get_visits`` utility::

    >>> from rubin_sim.sim_archive.vseqarchive import get_visits
    >>> visitseq_url = 's3://rubin:rubin-scheduler-prenight/opsim/vseq/simonyi/2025-12-08/e572d2b3-d07b-496b-a0c3-a69c915a5ad4/visits.h5'
    >>> visits = get_visits(visitseq_url)
    >>> visits.head()
    observationId    fieldRA   fieldDec  observationStartMJD  flush_by_mjd  visitExposureTime band filter  rotSkyPos  ...      moonRA    moonDec  moonDistance  solarElong  moonPhase  cummTelAz  observation_reason  science_program  cloud_extinction
    0              0  25.485040 -21.566665         61018.046632           0.0              150.0    r   r_57  98.333137  ...  136.719114  19.773488    116.187971  113.972844  68.590565   8.975898                cwfs       BLOCK-T630               0.0
    1              1  24.731275 -13.747277         61018.048581           0.0               30.0    r   r_57  92.888439  ...  136.749727  19.766991    115.028941  118.157641  68.575266   4.362920           singles_r        BLOCK-407               0.0
    2              2  26.590414 -11.364966         61018.049060           0.0               30.0    r   r_57  97.394703  ...  136.757232  19.765397    112.626404  121.019966  68.571516  10.054260           singles_r        BLOCK-407               0.0
    3              3  25.249855  -8.776389         61018.049521           0.0               30.0    r   r_57  93.312776  ...  136.764443  19.763864    113.121836  121.402330  68.567913   5.944932           singles_r        BLOCK-407               0.0
    4              4  23.904252  -6.185058         61018.049972           0.0               30.0    r   r_57  90.071152  ...  136.771498  19.762364    113.578828  121.674979  68.564388   2.665790           singles_r        BLOCK-407               0.0

The columns contained are those generated by the ``rubin_scheduler`` ``opsim`` simulation, and can therefore be analyzed using ``rubin_sim.maf`` tools.

The ``get_visits`` utility used above can accept a ``stackers`` keyword argument that takes a list of ``maf`` stackers, which supplement the base columns included in the HDF5 file with derived columns.

The HDF5 in the S3 store can alse be read directly using lower level tools::

    >>> from lsst.resources import ResourcePath
    >>> from tempfile import TemporaryDirectory
    >>> import pandas as pd
    >>> visitseq_url = 's3://rubin:rubin-scheduler-prenight/opsim/vseq/simonyi/2025-12-08/e572d2b3-d07b-496b-a0c3-a69c915a5ad4/visits.h5'
    >>> visitseq_origin = ResourcePath(visitseq_url)
    >>> with TemporaryDirectory() as temp_dir:
    ...     h5_destination = ResourcePath(temp_dir).join("visits.h5")
    ...     h5_destination.transfer_from(visitseq_origin, "copy")
    ...     visits = pd.read_hdf(h5_destination.ospath, key="observations")
    ...
    >>> visits.head()
    observationId    fieldRA   fieldDec  observationStartMJD  flush_by_mjd  visitExposureTime band filter  rotSkyPos  ...      moonRA    moonDec  moonDistance  solarElong  moonPhase  cummTelAz  observation_reason  science_program  cloud_extinction
    0              0  25.485040 -21.566665         61018.046632           0.0              150.0    r   r_57  98.333137  ...  136.719114  19.773488    116.187971  113.972844  68.590565   8.975898                cwfs       BLOCK-T630               0.0
    1              1  24.731275 -13.747277         61018.048581           0.0               30.0    r   r_57  92.888439  ...  136.749727  19.766991    115.028941  118.157641  68.575266   4.362920           singles_r        BLOCK-407               0.0
    2              2  26.590414 -11.364966         61018.049060           0.0               30.0    r   r_57  97.394703  ...  136.757232  19.765397    112.626404  121.019966  68.571516  10.054260           singles_r        BLOCK-407               0.0
    3              3  25.249855  -8.776389         61018.049521           0.0               30.0    r   r_57  93.312776  ...  136.764443  19.763864    113.121836  121.402330  68.567913   5.944932           singles_r        BLOCK-407               0.0
    4              4  23.904252  -6.185058         61018.049972           0.0               30.0    r   r_57  90.071152  ...  136.771498  19.762364    113.578828  121.674979  68.564388   2.665790           singles_r        BLOCK-407               0.0

=============================================================
Retrieving the latest nominal prenight simulation for a night
=============================================================

A particularly common use of the prenight archive is to simply return visits from the latest nominal simulation for a given night and telescope.
The ``rubin_sim.sim_archive.fetch_sim_for_nights`` utility simplifies this task,
reading the prenight index, finding the ones with desirend metadata, and loading the corresponding visits from the latest matching simulation::

    >>> from rubin_sim.sim_archive import fetch_sim_for_nights
    >>> visits = fetch_sim_for_nights('2025-12-10', '2025-12-12', which_sim={'tags': ('ideal', 'nominal'), 'telescope': 'simonyi', 'max_simulation_age': 30})
    >>> visits.head()
    index  observationId    fieldRA   fieldDec  observationStartMJD  flush_by_mjd  visitExposureTime band filter  ...      moonRA   moonDec  moonDistance  solarElong  moonPhase  cummTelAz  observation_reason  science_program  cloud_extinction
    0      0              0  27.728086 -21.211953         61020.047709           0.0              150.0    r   r_57  ...  161.557441  8.748750    133.877255  114.115281  55.110377   7.944240                cwfs       BLOCK-T630               0.0
    1      1              1  28.993789 -13.374352         61020.049674           0.0               30.0    r   r_57  ...  161.587119  8.738341    133.309112  119.900287  55.094211  10.547646           singles_r        BLOCK-407               0.0
    2      2              2  28.459780 -10.287782         61020.050117           0.0               30.0    r   r_57  ...  161.593783  8.736001    133.791467  121.288930  55.090581   8.213424           singles_r        BLOCK-407               0.0
    3      3              3  27.969350  -7.184119         61020.050559           0.0               30.0    r   r_57  ...  161.600448  8.733660    134.078759  122.629332  55.086950   6.614253           singles_r        BLOCK-407               0.0
    4      4              4  27.540610  -4.060129         61020.051002           0.0               30.0    r   r_57  ...  161.607111  8.731318    134.143320  123.935418  55.083320   5.542837           singles_r        BLOCK-407               0.0

The above call returns the latest simulation of the ``simonyi`` telescope that covers nights starting 2025-12-10 through 2025-12-12, has tags ``ideal`` and ``nominal``.
Simulations older than 30 days are ignored, such that no visits will be return if the latest simulation is older than this.

Like the ``rubin_sim.sim_archive.vseqarchive.get_visits`` described above, the ``rubin_sim.sim_archive.get_sim_for_nights`` function can take a list of stackers in the ``stackers`` keyword argument.
