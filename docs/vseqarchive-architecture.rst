.. py:currentmodule:: rubin_sim.sim_archive

.. _archive-architecture:

==========================
The Visit Sequence Archive
==========================

Introduction
~~~~~~~~~~~~

An assortment of tools used in Rubin Observatory depend on access to tables describing sequences of visits, including both opsim output and the results of queries to the consdb.
For example:

- The generation of simulations for both the pre-night briefing and other predictions of future scheduler behavior depend on pre-loading the scheduler with visits already completed, for example as queried from the consdb.
- The pre-night briefing report includes figures generated from opsim simulations of the night on which it is reporting.
- ``obsloctap`` provides predictions of visits to be scheduled for observing.
- Progress reports include maf metrics computed from this visit data.

The visit sequence archive is a service for storing and retrieving sequences of visits and ancilliary files associated with sequences of visits (e.g. tables of rewards), tracking metadata describing the sequences of visits, and searching for available sequences of visits based on this metadata.

Top-level components
~~~~~~~~~~~~~~~~~~~~

The visit sequence archive has three major components:

1. A data store that contains the table of visits (as an ``hdf5`` file) and ancilliary data files.
   This is implemented using ``lsst.resources``, and the data store used is set by a base URI relative to which files are stored.
   In production, on S3 bucket is used, while a directory in the local file system is used for testing and demonstrations.
2. A ``postgresql`` database with tables of sequences of visits with metadata on provenance, comments, and statistics (but not the table of visits themselves).
3. A python API and set of shell commands for adding, updating, and querying the archive.

The ``python`` API and shell commands
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``sim_archive`` submodule of ``rubin_sim`` holds the ``python`` API and shell commands that constitude the higher-level interface to the visit sequence archive.
It consists of four sub-sub-modules:

``rubin_sim.sim_archive.vseqarchive``
  contains a collection of functions that support interaction with the archive as a whole, combining interaction with the metadata database and data store where it makes sense to do so.
  For example, ``rubin_sim.sim_archive.vseqarchive.add_file`` combines the addition of a file to the data store and making a record of that file in the metadata database.
  This submodule also defines the ``vsarchive`` ``click.group`` and most functions within it are decorated with ``click`` decorators that place them in this group.
  As a result, these python functions each have corresponding shell commands that take the same arguments.
  (See `the click documentation <https://click.palletsprojects.com/en/stable/>`_ for more details.)
``rubin_sim.sim_archive.vseqmetadata``
  defines the ``VisitSequenceArchiveMetadata`` class, an API that manages queries to the database and provides methods that wrap queries for standard operations.
  In a typical use, ``python`` code will instantiate an instance of ``VisitSequenceArchiveMetadata`` with database connection parameters.
  Then, it can query of modify the database either by calling methods of this class directly, or passing the instance as an argument to functions provided by ``rubin_sim.sim_archive.vsarchive``.
  In corresponding shell commands created using ``click``, the ``click.group`` definition automatically instantiates an instance using command line arguments and passes it to the subcommand.
``rubin_sim.sim_archive.sim_archive``
  provides a handful of functions that replicate functions provided by the prototype implementation.
  For example, the ``obsloctap`` service use the prototype ``fetch_obsloctab_visits`` function to retrieve visits from the best pre-night simultation for a night.
  This submodule therefore implements ``fetch_obsloctap_visits`` here for backwards compatibility.
``rubin_sim.sim_archive.prenightindex``
  provides tools that return inventories of pre-night simulations in the archive.
  Few users have credentials for that allow access to the metadata database, and connections are only possible on a very limited subnet.
  A handful of use cases require broader access, particularly by services running at the observatory.
  These use cases require access to the metadata for only a very limited number of predictable queries, in particular getting inventories of pre-night simulations run for specific nights, and statistics on these simulations.
  The services that need this already need and have access to the data store.
  So, to provide access to the required invertories, the ``prenightindex`` submodule provides tools for querying the matadata database and placing the results in a predictable key in the data store,
  and functions that retrieve the needed data by first attempting to query the metadata database, but fall back on reading the pre-generated results from the data store if necessary.
``rubin_sim.sim_archive.prototype``
  Contains the functions that implemented the prototype data archive.
  These are retained (for now) to provide access to data recorded by the prototype.

The data store
~~~~~~~~~~~~~~

The visit sequerce archive uses the ``lsst.resources`` package to save and retrieve data.
Each visit sequence is indentified by a `UUID <https://www.rfc-editor.org/rfc/rfc9562>`_, and the archive store data at a URI according to a base URI for the data store, the telescope, the visit sequence UUID, the date of creation, and a file name:

.. parsed-literal::
    ${ARCHIVE_URI}/${TELESCOPE}/${CREATION_DATE}/${VISITSEQ_UUID}/${FILENAME}

Where the elements are:

ARCHIVE_URI
  is the base of the archive.
  The default is set to ``s3://rubin:rubin-scheduler-prenight/opsim/vseq/`` by the ``rubin_sim.sim_archive.vseqarchive.ARCHIVE_URL`` module-level variable.
  For testing, it is typically set to a temporary local directory (``file:///some/tmp/dir``) generated by ``python``'s ``tempfile`` standard  library.
TELESCOPE
  designates the relevant telescope, either ``simonyi`` or ``auxtel``
CREATION_DATE
  is the creation date (in the UTC-12 time zone used by `SITCOMTN-032 <https://sitcomtn-032.lsst.io/>`_ for ``dayobs``) of the visit sequence in ISO-8601 (``YYYY-MM-DD``) format.
  In the case of completed visits, this is the date on which the query was made.
  For simulations, it is the date on which the simulation was run.
  When this date is not available, the ``sim_archive`` tools default to the date on which the visit sequence was added to the archive.
FILENAME
  The name of the file in which the data is stored on local disk.

So, a typical URI will look like this:

.. parsed-literal::
    s3://rubin:rubin-scheduler-prenight/opsim/vseq/simonyi/2025-10-16/47ed5c53-ec5a-45a3-bdfe-6b93a3f67bf9/visits.h5

A URL with a ``FILENAME`` of ``visits.h5``, if present, holds the data for visits themselves in `HDF5 format <https://www.hdfgroup.org/solutions/hdf5/>`_, in the ``observations`` key, corresponding to the ``observations`` table in ``sqlite3`` database produced by the ``rubin_scehduler`` simulations.
If the visits originated with the database produced by a ``rubin_scheduler`` simulation, other tables in this database will be saved as tables in corresponding keys in ``visits.h5``.

The archive infrastructure does not limit the keys and file names of other data to be added, but other keys and filenames used can include:

``rewards.h5``
    An HDF5 containing reward data recorded by ``rubin_scheduler`` simulations when called with ``record_rewards=True``.
``opsim.db``
    The ``sqlite3`` file generated by ``rubin_scheduler`` simulations, as written by ``rubin_scheduler``.
    In general, this should be redundant with the ``visits.h5`` file.

The ``postgresql`` metadata database
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tables of sequences of visits
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The central tables in metadata database are those that save metadata on the visit sequences themselves, with one row per visit sequence.
There are three such tables:

``simulations``
  The ``simulations`` table stores metadata on sequences of simulated visits, for example as simulated by ``rubin_scheduler``.
  Visit sequences in these tables should include *only* simulated visits.
  Sequences that are created using a combination of completed and simulated visits, for example a sequence that includes completed visits pre-leaded into the scheduler and then simulated thereafter, should be saved in the ``mexedvisitseq`` table instead.
``completed``
  The ``completed`` table stores metadata on sequences of actually completed visits, for example results of queries to ``consdb``.
``mixed``
  The ``mixed`` table stores metadata in sequences that combine visits from other sequences of visits.
  For example metadat on a set of visits that include completed visits up to some date and simulated visits thereafter would be recorded in the ``mixed`` table.

These tables have the following columns in common:

.. list-table:: visitseq
   :widths: 25 20 20 35
   :header-rows: 1

   * - Column
     - Type
     - Default
     - Description
   * - visitseq_uuid
     - UUID
     - ``gen_random_uuid()``
     - Primary key – RFC 9562 Universally Unique Identifier.
   * - visitseq_sha256
     - BYTEA
     - *None*
     - SHA‑256 hash of bytes of the ``numpy.recarray`` representation of the visits table, as calculated in ``vseqmetadata.compute_visits_sha256``
   * - visitseq_label
     - TEXT
     - *None*
     - Human‑readable label for plots and tables
   * - visitseq_url
     - TEXT
     - *None*
     - URL to the full visit table (NULL if not available)
   * - telescope
     - TEXT
     - *None*
     - Telescope used (e.g. "simonyi", "auxtel")
   * - first_day_obs
     - DATE
     - *None*
     - Date (in the UTC-12 hour timezone) of the first night included in the sequence.
   * - last_day_obs
     - DATE
     - *None*
     - Date (in the UTC-12 hour timezone) of the last night included in the sequence.
   * - creation_time
     - TIMESTAMP WITH TIME ZONE
     - ``NOW()``
     - When the simulation was run or (if not set) when the sequence was added to the archive.

The values in ``first_day_obs`` and ``last_day_obs`` might not correspond to the dates of the first and last visits in the sequence, if the sequence covers dates on which there were no visits.
For example, if an entry in the ``completed`` table were created by querying ``consdb`` for visits between ``2025-10-01`` and ``2025-10-31``, but there no visits in ``consdb`` on ``2025-10-01``, the value of ``first_day_obs`` would still be ``2025-10-01``.
In such a case, a user can interpret such a record as a positive assertion that there were no visits on ``2025-10-01`` fitting the query criteria.

The visit tables for each type include extra columns.

``simulations`` has the following additional columns:

.. list-table::
   :widths: 25 20 55
   :header-rows: 1

   * - Column
     - Type
     - Description
   * - scheduler_version
     - TEXT
     - Version of ``rubin_scheduler`` used
   * - config_url
     - TEXT
     - URL of the configuration script, typically a URL for a specific commit of a specific file in github.
   * - conda_env_sha256
     - BYTEA
     - SHA‑256 hash of the output of ``conda list --json``
   * - parent_visitseq_uuid
     - UUID
     - UUID of the visitseq loaded into the scheduler before running
   * - sim_runner_kwargs
     - JSONB
     - Arguments passed to the simulation runner as a JSON dictionary
   * - parent_last_day_obs
     - DATE
     - Date (in the UTC-12hrs time zone) of the last visit loaded into the scheduler before running

The ``completed`` table has just one column (in addition to those all visit sequence tables have in common):

.. list-table::
   :widths: 25 20 55
   :header-rows: 1

   * - Column
     - Type
     - Description
   * - query
     - TEXT
     - Query used to select visits from ``consdb``

The ``mixed`` table has additional columns describing how the parent visit sequences were combined:

.. list-table::
   :widths: 25 20 55
   :header-rows: 1

   * - Column
     - Type
     - Description
   * - last_early_day_obs
     - DATE
     - The last day_obs drawn from the early parent visit sequence
   * - first_late_day_obs
     - DATE
     - The first day_obs drawn from the late parent visit sequence
   * - early_parent_uuid
     - UUID
     - UUID of the early parent visit sequence
   * - late_parent_uuid
     - UUID
     - UUID of the late parent visit sequence

These three tables are implemented in ``postgresql`` as childen of a single parent table, ``visitseq``.
Therefore, queries of the ``visitseq`` table will include rows from all three of these tables, but only columns they all have in common.

Tags
^^^^

The ``tags`` table associates tags with visit sequences:

.. list-table::
   :widths: 25 20 55
   :header-rows: 1

   * - Column
     - Type
     - Description
   * - visitseq_uuid
     - UUID
     - The visit sequence tagged
   * - tag
     - TEXT
     - The tag

Each row corresponds to a tag applied to a visit sequence.

To query the metadata archive and get a table with one row per visit sequence and lists of tags as a column, use the ``json`` tools in ``postgresql``.
For example:

.. parsed-literal::
  SET SEARCH_PATH TO vsmd;
  SELECT s.visitseq_uuid,
         s.visitseq_label,
         COALESCE (
           JSONB_AGG(DISTINCT t.tag) FILTER (WHERE t.tag IS NOT NULL),
           '[]'::JSONB) AS tags
         FROM simulations AS s
         LEFT JOIN tags AS t ON t.visitseq_uuid=s.visitseq_uuid
         GROUP BY s.visitseq_uuid, visitseq_label;

Reporting tools use tags to identify visit sequences generated to support specific reports.
For example, the ``prenight`` tag identifies simulations made for the pre-night briefing.

Comments
^^^^^^^^

The ``comments`` table associates comments with visit sequences:

.. list-table::
   :widths: 25 25 55
   :header-rows: 1

   * - Column
     - Type
     - Description
   * - visitseq_uuid
     - UUID
     - Identifier of the visit sequence to which the comment belongs
   * - comment_time
     - TIMESTAMP WITH TIME ZONE
     - When the comment was added (defaults to ``NOW()``)
   * - author
     - TEXT
     - User or system that added the comment
   * - comment
     - TEXT
     - The comment text (not nullable)

Files
^^^^^

The ``files`` table associates URIs of files with file types and visit sequences.

.. list-table::
   :widths: 25 20 55
   :header-rows: 1

   * - Column
     - Type
     - Description
   * - visitseq_uuid
     - UUID
     - Identifier of the visit sequence that the file belongs to
   * - file_type
     - TEXT
     - The type of file (e.g., ``rewards``)
   * - file_sha256
     - BYTEA
     - SHA‑256 hash of the file contents
   * - file_url
     - TEXT
     - URL where the file can be retrieved; may be ``NULL`` if only the hash is stored

Note that the ``visits`` ``file_type`` is special, and stored in the corresponding visits sequence table itself rather than in this ``files`` table.

``conda`` environments
^^^^^^^^^^^^^^^^^^^^^^

The ``simulations`` table records the hash of the specifications for the conda environment (as reported by ``conda list --json``) in which the simulations was run.
By itself, this record allows a user to identify which simulations were made with the same environment, but not what that environment was.
The ``conda_env`` table records the actual content of the ``conda list --json`` output, in a format that can be use with ``postgresql``'s json tools.

.. list-table::
   :widths: 25 20 55
   :header-rows: 1

   * - Column
     - Type
     - Description
   * - conda_env_hash
     - BYTEA
     - Primary key – SHA‑256 hash of the output of ``conda list --json``
   * - conda_env
     - JSONB
     - Full JSON representation of the conda environment (``conda list --json`` output)

The ``conda_packages`` view supports querying this table as if each package were stored in its own row of a table.
For example, to get the ``astropy`` versions for all simulations for which the conda environment is recorded:

.. parsed-literal::
  SET SEARCH_PATH TO vsmd;
  SELECT creation_time, visitseq_uuid, package_version AS astropy_version FROM simulations NATURAL JOIN conda_packages WHERE package_name='astropy';

Nightly statistics
^^^^^^^^^^^^^^^^^^

The nightly_stats table can records basic statistics by night for any value for which each visit has an associated value.
Examples can be columns in the visits table referenced by ``visitseq_url``, but may also be derived quentities such as those produced by ``maf`` stackers.

.. list-table::
   :widths: 25 25 55
   :header-rows: 1

   * - Column
     - Type
     - Description
   * - visitseq_uuid
     - UUID
     - Identifier of the visit sequence
   * - day_obs
     - DATE
     - The date (in the UTC-12hrs timezone, following SITCOMTN-032) of the night
   * - value_name
     - TEXT
     - Name of the metric or column being summarized
   * - accumulated
     - BOOLEAN
     - ``TRUE`` if the values include all data through *day_obs*,
       ``FALSE`` if only the data from *day_obs* itself
   * - count
     - INTEGER
     - Number of values in the distribution
   * - mean
     - DOUBLE PRECISION
     - Arithmetic mean of the values
   * - std
     - DOUBLE PRECISION
     - Standard deviation of the values
   * - min
     - DOUBLE PRECISION
     - Minimum value
   * - p05
     - DOUBLE PRECISION
     - 5% quantile
   * - q1
     - DOUBLE PRECISION
     - First quartile (25% quantile)
   * - median
     - DOUBLE PRECISION
     - Median (50% quantile)
   * - q3
     - DOUBLE PRECISION
     - Third quartile (75% quantile)
   * - p95
     - DOUBLE PRECISION
     - 95% quantile
   * - max
     - DOUBLE PRECISION
     - Maximum value


``maf`` results
^^^^^^^^^^^^^^^

Additional tables exist for possible future support of saving ``maf`` summary metrics in the visit sequence metadata database.
There are currently no tools to support their use.

These tables are:

``maf_metrics``
  records parameters used to run metrics.
  Columns are ``maf_metric_name``, ``rubin_sim_version``, ``maf_constraint``, ``metric_class_name``, ``metric_args``, ``slicer_class_name``, ``slicer_args``
``maf_summary_metrics``
  records the values of summary metrics themselves for a given visit sequence.
  Columns are ``visitseq_uuid``, ``maf_metric_name``, ``day_obs``, ``accumulated``, ``summary_value``.
  The combination of the ``day_obs`` and ``accumulated`` columns support recording values from either visits only on (if ``accumulated`` is ``false``) a specific night (``day_obs``),
  or all visits (if ``accumulated`` is ``true``) up to and including a specific night (``day_obs``).
``maf_metric_sets``
  defines sets of metrics, following the use of such sets in ``rubin_sim.maf.run_comparison``.
``maf_summary``
  is a view that makes it easy to get everything for the summary metrics for one metric set applied to runs with specified tags.
``maf_healpix_stats``
  supports recording of statistics of metric values when the metrics return healpix arrays.
