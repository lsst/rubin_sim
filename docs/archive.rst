.. py:currentmodule:: rubin_sim.sim_archive

.. _archive:

==========================
The Visit Sequence Archive
==========================

Introduction
------------

An assortment of tools used in Rubin Observatory depend on access to tables describing sequences of visits, including both opsim output and the results of queries to the consdb.
For example:

- The generation of simulations for both the pre-night briefing and other predictions of future scheduler behavior depend on pre-loading the scheduler with visits already completed, for example as queried from the consdb.
- The pre-night briefing report includes figures generated from opsim simulations of the night on which it is reporting.
- ``obsloctap`` provides predictions of visits to be scheduled for observing.
- Progress reports include maf metrics computed from this visit data.

The visit sequence archive is a service for storing and retrieving sequences of visits and ancilliary files associated with sequences of visits (e.g. tables of rewards), tracking metadata describing the sequences of visits, and searching for available sequences of visits based on this metadata.

Prototype
---------

An initial prototype (described in :doc:`protoarchive`) used for pre-night simulations and reports saved both data and metadata in an S3 bucket.
This prototype saved the visits themselves as ``sqlite3`` database files as produced by ``rubin_scheduler``, and metadata in ``yaml`` files for each sequence.
A python function provided by the prototype configured simulations, executed them (by calling ``rubin_scheduler.scheduler.sim_runner``), wrote metadata to a yaml file, and added of results (including the opsim database output, metadata file, and ancilliary files) to the S3 bucket. and wrapped this in a command line call to run the simulation from with a shell script submitter as a batch job.
Other functions supported searching metadata in the yaml files for pre-night simulations for a given night and retrieving the corresponding visit tables and other files.

The prototype had two significant problems:

1. Saving metadata for each simulation in its own yaml file meant that searching for simulations according to metadata required retrieving the metadata yaml files for all simulations in the archive from the S3 bucket, which is not scalable. This was partially addressed through creation of an index file in the same S3 bucket which combined metadata from all simulations up to some date, but this was just a stop-gap measure.
2. The bundling of insertion of data into the archive with driving the execution of the simulation made the archive itself inflexible, making it awkward to modify either how the simulation were driven or data were archived separately.

The visit sequence archive addresses these concerns by:

1. Keeping the metadata in a separate ``postgresql`` database.
2. Separating the archiving API from the simulation driver: commands that add data to the archive are independent of how the visit data are generated.

Architecture
------------

Top-level components
~~~~~~~~~~~~~~~~~~~~

The visit sequence archive has three major components:

1. A data store that contains the table of visits (as an ``hdf5`` file) and ancilliary data files.
   This is implemented using ``lsst.resources``, and the data store used is set by a base URI relative to which files are stored.
   In production, on S3 bucket is used, while a directory in the local file system is used for testing and demonstrations.
3. A ``postgresql`` database with tables of sequences of visits with metadata on provenance, comments, and statistics (but not the table of visits themselves).
4. A python API and set of shell commands for adding, updating, and querying the archive.

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
  The default is set to ``s3://rubin:rubin-scheduler-prenight/opsim/vseq/`` by the `rubin_sim.sim_archive.vseqarchive.ARCHIVE_URL` module-level variable.
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

A URL with a ``FILENAME`` of ``visits.h5``, if present, holds the data for visits themselves in `HDF5 format <https://www.hdfgroup.org/solutions/hdf5/>`_, in the ``observations`` key, corresponding to the ``observations`` table in ``sqlite3`` database produced by the ``rubin_scehduler`` simulations.
If the visits originated with the database produced by a ``rubin_scheduler`` simulation, other tables in this database will be saved as tables in corresponding keys in ``visits.h5``.

The archive infrastructure does not limit the keys and file names of other data to be added, but other keys and filenames used can include:

``rewards.h5``
    An HDF5 containing reward data recorded by ``rubin_scheduler`` simulations when called with ``record_rewards=True``.
``opsim.db``
    The ``sqlite3`` file generated by ``rubin_schedelur`` simulations, as written by ``rubin_scheduler``.
    In general, this should be redundant with the ``visits.h5`` file.

The ``postgresql`` metadata database
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
