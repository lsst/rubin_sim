.. py:currentmodule:: rubin_sim.sim_archive

.. _archive:

========================================================
The Prototype OpSim Archive for ``schedview`` Dashboards
========================================================

Introduction
------------

Several tools will require an archive that provides access to simulations provided by ``rubin_scheduler``.
For example, the prenight briefing dashboard supplied by ``schedview`` is a tool for visualizing simualtions of a night of observing, and it requires access to such simulations.
There will eventually be other dashboards that will also need to read such tables for visualization, as well as tables describing visits completed by the actual instruments.
Users of these dashboards will need to use them to select which data sets are to be visualised: the dashboard code needs to provide both user interface elements that let the user select the desired table of visits, and also actually load the table itself.
The dashboard servers run within containers on kubernetes-based infrastructure, which operate best when persistent data is stored outside the containers themselves.
Therefore, these dashboards require an external (to the container) resource that supports searching for available simulations, and access to the data itself.

This archive design is intended primarily as a prototype, something to experiment with for a better informed development of requiriments.
So, flexibility and speed of implementation have been prioritized, with the intention that there be a significant rofactoring (or even outright replacement) when requirements have been more thoroughly developed.

Design
------

The archive itself is a directory tree.
The URI of an archive is the URI of the root of the directory tree as supported by the ``lsst.resources`` package.
Each simulation in the archive has its own directory in this directory tree, named according to the ISO-8601 date on which a simulation was added to the archive, and a simple incrementing index separating different simulations added on the same day, such that the format for the directory for a specific simulation is::

    ${ARCHIVE_URI}/{ISO_DATE}/{ADDITION_INDEX}

For example, if the URI of the archive is::

    file:///my_data/sim_archive

then the URI of the third simulation added on June 21, 2030 will be::

    file:///my_data/sim_archive/2030-06-21/3

Each simulation directory contains a metadata yaml file named ``sim_metadata.yaml``.
In the above example, the URI for this metadata file would be::

    file:///my_data/sim_archive/2030-06-21/3/sim_metadata.yaml

A minimal ``sim_metadata.yaml`` file specifies the name of the sqlite3 database file with the visits.
For example, if the URI for the visit database in the above example is ``file:///my_data/sim_archive/2030-06-21/3/opsim.db``, then the minimal content of ``sim_metadata.yaml`` would be::

    files:
      observations:
        name: 'opsim.db'

All other data in the metadata file is optional, but additional metadata will be required if the archived simulation is to be used for some use cases.
For example, if ``schedview``'s ``prenight`` dashboard is to be able to load the reward data, it must be able to locate the reward data from the metadata file, so that the metadata file needs to look something like this::

    files:
      observations:
        name: 'opsim.db'
      rewards:
        name: 'rewards.h5'

Clients of the archive will also need to search available simulations for those meeting relevant criteria.
For example, the ``prenight`` dashboard will seach for simulations the include a desired night, in which case the range of nights covered by the simulation must be included.

A sample metadata file that includes an early guess at what the ``prenight`` dashboard will use looks like this::

    files:
        observations:
            name: opsim.db
        rewards:
            name: rewards.h5
    label: Notebook test on 2024-01-04 16:49:44.299
    simulated_dates:
        first: '2025-05-05'
        last: '2025-05-05'

In the above:

``label``
  Simulations will appear in drop-down section widgets in dashdoards such as the pre-night dashboard.
  The ``label`` element in the determines how the simulation will appear in the dropdown.
  In other applications, this element may also be used as plot annotations or column or row headings.

``simulation_dates``
  Shows the range of dates covered by the simulation.
  When the user specifies a night, the ``prenight`` dashboard will restrict the offered to those that cover the specified date.


Finally, a number of other elements may be included for debugging purposes.
A full file might look something like this::

    files:
        environment:
            md5: 4381d7cc82049141c70216121e39f56d
            name: environment.txt
        notebook:
            md5: 6b75c1dd8c4a3b83797c873f3270cc04
            name: notebook.ipynb
        observations:
            md5: 1909d1afaf744ee50bdcf1a9625826ab
            name: opsim.db
        pypi:
            md5: 9c86ea9b4e7aa40d3e206fad1a59ea31
            name: pypi.json
        rewards:
            md5: 6d3c9d3e0dd7764ed60312e459586e1b
            name: rewards.h5
        scheduler:
            md5: 5e88dfee657e6283dbc7a343f048db92
            name: scheduler.pickle.xz
        statistics:
            md5: c515ba27d83bdbfa9e65cdefff2d9d75
            name: obs_stats.txt
    label: Notebook test on 2024-01-04 16:49:44.299
    simulated_dates:
        first: '2025-05-05'
        last: '2025-05-05'
    scheduler_version: 1.0.1.dev25+gba1ca4d.d20240102
    sim_runner_kwargs:
        mjd_start: 60800.9565967191
        record_rewards: true
        survey_length: 0.5155218997970223
    tags:
    - notebook
    - devel
    host: neilsen-nb
    username: neilsen

This example has a number of additional elements useful for debugging, and which pehaps might be useful for future applictions, but which are not used (or planned to be used) by the prenight dashboard.

``files/*``
  A number of other types of files associated with specific simulations may be included.
  These may be useful in future applications, or for debugging only.
  See below for descriptions of the extra types of files in this example.
``files/${TYPE}/md5``
  Checksums for various files.
  These can be useful both for checking for corruption, and for determining whether two simulations are identical without needing to download either.
``scheduler_version``
  The version of the scheduler used to produce the simualtions.
``sim_runner_kwargs``
  The arguments to the execution of ``sim_runner`` used to run the simulation.
``tags``
  A list of ad-hoc keywords.
  For example, simulations used to test a specific jira issue may all have the name of the issue as a keyword.
  Simulations used to support a give tech note may have the name of the tech note.
``host``
  The hostname on which the simulation was run.
``username``
  The username of the user who ran the simulation.

Optional (for debugging or speculative future uses only) file types listed above are:

``environment``
  The conda environment specification for the environment used to run the simulation.
``notebook``
  The notebook used to create the simulation, for example as created using the ``%notebook`` jupyter magic.
``pypy``
  The ``pypy`` package list of the environment used to run the simulation.
  If the simulation is run using only conda-installed packages, this will be redundant with ``environment``.
``scheduler``
  A python pickle of the scheduler, in the state as of the start of the simulation.
``statistics``
  Basic statistics for the visit database.

Metadata cache
--------------

Reading each ``sim_metadata.yaml`` individually when loading metadata for a large number of simulations can be slow.
Therefore, metadata for sets of simulations can be compiled into a ``compiled_metadata_cache.h5`` file.
This file stores four tables in `hdf5` format: ``simulations``, ``files``, ``kwargs``, and ``tags``.
Each of these tables is indexed by the URI of a simulation.

The ``files`` table contains one column for each key in the ``files`` dictionary in the yaml metadata file for the simulation, providing the metadata needed to reconstruct this element of the dictionary.

The ``kwargs`` table contains one column for each key in the ``sim_runner_kwargs`` dictionary in the yaml metadata file for the simulation, providing the metadata needed to reconstruct this element of the dictionary.
If a keyword argument is not set, an `numpy.nan` value is stored in the table.

The ``tags`` table contains one column: ``tag``, and contains one row for each tag in each simulation.

The ``simulations`` table contains one column for every other keyword found in the metadata yaml files.
If a keyword argument is not set, an `numpy.nan` value is stored in the table.

The ``compile_sim_archive_metadata_resource`` command in ``rubin_sim`` maintains the ``compiled_metadata_cache.h5`` file in an archive.
By default, it reads every ``sim_metadata.yaml`` file in the archive and builds a corresponding cache hdf5 file from scratch.
If called with an ``--append`` flag, it reads an existing metadata cache file, reads ``sim_metadata.yaml`` files for simulations more recently added than the last file in the existing cache, appends them to the previous results from the cache, and writes the result to the cache.
The ``append`` flag therefore speeds up the update considerably, but does not update the cache for any changes to previously added simulations (including deletions).

The ``compile_sim_archive_metadata_resource`` needs to be run to update the cache.
Normall, a cron job will execute this command routinely to keep the cache reasonably up to date.
Because the tools read the metadata yaml files for any simulations added after the most recent cache update, it will function correctly even if the cache is out of date (but slower).

Automatic archiving of generated data
-------------------------------------

The ``rubin_sim`` package provides a tool to combine running a simulation and adding the results to an archive, including any metadata that can be derived automatically.
The ``rubin_sim.sim_archive.drive_sim`` function is wrapper around ``rubin_sim.scheduler.sim_runner`` that incorporates this metadata collection and the creation of the entry in an archive.
It takes all of the same arguments that ``sim_runner`` does, and passes them directly to ``sim_runner``.
In addition, it takes a few arguments that specify the archive into which it is to be added (``archive_uri``), the label to be included in the metadata (``label``), and the code used to run the simulation (ethier ``script`` or ``notebook``).
Details are available in the ``drive_sim`` docstring.

For example, if this code is put into a file and run as a script, it will run the specificed simulation and add it to the specified archive::

  from astropy.time import Time

  from rubin_sim.scheduler.example import example_scheduler
  from rubin_sim.scheduler.model_observatory import ModelObservatory
  from rubin_sim.sim_archive import drive_sim

  sim_mjd_start = Time("2025-05-05").mjd + 0.5
  # The start date of the simualtion.
  # Offset by 0.5 to avoid starting late when the MJD rollover occurs during or
  # after twilight. See dayObs in SITCOMTN-32: https://sitcomtn-032.lsst.io/ .

  sim_length = 1.0
  # Passed to sum_runner, in units of days.

  archive_uri = "file:///sdf/data/rubin/user/neilsen/data/test_sim_archive/"
  # The URI of the root of the archive. The trailing "/" is required.

  observatory = ModelObservatory()
  scheduler = example_scheduler()
  scheduler.keep_rewards = True

  results = drive_sim(
      observatory=observatory,
      scheduler=scheduler,
      archive_uri=archive_uri,
      label=f"Example simulation started at {Time.now().iso}.",
      script=__file__,
      tags=["example"],
      mjd_start=sim_mjd_start,
      survey_length=sim_length,
      record_rewards=True,
  )

The result looks like this::

  bash$ ls /sdf/data/rubin/user/neilsen/data/test_sim_archive/2024-01-18/1
  environment.txt  example_archived_sim_driver.py  obs_stats.txt  opsim.db  pypi.json  rewards.h5  scheduler.pickle.xz  sim_metadata.yaml
  bash$ cat /sdf/data/rubin/user/neilsen/data/test_sim_archive/2024-01-18/1/sim_metadata.yaml
  files:
      environment:
          md5: 33f94ddf8975f9641a1f524fd22e362e
          name: environment.txt
      observations:
          md5: 8b1ee9a604a88d2708d2bfd924ac3cd9
          name: opsim.db
      pypi:
          md5: 51a8deee5018f59f20d5741fd1a64778
          name: pypi.json
      rewards:
          md5: 10e4ab9397382bfa108fa21354da3526
          name: rewards.h5
      scheduler:
          md5: 35713860dc9ba7a425500f63939d0e02
          name: scheduler.pickle.xz
      script:
          md5: b4a476a4fd1231ea1ca44149784f1c3f
          name: example_archived_sim_driver.py
      statistics:
          md5: 7c6a6af38aff3ce4145146e35f929b47
          name: obs_stats.txt
  host: sdfrome002.sdf.slac.stanford.edu
  label: Example simulation started at 2024-01-18 15:46:27.758.
  scheduler_version: 1.0.1.dev25+gba1ca4d.d20240102
  sim_runner_kwargs:
      mjd_start: 60800.5
      record_rewards: true
      survey_length: 1.0
  simulated_dates:
      first: '2025-05-04'
      last: '2025-05-04'
  tags:
  - example
  username: neilsen

Alternately, the simulation can be run from a ``jupyter`` notebook similarly, excepet that instead of saving the script that generated the simulation, a notebook with the cells of notebook that created the simulation up to the cell that runs the simulation can be stored instead.
An example can be found in the ``archive/sim_and_archive.ipynb`` in the `rubin_sim_notebook github repository <https://github.com/lsst/rubin_sim_notebooks>`_.
