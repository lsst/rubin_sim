.. py:currentmodule:: rubin_sim.sim_archive

.. vseqarchive-setup:


Setting up and administering the visit sequence metadata database
=================================================================

Introduction
------------

This page documents the procedure followed to create the visit sequence archive metadata database.
**It does not provide instructions that are expected to be repeated again,** but are rather intended as documentation of provenance of the database.
The code blocks listed here were exported from a jupyter notebook used for the initial setup.

Setting up the python kernel
----------------------------

The setup of the database began by importing necessary python modules:

.. code:: ipython3

    from pathlib import Path
    from psycopg2 import sql
    from rubin_sim.sim_archive import vseqarchive

Look at what’s in the database
------------------------------

I then checked what schema are already there:

.. code:: bash

    %%bash
    psql --host 134.79.23.205 --username rubin --command "\dn" opsim_log


.. parsed-literal::

          List of schemas
      Name  |       Owner
    --------+-------------------
     public | pg_database_owner
    (1 row)

Note the host used for the database may have changed since the initial setup: the current IP address for the database may be different.


Create a test schema
--------------------

Create an interface to the visit sequence archive metadata database.
Start by setting connection parameters to a user and host with permissions.
The ``opsim_log`` database and ``rubin`` user are those set up by the SLAC computing administrators, not by anyone on the scheduling team.

.. code:: ipython3

    metadata_db_kwargs = {
        'database': 'opsim_log',
        'host': '134.79.23.205',
        'user': 'rubin'
    }

Make an instance of the interface.
Set the schema to ``test``, because (for safety reasons) the interface only allows creation of schema with ``test`` in the name.

.. code:: ipython3

    vsarchive = vseqarchive.VisitSequenceArchiveMetadata(
        metadata_db_kwargs,
        metadata_db_schema='test'
    )

Run the method that creates an instance of the schema (with name
``test``):

.. code:: ipython3

    vsarchive.create_schema_in_database()


.. parsed-literal::

    Created test database and schema  test


Check that it was created:

.. code:: bash

    %%bash
    psql --host 134.79.23.205 --username rubin --command "\dn" opsim_log


.. parsed-literal::

          List of schemas
      Name  |       Owner
    --------+-------------------
     public | pg_database_owner
     test   | rubin
    (2 rows)



Rename our newly created ``test`` schema to the production schema name, so it becomes our production schema:

.. code:: bash

    %%bash
    psql --host 134.79.23.205 --username rubin --command "ALTER SCHEMA test RENAME TO vsmd" opsim_log


.. parsed-literal::

    ALTER SCHEMA


Check that it did what we wanted:

.. code:: bash

    %%bash
    psql --host 134.79.23.205 --username rubin --command "\dn" opsim_log


.. parsed-literal::

          List of schemas
      Name  |       Owner
    --------+-------------------
     public | pg_database_owner
     vsmd   | rubin
    (2 rows)



.. code:: bash

    %%bash
    psql --host 134.79.23.205 --username rubin --command "SELECT table_schema, table_name, table_type FROM information_schema.tables WHERE table_schema = 'vsmd';" opsim_log



.. parsed-literal::

     table_schema |     table_name      | table_type
    --------------+---------------------+------------
     vsmd         | visitseq            | BASE TABLE
     vsmd         | simulations         | BASE TABLE
     vsmd         | completed           | BASE TABLE
     vsmd         | mixedvisitseq       | BASE TABLE
     vsmd         | tags                | BASE TABLE
     vsmd         | comments            | BASE TABLE
     vsmd         | files               | BASE TABLE
     vsmd         | simulations_extra   | VIEW
     vsmd         | conda_env           | BASE TABLE
     vsmd         | conda_packages      | VIEW
     vsmd         | simulation_packages | VIEW
     vsmd         | nightly_stats       | BASE TABLE
     vsmd         | maf_summary_metrics | BASE TABLE
     vsmd         | maf_metrics         | BASE TABLE
     vsmd         | maf_metric_sets     | BASE TABLE
     vsmd         | maf_summary         | VIEW
     vsmd         | maf_healpix_stats   | BASE TABLE
    (17 rows)



I also want an actual schema named ``test``, so make it:

.. code:: ipython3

    vsarchive.create_schema_in_database()


.. parsed-literal::

    Created test database and schema  test


.. code:: bash

    %%bash
    psql --host 134.79.23.205 --username rubin --command "\dn" opsim_log


.. parsed-literal::

          List of schemas
      Name  |       Owner
    --------+-------------------
     public | pg_database_owner
     test   | rubin
     vsmd   | rubin
    (3 rows)



Creating roles and giving them permissions
------------------------------------------

Create three roles with login permissions initially.
Note that, in postgresql, a "user" is just a role with login permissions.

- ``reader`` will be a shared account for read-only access.
- ``writer``` will be a used by the pre-night simulation process to add pre-night simulations to the database.
- ``tester``` will be used for testing.

Use the ``\password`` ``psql`` command to avoid the password being recorded in the ``psql`` history.

.. parsed-literal::

    CREATE USER reader;
    \password reader
    CREATE USER writer;
    \password writer
    CREATE USER tester;
    \password tester

Create groups for the users:


.. parsed-literal::

    CREATE GROUP readers WITH USER reader;
    CREATE GROUP testers WITH USER tester;
    CREATE GROUP writers WITH USER writer;

Give the groups permissions:

.. parsed-literal::

    GRANT SELECT ON ALL TABLES IN SCHEMA vsmd TO readers, writers;
    GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA test TO testers;
    GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA vsmd TO writers;
    GRANT USAGE ON SCHEMA vsmd TO readers, writers;
    GRANT USAGE ON SCHEMA test TO testers;
