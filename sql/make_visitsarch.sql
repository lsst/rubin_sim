CREATE SCHEMA visitsarch;

SET SEARCH_PATH TO visitsarch;

CREATE TABLE visitseq (
    visitseq_uuid   UUID PRIMARY KEY,   -- RFC 4122 Universally Unique IDentifier
    visitseq_sha256 BYTEA NOT NULL,     -- hash of the visit table
    label           TEXT NOT NULL,      -- label for plots and table
    visiseq_url     TEXT,               -- If null, the actual visits are not available
    telescope       TEXT NOT NULL,      -- (probably) "simonyi" or "auxtel"
    first_day_obs   DATE,               -- local calendar date of evening of first night in the set
    last_day_obs    DATE                -- local calendar date of the eveninig of the last night in the set
);

CREATE TABLE simulations (
    creation_time       TIMESTAMP WITH TIME ZONE NOT NULL,  -- when the simulation was run
    scheduler_version   TEXT,                               -- version of rubin_scheduler
    config_url          TEXT,                               -- URL for the config script, perhaps on github based on hash
    conda_env_sha256    BYTEA,                              -- SHA256 hash of output of conda list --json
    parent_visitseq_uuid    UUID,                           -- UUID of visitseq loaded into scheduler before running
    parent_last_day_obs DATE,                               -- day_obs of last visit loaded into scheduler before running
    PRIMARY KEY (visitseq_uuid)
) INHERITS (visitseq);

CREATE TABLE completed (
    creation_time   TIMESTAMP WITH TIME ZONE,               -- when the consdb was queried
    query           TEXT,                                   -- The query to the consdb used
    PRIMARY KEY (visitseq_uuid)
) INHERITS (visitseq);

CREATE TABLE mixedvisitseq (
    last_early_day_obs  DATE,                               -- the last day obs drawn from the early parent
    first_late_day_obs  DATE,                               -- the first day obs drawn from the late parent
    early_parent_uuid   UUID,                               -- the UUID of the early parent
    late_parent_uuid    UUID,                               -- the UUID of the late parent
     PRIMARY KEY (visitseq_uuid)
) INHERITS (visitseq);

-- All of the tables that follow include a visitseq_uuid that refereneces
-- visitseq or one of its children.
-- These are not declared as foreign keys, because postgresql cannot enforce
-- a FK on both a table and its children: we would need to specify which child
-- the uuid would refer to, when really we need it to be allowed to reference
-- any of the children.
-- (In principle, we could do something complicated with triggers.)

CREATE TABLE tags (
    visitseq_uuid   UUID,                                   -- RFC 4122 Universally Unique IDentifier
    tag             TEXT NOT NULL                           -- The tag
);

CREATE TABLE comments (
    visitseq_uuid   UUID,                                   -- RFC 4122 Universally Unique IDentifier
    comment_time    TIMESTAMP WITH TIME ZONE,               -- The date and time at which the comment was added
    author          TEXT,                                   -- The user who added the comment
    comment         TEXT NOT NULL                           -- The comment
);

CREATE TABLE files (
    visitseq_uuid   UUID,                                   -- RFC 4122 Universally Unique IDentifier
    file_type       TEXT NOT NULL,                          -- Examples include "rewards", "scheduler", and "observatory"
    file_sha256     BYTEA NOT NULL,                         -- hash of the file
    file_url        TEXT,                                   -- URL for the file, may be null if we wish only to store the hash
    PRIMARY KEY (visitseq_uuid, file_type)
);

CREATE TABLE conda_env (
    conda_env_hash  BYTEA PRIMARY KEY,                      -- SHA256 hash of output of `conda list --json`
    conda_env       JSONB NOT NULL                          -- output of `conda list --json`
);

CREATE TABLE nightly_stats (
    visitseq_uuid   UUID NOT NULL,                          -- RFC 4122 Universally Unique IDentifier
    day_obs         DATE NOT NULL,                          -- The day obs of the visits included
    value_name      TEXT NOT NULL,                          -- metric or column name
    accumulated     BOOLEAN NOT NULL,                       -- Whether the statistics include all data through night day_obs, or only data on night day_obs
    count           INTEGER,                                -- number of values in distribution
    mean            DOUBLE PRECISION,                       -- mean
    std             DOUBLE PRECISION,                       -- standard deviation
    min             DOUBLE PRECISION,                       -- min value
    p05             DOUBLE PRECISION,                       -- 5% quantile
    q1              DOUBLE PRECISION,                       -- first quartile
    median          DOUBLE PRECISION,                       -- median
    q3              DOUBLE PRECISION,                       -- third quartile
    p95             DOUBLE PRECISION,                       -- 95% quantile
    max             DOUBLE PRECISION,                       -- maximum
    PRIMARY KEY (visitseq_uuid, day_obs, value_name, accumulated)
);

-- Tables to support functionality like that of rubin_sim.maf.run_comparison

-- Use tags to mark visit sequences as being runs that are part of a given run family.

-- Table to keep track of exactly what metrics were run in maf.
CREATE TABLE maf_stats (
    maf_stat_name       TEXT PRIMARY KEY,                   -- name for this maf stat
    rubin_sim_version   TEXT,                               -- the version of rubin_sim used
    maf_constraint      TEXT,                               -- constraint imposed in maf
    metric_class_name   TEXT,                               -- class name of the metric
    metric_args         JSONB,                              -- arguments to the metric constructor
    slicer_class_name   TEXT,                               -- class name of the slicer
    slicer_args         JSONB                               -- arguments to the slicer constructor
);

-- Support functionality like that of rubin_sim.maf.run_comparison run summaries,
-- but sampled at whatever dates are of interest rather than just the end.
-- To get it into the "wide" format like that in run_comparison, use either
-- a tablefunc in the postgresql query, or (probably easier) use unstack on
-- the returned dataframe.
CREATE TABLE maf_summary_stats (
    visitseq_uuid   UUID,                                   -- RFC 4122 Universally Unique IDentifier
    maf_stat_name   TEXT REFERENCES maf_stats(maf_stat_name), -- The name of the maf statistic
    day_obs         DATE,                                   -- The day obs of the visits included
    accumulated     BOOLEAN,                                -- Whether the statistics include all data through night day_obs, or only data on night day_obs
    summary_value   DOUBLE PRECISION,
    PRIMARY KEY (visitseq_uuid, maf_stat_name)
);

-- metric sets for rubin_sim.maf.run_comparison
CREATE TABLE maf_metric_sets (
    metric_set      TEXT,                                   -- name of the metric set
    maf_stat_name   TEXT REFERENCES maf_stats(maf_stat_name), -- name of the maf stat
    short_name      TEXT,                                   -- a shorter label to use in plots
    style           TEXT,                                   -- matplotlib line style
    invert          BOOLEAN,                                -- negative is lower is better
    mag             BOOLEAN,                                 -- value is a magnitude
    PRIMARY KEY (metric_set, maf_stat_name)
);


CREATE TABLE maf_healpix_stats (
    visitseq_uuid   UUID NOT NULL,                          -- RFC 4122 Universally Unique IDentifier
    maf_stat_name   TEXT REFERENCES maf_stats(maf_stat_name), -- The name of the maf statistic
    day_obs         DATE NOT NULL,                          -- The day obs of the visits included
    accumulated     BOOLEAN NOT NULL,                       -- Whether the statistics include all data through night day_obs, or only data on night day_obs
    nside           INTEGER NOT NULL,                       -- The nside of the healpix array
    mean            DOUBLE PRECISION,                       -- mean
    std             DOUBLE PRECISION,                       -- standard deviation
    min             DOUBLE PRECISION,                       -- min value
    p05             DOUBLE PRECISION,                       -- 5% quantile
    q1              DOUBLE PRECISION,                       -- first quartile
    median          DOUBLE PRECISION,                       -- median
    q3              DOUBLE PRECISION,                       -- third quartile
    p95             DOUBLE PRECISION,                       -- 95% quantile
    max             DOUBLE PRECISION,                       -- maximum
    url             TEXT,                                   -- URL for the array of helpix values, if saved
    PRIMARY KEY (visitseq_uuid, maf_stat_name, day_obs, accumulated, nside)
);
