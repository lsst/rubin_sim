CREATE SCHEMA {};

SET SEARCH_PATH TO {};

CREATE TABLE visitseq (
    visitseq_uuid   UUID PRIMARY KEY DEFAULT gen_random_uuid(),   -- RFC 4122 Universally Unique IDentifier
    visitseq_sha256 BYTEA NOT NULL,     -- hash of the visit table
    visitseq_label  TEXT NOT NULL,      -- label for plots and table
    visitseq_url    TEXT,               -- If null, the actual visits are not available
    telescope       TEXT NOT NULL,      -- (probably) "simonyi" or "auxtel"
    first_day_obs   DATE,               -- local calendar date of evening of first night in the set
    last_day_obs    DATE,                -- local calendar date of the eveninig of the last night in the set
    creation_time   TIMESTAMP WITH TIME ZONE DEFAULT NOW() -- when the simulation was run
);

CREATE TABLE simulations (
    scheduler_version   TEXT,                               -- version of rubin_scheduler
    config_url          TEXT,                               -- URL for the config script, perhaps on github based on hash
    conda_env_sha256    BYTEA,                              -- SHA256 hash of output of conda list --json
    parent_visitseq_uuid    UUID,                           -- UUID of visitseq loaded into scheduler before running
    sim_runner_kwargs   JSONB,                              -- arguments to sim runner as a json dict
    parent_last_day_obs DATE,                               -- day_obs of last visit loaded into scheduler before running
    PRIMARY KEY (visitseq_uuid)
) INHERITS (visitseq);

CREATE TABLE completed (
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
-- we will most often want to search for all runs with a given tag.
CREATE INDEX tag_idx ON tags(tag);

CREATE TABLE comments (
    visitseq_uuid   UUID,                                   -- RFC 4122 Universally Unique IDentifier
    comment_time    TIMESTAMP WITH TIME ZONE DEFAULT NOW() ,  -- The date and time at which the comment was added
    author          TEXT,                                   -- The user who added the comment
    comment         TEXT NOT NULL                           -- The comment
);
-- we will most often want to search for all comments for a given visitseq
CREATE INDEX comments_visitseq_idx ON comments(visitseq_uuid);

CREATE TABLE files (
    visitseq_uuid   UUID,                                   -- RFC 4122 Universally Unique IDentifier
    file_type       TEXT NOT NULL,                          -- Examples include "rewards", "scheduler", and "observatory"
    file_sha256     BYTEA NOT NULL,                         -- hash of the file
    file_url        TEXT,                                   -- URL for the file, may be null if we wish only to store the hash
    PRIMARY KEY (visitseq_uuid, file_type)
);
CREATE INDEX files_visitseq_idx ON files(visitseq_uuid);

-- A view to make it easy to query simulations and get tags, comments, and files back.
CREATE VIEW simulations_extra AS
    SELECT
        DATE(s.creation_time - INTERVAL '12 hours') AS sim_creation_day_obs,
        ROW_NUMBER() OVER (PARTITION BY DATE(s.creation_time - INTERVAL '12 hours') ORDER BY s.creation_time - INTERVAL '12 hours') AS daily_id,
        s.visitseq_uuid,
        s.visitseq_label,
        s.visitseq_url,
        s.telescope,
        s.first_day_obs,
        s.last_day_obs,
        s.creation_time,
        s.scheduler_version,
        s.config_url,
        s.sim_runner_kwargs,
        s.conda_env_sha256,
        s.parent_visitseq_uuid,
        s.parent_last_day_obs,
        COALESCE (
            JSONB_AGG(DISTINCT t.tag) FILTER (WHERE t.tag IS NOT NULL),
            '[]'::JSONB
        ) AS tags,
        COALESCE (
             JSONB_OBJECT_AGG(c.comment_time, c.comment) FILTER (WHERE c.comment_time IS NOT NULL),
             '{{}}'::JSONB
        ) AS comments,
        COALESCE (
             JSONB_OBJECT_AGG(f.file_type, f.file_url) FILTER (WHERE f.file_type IS NOT NULL),
             '{{}}'::JSONB
        ) AS files
    FROM simulations AS s
    LEFT JOIN tags AS t ON t.visitseq_uuid=s.visitseq_uuid
    LEFT JOIN comments AS c ON c.visitseq_uuid=s.visitseq_uuid
    LEFT JOIN files AS f ON f.visitseq_uuid=s.visitseq_uuid
    GROUP BY s.visitseq_uuid,
        s.visitseq_label,
        s.visitseq_url,
        s.telescope,
        s.first_day_obs,
        s.last_day_obs,
        s.creation_time,
        s.scheduler_version,
        s.config_url,
        s.sim_runner_kwargs,
        s.conda_env_sha256,
        s.parent_visitseq_uuid,
        s.parent_last_day_obs;

CREATE TABLE conda_env (
    conda_env_hash  BYTEA PRIMARY KEY,                      -- SHA256 hash of output of `conda list --json`
    conda_env       JSONB NOT NULL                          -- output of `conda list --json`
);

-- Make a view so we do not need to remember
-- postgresql syntex for jsonb.
CREATE VIEW conda_packages AS (
    SELECT
        c.conda_env_hash,
        p.name AS package_name,
        p.channel,
        p.version AS package_version,
        p.base_url,
        p.platform,
        p.dist_name,
        p.build_number,
        p.build_string
    FROM
        conda_env AS c,
        jsonb_to_recordset(conda_env) AS p (
            name TEXT,
            channel TEXT,
            version TEXT,
            base_url TEXT,
            platform TEXT,
            dist_name TEXT,
            build_number TEXT,
            build_string TEXT
        )
);

CREATE VIEW simulation_packages AS (
    SELECT s.visitseq_uuid, p.*
    FROM conda_packages AS p
    JOIN simulations AS s ON p.conda_env_hash = s.conda_env_sha256
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
CREATE TABLE maf_metrics (
    maf_metric_name     TEXT PRIMARY KEY,                   -- name for this maf metric
    rubin_sim_version   TEXT,                               -- the version of rubin_sim used
    maf_constraint      TEXT,                               -- constraint imposed in maf
    metric_class_name   TEXT,                               -- class name of the metric
    metric_args         JSONB,                              -- arguments to the metric constructor
    slicer_class_name   TEXT,                               -- class name of the slicer
    slicer_args         JSONB                               -- arguments to the slicer constructor
);
CREATE INDEX maf_metrics_name_idx ON maf_metrics(maf_metric_name);

-- Support functionality like that of rubin_sim.maf.run_comparison run summaries,
-- but sampled at whatever dates are of interest rather than just the end.
-- To get it into the "wide" format like that in run_comparison, use either
-- a tablefunc in the postgresql query, or (probably easier) use unstack on
-- the returned dataframe.
CREATE TABLE maf_summary_metrics (
    visitseq_uuid   UUID,                                   -- RFC 4122 Universally Unique IDentifier
    maf_metric_name   TEXT REFERENCES maf_metrics(maf_metric_name), -- The name of the maf metric
    day_obs         DATE,                                   -- The day obs of the visits included
    accumulated     BOOLEAN,                                -- Whether the statistics include all data through night day_obs, or only data on night day_obs
    summary_value   DOUBLE PRECISION,
    PRIMARY KEY (visitseq_uuid, maf_metric_name)
);
CREATE INDEX maf_summary_metrics_name_idx ON maf_summary_metrics(maf_metric_name);

-- metric sets for rubin_sim.maf.run_comparison
CREATE TABLE maf_metric_sets (
    metric_set      TEXT,                                   -- name of the metric set
    maf_metric_name   TEXT REFERENCES maf_metrics(maf_metric_name), -- name of the maf metric
    short_name      TEXT,                                   -- a shorter label to use in plots
    style           TEXT,                                   -- matplotlib line style
    invert          BOOLEAN,                                -- negative is lower is better
    mag             BOOLEAN,                                 -- value is a magnitude
    PRIMARY KEY (metric_set, maf_metric_name)
);
CREATE INDEX maf_metric_sets_metric_name_idx ON maf_metric_sets(maf_metric_name);
CREATE INDEX maf_metric_sets_set_idx ON maf_metric_sets(metric_set);

-- Create a view to make it easy to get everything for the summary metrics for one metric set
-- applied to runs with specified tags.
CREATE VIEW maf_summary AS
    SELECT mms.metric_set, mms.maf_metric_name, v.visitseq_label,
        msm.day_obs, msm.accumulated, msm.summary_value,
        mms.short_name, mms.style, mms.invert, mms.mag, t.tag
    FROM maf_metric_sets AS mms
    JOIN maf_summary_metrics AS msm ON msm.maf_metric_name=mms.maf_metric_name
    JOIN tags AS t ON t.visitseq_uuid=msm.visitseq_uuid
    JOIN visitseq AS v ON msm.visitseq_uuid=v.visitseq_uuid
    ;


CREATE TABLE maf_healpix_stats (
    visitseq_uuid   UUID NOT NULL,                          -- RFC 4122 Universally Unique IDentifier
    maf_metric_name   TEXT REFERENCES maf_metrics(maf_metric_name), -- The name of the maf statistic
    day_obs         DATE NOT NULL,                          -- The day obs of the visits included
    accumulated     BOOLEAN NOT NULL,                       -- Whether the statistics include all data through night day_obs, or only data on night day_obs
    nside           INTEGER NOT NULL,                       -- The nside of the healpix array
    count           INTEGER NOT NULL,                       -- The number of unmasked values in the distribution
    mean            DOUBLE PRECISION,                       -- mean
    std             DOUBLE PRECISION,                       -- standard deviation
    min             DOUBLE PRECISION,                       -- min value
    p05             DOUBLE PRECISION,                       -- 5% quantile
    q1              DOUBLE PRECISION,                       -- first quartile
    median          DOUBLE PRECISION,                       -- median
    q3              DOUBLE PRECISION,                       -- third quartile
    p95             DOUBLE PRECISION,                       -- 95% quantile
    max             DOUBLE PRECISION,                       -- maximum
    url             TEXT,                                   -- URL for the array of healpix values, if saved
    PRIMARY KEY (visitseq_uuid, maf_metric_name, day_obs, accumulated, nside)
);
