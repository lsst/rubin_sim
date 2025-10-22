import datetime
import hashlib
import logging
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Sequence, Tuple
from uuid import UUID

import click
import pandas as pd
from astropy.time import Time
from lsst.resources import ResourcePath

from rubin_sim.maf.stackers import BaseStacker

from .util import compute_conda_env, dayobs_to_date, hdf5_to_opsimdb, opsimdb_to_hdf5
from .vseqmetadata import VSARCHIVE_PGSCHEMA, VisitSequenceArchiveMetadata

__all__ = [
    "compute_nightly_stats",
    "get_visits",
    "record_visitseq_metadata",
    "set_visitseq_url",
    "get_visitseq_url",
    "update_visitseq_metadata",
    "is_tagged",
    "tag",
    "untag",
    "comment",
    "get_comments",
    "archive_file",
    "get_file",
    "add_nightly_stats",
    "query_nightly_stats",
    "record_conda_env",
    "import_proto",
    "export_proto",
    "ARCHIVE_URL",
    "PRENIGHT_INDEX_URL",
]

ARCHIVE_URL = "s3://rubin:rubin-scheduler-prenight/opsim/vseq/"
PRENIGHT_INDEX_URL = "s3://rubin:rubin-scheduler-prenight/opsim/prenight_index/"
SQLITE_EXTINSIONS = {".db", ".sqlite", ".sqlite3", ".db3"}

LOGGER = logging.getLogger(__name__)

#
# Computation
#


def compute_nightly_stats(
    visits: pd.DataFrame, columns: Tuple[str, ...] = ("altitude", "azimuth")
) -> pd.DataFrame:
    if "day_obs" not in visits.columns:
        # Pandas seems to work better with type hinding that astropy Time
        start_mjd_field = ""
        for candidate_field in ["obs_start_mjd", "observationStartMJD"]:
            if candidate_field in visits.columns:
                start_mjd_field = candidate_field
                break
        if len(start_mjd_field) == 0:
            raise ValueError("No day_obs or start mjd field found")

        times = pd.to_datetime(visits.loc[:, start_mjd_field] + 2400000, unit="D", origin="julian")
        visits = visits.assign(day_obs=pd.Series(times).dt.date)

    stats_df = (
        visits.groupby("day_obs")[list(columns)]
        .describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
        .stack(level=0)
        .reset_index()
        .rename(
            columns={
                "level_1": "value_name",
                "5%": "p05",
                "25%": "q1",
                "50%": "median",
                "75%": "q3",
                "95%": "p95",
            }
        )
    )
    stats_df["accumulated"] = False
    return stats_df


def construct_base_resource_path(
    archive_base: ResourcePath, telescope: str, creation_time: Time, visitseq_uuid: UUID
) -> ResourcePath:
    """Build the base ``ResourcePath`` for a visit‑sequence
    archive entry.

    Parameters
    ----------
    archive_base : `ResourcePath`
        The base of the archive relative to which the visit sequence
        will be added.
    telescope : `str`
        Identifier for the telescope (e.g. ``"simonyi"`` or ``"auxtel"`).
    creation_time : `Time`
        The timestamp when the visit sequence was created.
    visitseq_uuid : `UUID`
        The unique identifier of the visit sequence.

    Returns
    -------
    ResourcePath
        A ``ResourcePath`` instance representing the base directory for
        this visit sequence.  The directory is structured as:

        ``<archive_base>/<telescope>/<date>/<visitseq_uuid>/``

        where ``<date>`` is the UTC‑12 date derived from ``creation_time``.
    """
    creation_time_iso_dayobs = dayobs_to_date(Time(creation_time.mjd, format="mjd")).isoformat()
    visitseq_base_rp = (
        archive_base.join(telescope)
        .join(creation_time_iso_dayobs)
        .join(str(visitseq_uuid), forceDirectory=True)
    )
    return visitseq_base_rp


#
# Interaction with the file store
#


def _write_data_to_archive(content: bytes, visitseq_base_rp: ResourcePath, file_name: str) -> ResourcePath:
    # Make sure the base for the visit sequence exists
    if not visitseq_base_rp.exists():
        visitseq_base_rp.mkdir()
        LOGGER.debug(f"Created {visitseq_base_rp}.")

    destination_rp = visitseq_base_rp.join(file_name)
    LOGGER.debug(f"Writing {destination_rp}.")
    destination_rp.write(content)
    LOGGER.debug(f"Wrote {destination_rp}.")
    return destination_rp


def _write_file_to_archive(
    origin: str | Path,
    visitseq_base_rp: ResourcePath,
) -> Tuple[ResourcePath, bytes]:
    file_path = origin if isinstance(origin, Path) else Path(origin)
    with open(file_path, "rb") as origin_io:
        content = origin_io.read()

    content_sha256 = bytes.fromhex(hashlib.sha256(content).hexdigest())
    destination_rp = _write_data_to_archive(content, visitseq_base_rp, file_path.name)
    return destination_rp, content_sha256


def add_file(
    vsarch: VisitSequenceArchiveMetadata,
    uuid: UUID,
    origin: str | Path,
    file_type: str,
    archive_base: str | ResourcePath,
    update: bool = False,
) -> ResourcePath:
    """Archive a file associated with a visit sequence and
    register its location in the metadata database.

    Parameters
    ----------
    vsarch : `VisitSequenceArchiveMetadata`
        Instance of `VisitSequenceArchiveMetadata` used to
        query the sequence metadata and register the file.
    uuid : `UUID`
        The unique identifier of the visit sequence that the file
        should be linked to.
    origin : `str` or `pathlib.Path`
        Path to the local file to be archived.
    file_type : `str`
        Identifier for the type of file being archived.
    archive_base : `lsst.resources.ResourcePath`
        Base location of the archive.
    update : `bool`, optional
        If ``True`` and a record for the same ``visitseq_uuid`` and
        ``file_type`` already exists, the existing row will be
        updated with the new SHA‑256 hash and URL.  If ``False`` a
        `ValueError` will be raised on duplicate.

    Returns
    -------
    file_rp : `lsst.resources.ResourcePath`
        The `ResourcePath` pointing to the archived file
        on the file store.
    """
    if isinstance(archive_base, str):
        archive_base = ResourcePath(archive_base)
    assert isinstance(archive_base, ResourcePath)

    visitseq_metadata = vsarch.get_visitseq_metadata(uuid)
    uuid = visitseq_metadata["visitseq_uuid"]
    telescope = visitseq_metadata["telescope"]
    creation_time = Time(visitseq_metadata["creation_time"])
    visitseq_base_rp = construct_base_resource_path(archive_base, telescope, creation_time, uuid)

    # If we are adding a file that looks like opsim sqlite3 output but
    # specify the file type as visits, convert to hdf5.
    if file_type == "visits" and Path(origin).suffix.lower() in SQLITE_EXTINSIONS:
        with TemporaryDirectory() as temp_dir:
            new_origin = Path(temp_dir).joinpath("visits.h5")
            opsimdb_to_hdf5(origin, new_origin)
            location = add_file(vsarch, uuid, new_origin, "visits", archive_base, update)
        return location

    location, file_sha256 = _write_file_to_archive(origin, visitseq_base_rp)
    if file_type == "visits":
        vsarch.set_visitseq_url(uuid, location.geturl())
    else:
        vsarch.register_file(uuid, file_type, file_sha256, location, update=update)

    return location


def get_visits(
    visits_url: str | ResourcePath,
    query: str = "",
    stackers: Sequence[BaseStacker] = (),
) -> pd.DataFrame:
    """Retrieve visits for a visit sequence.

    Parameters
    ----------
    visits_url : `str`
        The origin of the visits.
    query : `str`
        A query in `pandas.DataFrame.query` syntax to select visits
        from the returned sequence. Applied before stackers are run.
    stackers : `Sequence` [`BaseStacker`]
        A sequence of maf stackers to apply.

    Returns
    -------
    visits: `pd.DataFrame`
        The visits in the sequence.
    """
    origin_rp = ResourcePath(visits_url)

    with TemporaryDirectory() as temp_dir:
        h5_destination = ResourcePath(temp_dir).join("visits.h5")
        h5_destination.transfer_from(origin_rp, "copy")
        visits = pd.read_hdf(h5_destination.ospath, key="observations")

    assert isinstance(visits, pd.DataFrame)

    if len(query) > 0:
        visits.query(query, inplace=True)

    if len(stackers) > 0:
        visit_records = visits.to_records()
        for stacker in stackers:
            visit_records = stacker.run(visit_records)
        visits = pd.DataFrame(visit_records)

    return visits


#
# API
#


@click.group()
@click.option(
    "--database",
    default=None,
    help="PostgreSQL database name of the metadata database",
)
@click.option(
    "--host",
    default=None,
    help="PostgreSQL host address of the metadata database",
)
@click.option(
    "--user",
    default=None,
    help="PostgreSQL user name to use to connect to the metadata database",
)
@click.option(
    "--schema",
    default=None,
    help="Schema of the metadata database containing the visit‑sequence tables",
)
@click.pass_context
def vseqarchive(
    click_context: click.Context,
    database: str,
    host: str,
    user: str | None,
    schema: str,
) -> None:
    """visitseq command line interface."""
    metadata_db_kwargs = {}
    if database is not None:
        metadata_db_kwargs["database"] = database

    if user is not None:
        metadata_db_kwargs["user"] = user

    if host is not None:
        metadata_db_kwargs["host"] = host

    if schema is None:
        if "VSARCHIVE_PGSCHEMA" in os.environ:
            schema = os.environ["VSARCHIVE_PGSCHEMA"]
        else:
            schema = VSARCHIVE_PGSCHEMA
    assert isinstance(schema, str)

    click_context.obj = VisitSequenceArchiveMetadata(metadata_db_kwargs, schema)


@vseqarchive.command()
@click.argument("table", type=str)
@click.argument("visits_file", type=click.Path(exists=True))
@click.argument("label")
@click.option(
    "--telescope",
    default="simonyi",
    show_default=True,
    help="Telescope identifier (e.g. simonyi or auxtel)",
)
@click.option("--url", default=None, help="Optional URL for the visits file")
@click.option("--first_day_obs", default=None, help="First night of observations (YYYY-MM-DD or int)")
@click.option("--last_day_obs", default=None, help="Last night of observations (YYYY-MM-DD or int)")
@click.option(
    "--creation_time",
    default=None,
    help="Creation time of the visit sequence in ISO format (e.g. 2025-01-01T00:00:00)",
)
@click.pass_obj
def record_visitseq_metadata(
    vsarch: VisitSequenceArchiveMetadata,
    table: str,
    visits_file: str,
    label: str,
    telescope: str,
    url: str | None,
    first_day_obs: str | int | None,
    last_day_obs: str | int | None,
    creation_time: str | None,
) -> None:
    """Record visit‑sequence metadata from an HDF5 visits file.

    Parameters
    ----------
    vsarch : `VisitSequenceArchiveMetadata`
        The database interface instance.
    table : `str`
        Target table for the record (e.g. ``visitseq``, ``simulations``,
        ``completed`` or ``mixedvisitseq``).
    visits_file : `str`
        Path to the HDF5 file (or SQLite3 file) containing the
        ``visits`` dataset.
    label : `str`
        Human‑readable label for the visit sequence.
    telescope : `str`, optional
        Telescope identifier, default ``simonyi``.
    url : `str`, optional
        URL where the visits file can be downloaded.
    first_day_obs : `str` or `int`, optional
        First day of observation in SMTN‑032 format, ISO string, or
        ``YYYY‑MM‑DD``.
    last_day_obs : `str` or `int`, optional
        Last day of observation (same formats as ``first_day_obs``).
    creation_time : `str`, optional
        ISO‑formatted timestamp of when the sequence was created;
        if omitted, the current time is used.

    Returns
    -------
    None
        The UUID of the inserted visit sequence is printed to stdout.
    """

    if Path(visits_file).suffix.lower() in SQLITE_EXTINSIONS:
        with TemporaryDirectory() as temp_dir:
            hdf5_path = Path(temp_dir).joinpath("visits.h5")
            opsimdb_to_hdf5(visits_file, hdf5_path)
            visits_df = pd.read_hdf(str(hdf5_path), key="observations")
    else:
        visits_df = pd.read_hdf(visits_file, key="observations")
    assert isinstance(visits_df, pd.DataFrame)

    # Convert the optional creation time string to an astropy Time object
    creation_ap_time = Time(creation_time) if creation_time is not None else Time.now()

    # Record the visit sequence metadata
    visitseq_uuid = vsarch.record_visitseq_metadata(
        visits_df,
        label,
        telescope=telescope,
        table=table,
        url=url,
        first_day_obs=first_day_obs,
        last_day_obs=last_day_obs,
        creation_time=creation_ap_time,
    )
    print(visitseq_uuid)


@vseqarchive.command()
@click.argument("uuid", type=click.UUID)
@click.option(
    "--table",
    default="visitseq",
    show_default=True,
    help="Table to query (e.g. visitseq, simulations, completed, mixedvisitseq)",
)
@click.pass_obj
def get_visitseq_metadata(vsarch: VisitSequenceArchiveMetadata, uuid: UUID, table: str) -> None:
    """Retrieve and display metadata for a visit sequence.

    Parameters
    ----------
    vsarch : `VisitSequenceArchiveMetadata`
        The metadata interface instance.
    uuid : `UUID`
        The UUID of the visit sequence to query.
    table : `str`, optional
        Name of the child table to search; one of ``visitseq``,
        ``simulations``, ``completed`` or ``mixedvisitseq``.
        Defaults to ``visitseq``.
    """
    sequence_metadata = vsarch.get_visitseq_metadata(uuid, table=table)

    # Print the DataFrame as a tab‑separated table
    print(sequence_metadata.to_frame().T.to_csv(sep="\t", index=False).rstrip("\n"))


@vseqarchive.command()
@click.argument("uuid", type=click.UUID)
@click.argument("url", type=click.STRING)
@click.pass_obj
def set_visitseq_url(vsarch: VisitSequenceArchiveMetadata, uuid: UUID, url: str) -> None:
    """Update the stored URL for a visit‑sequence file.

    Parameters
    ----------
    vsarch : `VisitSequenceArchiveMetadata`
        The metadata interface instance.
    uuid : `UUID`
        The UUID of the visit sequence whose URL is to be updated.
    url : `str`
        The new URL pointing to the visits file.
    """
    vsarch.set_visitseq_url(uuid, url)


@vseqarchive.command()
@click.argument("uuid", type=click.UUID)
@click.pass_obj
def get_visitseq_url(vsarch: VisitSequenceArchiveMetadata, uuid: UUID) -> None:
    """Retrieve and print the URL for the visits in a visit sequence.

    Parameters
    ----------
    vsarch : `VisitSequenceArchiveMetadata`
        The metadata interface instance.
    uuid : `UUID`
        The UUID of the visit sequence whose URL is to be retrieved.
    """
    url = vsarch.get_visitseq_url(uuid)
    click.echo(url)


@vseqarchive.command()
@click.argument("uuid", type=click.UUID)
@click.argument("field", type=str)
@click.argument("value", type=str)
@click.pass_obj
def update_visitseq_metadata(
    vsarch: VisitSequenceArchiveMetadata,
    uuid: UUID,
    field: str,
    value: str,
) -> None:
    """Update a single metadata field for a visit sequence.

    Parameters
    ----------
    vsarch : `VisiteSequenceArchiveMetadata`
        An instance of the interface to the metadata archive.
    uuid : `UUID`
        The UUID of the visit sequence to update.
    field : `str`
        The name of the column in the metadata table to modify.
    value : `str`
        The new value for the field.  Postgresql will cast the type
        to what is expected in the database schema.
    """
    # psycopg2 usually converts strings to the right
    # type base on column type in the database, but if
    # hex values are provided for bytes colums it
    # gets it wrong.
    # So, if we're asking to update a bytes column, convert
    # it from hex ourselves.
    update_value: str | bytes = value
    if field.endswith("sha256"):
        update_value = bytes.fromhex(value)

    vsarch.update_visitseq_metadata(uuid, field, update_value)


@vseqarchive.command()
@click.argument("uuid", type=click.UUID)
@click.argument("tag", type=str)
@click.pass_obj
def is_tagged(vsarch: VisitSequenceArchiveMetadata, uuid: UUID, tag: str) -> None:
    """Return whether a visit sequence is tagged with a given tag.

    The command prints ``true`` if the tag exists for the visit sequence
    and ``false`` otherwise.

    Parameters
    ----------
    vsarch : `VisitSequenceArchiveMetadata`
        The interface to the metadata database.
    uuid : `UUID`
        The UUID of the visit sequence to check.
    tag : `str`
        The tag name to query.
    """
    seq_is_tagged = vsarch.is_tagged(uuid, tag)
    if seq_is_tagged:
        click.echo("true")
    else:
        click.echo("false")


@vseqarchive.command()
@click.argument("uuid", type=click.UUID)
@click.argument("tags", nargs=-1, required=True)
@click.pass_obj
def tag(vsarch: VisitSequenceArchiveMetadata, uuid: UUID, tags: Tuple[str, ...]) -> None:
    """Add one or more tags to a visit sequence.

    Parameters
    ----------
    vsarch : `VisitSequenceArchiveMetadata`
        The interface to the metadata database.
    uuid : `UUID`
        The UUID of the visit sequence to tag.
    tags : `str`
        One or more tag names to add.
    """
    if len(tags) < 1:
        raise ValueError("At least one tag must be requested.")

    vsarch.tag(uuid, *tags)


@vseqarchive.command()
@click.argument("uuid", type=click.UUID)
@click.argument("tag", type=str)
@click.pass_obj
def untag(vsarch: VisitSequenceArchiveMetadata, uuid: UUID, tag: str) -> None:
    """Remove a tag from a visit sequence.

    Parameters
    ----------
    vsarch : `VisitSequenceArchiveMetadata`
        The interface to the metadata database.
    uuid : `UUID`
        The UUID of the visit sequence to modify.
    tag : `str`
        The name of the tag to remove.
    """
    vsarch.untag(uuid, tag)


@vseqarchive.command()
@click.argument("uuid", type=click.UUID)
@click.argument("comment")
@click.option(
    "--author",
    default=None,
    help="The author of the comment",
)
@click.pass_obj
def comment(
    vsarch: VisitSequenceArchiveMetadata, uuid: UUID, comment: str, author: str | None = None
) -> None:
    """Attach a comment to a visit sequence in the metadata database.

    Parameters
    ----------
    vsarch : `VisitSequenceArchiveMetadata`
        The interface to the metadata database.
    uuid : `UUID`
        The UUID of the visit sequence to comment on.
    comment : `str`
        The comment text to attach.
    author : `str`, optional
        The author of the comment. If omitted the comment will be recorded
        without an author field.
    """
    vsarch.comment(uuid, comment, author)


@vseqarchive.command()
@click.argument("uuid", type=click.UUID)
@click.pass_obj
def get_comments(vsarch: VisitSequenceArchiveMetadata, uuid: UUID) -> None:
    """Retrieve all comments attached to a specific visit sequence.

    Parameters
    ----------
    uuid : `UUID`
        The UUID of the visit sequence whose comments should be fetched.
    """
    comments_df = vsarch.get_comments(uuid)
    if not comments_df.empty:
        # Pandas will include the index by default; omit it.
        click.echo(comments_df.to_csv(sep="\t", index=False).rstrip("\n"))


@vseqarchive.command()
@click.argument("uuid", type=click.UUID)
@click.argument("origin", type=click.Path(exists=True))
@click.argument("file_type", type=click.STRING)
@click.option(
    "--archive-base",
    default=ARCHIVE_URL,
    show_default=True,
    help="Base directory for the archive (e.g. file://data/archive).",
)
@click.option(
    "--update",
    is_flag=True,
    help="If set, overwrite an existing record for the same visitseq_uuid and file_type.",
)
@click.pass_obj
def archive_file(
    vsarch: VisitSequenceArchiveMetadata,
    uuid: UUID,
    origin: str,
    file_type: str,
    archive_base: str,
    update: bool,
) -> None:
    """Archive a file and register its location in the metadata database.

    Parameters
    ----------
    vsarch : `VisitSequenceArchiveMetadata`
        The metadata interface instance.
    uuid : `UUID`
        The UUID of the visit sequence that the file should be linked to.
    origin : `str`
        Path to the local file to be archived.
    file_type : `str`
        Identifier for the type of file being archived.
    archive_base : `str`, optional
        Base location of the archive.
    update : `bool`, optional
        If true, an existing record for the same
        ``visitseq_uuid``/``file_type`` combination is updated rather
        than raising an error.
    """
    # Convert the base path string into a ResourcePath
    archive_base_rp = ResourcePath(archive_base)

    archived_location = add_file(vsarch, uuid, origin, file_type, archive_base_rp, update=update)

    click.echo(archived_location.geturl())


@vseqarchive.command()
@click.argument("uuid", type=click.UUID)
@click.argument("file_type", type=click.STRING)
@click.argument("destination", type=click.Path(exists=False))
@click.pass_obj
def get_file(
    vsarch: VisitSequenceArchiveMetadata,
    uuid: UUID,
    file_type: str,
    destination: str,
) -> None:
    """Retrieve a registered file and copy it to a local destination.

    Parameters
    ----------
    vsarch : `VisitSequenceArchiveMetadata`
        The metadata interface used to look up the file URL.
    uuid : `UUID`
        The unique identifier of the visit sequence containing the
        requested file.
    file_type : `str`
        The handle for the type of file to retrieve (e.g. ``visits``).
    destination : `str`
        Path where the retrieved file should be written. If the
        destination does not exist it will be created.  For files
        registered with the ``visits`` type and a SQLite3 extension
        suffix, the file will be downloaded as an HDF5 ``visits.h5`` file,
        converted back to the original opsim SQLite format, and written to
        ``destination``.
    """
    file_url = vsarch.get_file_url(uuid, file_type)
    origin_rp = ResourcePath(file_url)
    if file_type == "visits" and Path(destination).suffix.lower() in SQLITE_EXTINSIONS:
        with TemporaryDirectory() as temp_dir:
            h5_destination = ResourcePath(temp_dir).join("visits.h5")
            h5_destination.transfer_from(origin_rp, "copy")
            click.echo(f"Copied {origin_rp.geturl()} to {h5_destination.geturl()}")
            hdf5_to_opsimdb(h5_destination.ospath, destination)
            click.echo(f"Converted to opsim and wrote to {destination}")
    else:
        destination_rp = ResourcePath(destination)
        destination_rp.transfer_from(origin_rp, "copy")
        click.echo(f"Copied {origin_rp.geturl()} to {destination_rp.geturl()}")


@vseqarchive.command()
@click.argument("uuid", type=click.UUID)
@click.argument("visits_file", type=click.Path(exists=True))
@click.argument(
    "columns",
    type=str,
    required=False,
    nargs=-1,
)
@click.pass_obj
def add_nightly_stats(
    vsarch: VisitSequenceArchiveMetadata,
    uuid: UUID,
    visits_file: str,
    columns: tuple[str, ...],
) -> None:
    """Compute nightly statistics for a visit sequence and add
    them to the visit sequence metadata database.

    Parameters
    ----------
    vsarch : `VisitSequenceArchiveMetadata`
        The metadata interface.
    uuid : `UUID`
        The UUID of the visit sequence to attach the statistics to.
    visits_file : `str`
        Path to an HDF5 file containing a ``visits`` dataset.
    columns : `tuple`
        List of columns for which stats should be added.
    """
    # Load the visits table from the HDF5 file
    visits_df = pd.read_hdf(visits_file, key="observations")
    assert isinstance(visits_df, pd.DataFrame), "Expected a DataFrame from key 'observations'"

    # Compute the nightly statistics, passing the requested columns if any
    if columns:
        stats_df = compute_nightly_stats(visits_df, columns=tuple(columns))
    else:
        stats_df = compute_nightly_stats(visits_df)

    # Insert into the database
    vsarch.insert_nightly_stats(uuid, stats_df)

    # Print the statistics as TSV
    tsv = stats_df.to_csv(sep="\t", index=False).rstrip("\n")
    click.echo(tsv)


@vseqarchive.command()
@click.argument("uuid", type=click.UUID)
@click.pass_obj
def query_nightly_stats(vsarch: VisitSequenceArchiveMetadata, uuid: UUID) -> None:
    """Retrieve nightly statistics for a visit sequence and
    print them as a tab‑separated table.

    Parameters
    ----------
    vsarch : `VisitSequenceArchiveMetadata`
        The metadata interface.
    uuid : `UUID`
        The unique identifier of the visit sequence.
    """
    stats_df = vsarch.query_nightly_stats(uuid)
    output = stats_df.to_csv(sep="\t", index=False).rstrip("\n") if not stats_df.empty else ""
    click.echo(output)


@vseqarchive.command()
@click.pass_obj
def record_conda_env(vsarch: VisitSequenceArchiveMetadata) -> None:
    """Record the current Conda environment in the metadata database,
    and print the hash used as a key for the env in the database.

    Parameters
    ----------
    vsarch : `VisitSequenceArchiveMetadata`
        The metadata interface.
    """
    env_hash, env_json = compute_conda_env()
    vsarch.record_conda_env(env_hash, env_json)
    click.echo(env_hash.hex())


@vseqarchive.command("import-proto")
@click.argument("archive_base", type=str)
@click.argument("sim_date", type=str)
@click.argument("sim_index", type=str)
@click.option(
    "--proto-sim-archive-url",
    default="s3://rubin:rubin-scheduler-prenight/opsim/",
    show_default=True,
    help="Base URI of the prototype simulation archive.",
)
@click.pass_obj
def import_proto(
    vsarch: VisitSequenceArchiveMetadata,
    archive_base: str,
    sim_date: str,
    sim_index: str,
    proto_sim_archive_url: str = "s3://rubin:rubin-scheduler-prenight/opsim/",
) -> None:
    """Import a simulation from the prototype simulation archive,
    and print the UUID of the resultant entry in the destintation
    archive.

    Parameters
    ----------
    vsarch : `VisitSequenceArchiveMetadata`
        The metadata interface instance.
    archive_base :`str`
        Base location of the destination archive.
    sim_date : `str`
        The ISO‑formatted date (e.g. ``'2025-03-12'``) by
        which the prototype simulation is indexed.
    sim_index : `str`
        The index of the simulation which completes the
        identification of the simulation in the prototype.
    proto_sim_archive_url : `str`, optional
        Base URI of the prototype simulation archive.  Defaults to
        ``s3://rubin:rubin-scheduler-prenight/opsim/``.
    """
    sim_uuid = vsarch.import_sim_from_prototype_sim_archive(
        archive_base,
        sim_date,
        sim_index,
        proto_sim_archive_url,
    )
    click.echo(sim_uuid)


@vseqarchive.command("export-proto")
@click.argument("uuid", type=click.UUID)
@click.option(
    "--proto-sim-archive-url",
    default="s3://rubin:rubin-scheduler-prenight/opsim/",
    show_default=True,
    help="Base URL of the prototype simulation archive to which to export.",
)
@click.pass_obj
def export_proto(
    vsarch: VisitSequenceArchiveMetadata,
    uuid: UUID,
    proto_sim_archive_url: str = "s3://rubin:rubin-scheduler-prenight/opsim/",
) -> None:
    """Export a simulation from the visit sequence archive
    to the prototype simulation archive, and print the
    URL in the prototype sim archive.

    Parameters
    ----------
    vsarch : `VisitSequenceArchiveMetadata`
        The metadata interface instance.
    uuid : `UUID`
        UUID of the simulation to export.
    proto_sim_archive_url : `str`
        Base URL of the prototype simulation archive to which the simulation
        should be exported.
    """
    # Import here to avoid circular imports
    from .prototype import export_sim_to_prototype_sim_archive

    destination_rp = export_sim_to_prototype_sim_archive(
        archive_metadata=vsarch,
        sim_uuid=uuid,
        proto_sim_archive_url=proto_sim_archive_url,
    )
    click.echo(destination_rp.geturl())


@vseqarchive.command()
@click.argument("dayobs", type=click.DateTime(formats=["%Y-%m-%d", "%Y%m%d"]))
@click.argument("telescope", type=click.Choice(["simonyi", "auxtel"]))
@click.option(
    "--destination",
    default=PRENIGHT_INDEX_URL,
    type=str,
    help="Base of the resource to which to write the prenight index.",
)
@click.pass_obj
def make_prenight_index(
    vsarch: VisitSequenceArchiveMetadata,
    dayobs: str | datetime.datetime | datetime.date | int | Time,
    telescope: str,
    destination: str,
) -> None:
    """Write a json index of prenight for a telescope and dayobs to a resource.

    Parameters
    ----------
    vsarch : `VisitSequenceArchiveMetadata`
        The metadata interface instance.
    dayobs : `datetime` or `date` or `int` or `str`
        A datetime in the desired dayobs
    telescope : `str`
        The telescope, either ``simonyi`` or ``auxtel``
    destination : `str`
        Base of the resource to which to write the prenight index.
    """
    dayobs = dayobs_to_date(dayobs)
    assert isinstance(dayobs, datetime.date)

    prenights = vsarch.sims_on_night_with_stats(
        dayobs, tags=("prenight",), telescope=telescope, max_simulation_age=4000
    ).set_index("visitseq_uuid")

    def to_hex(data: memoryview | None) -> str | None:
        if data is None:
            return None

        assert isinstance(data, memoryview)
        return data.hex()

    prenights["conda_env_sha256"] = prenights["conda_env_sha256"].apply(to_hex)

    def to_str_dayobs(d: datetime.date | None) -> str | None:
        if d is None:
            return None
        return d.isoformat()

    for column in ("sim_creation_day_obs", "first_day_obs", "last_day_obs", "parent_last_day_obs"):
        prenights[column] = prenights[column].apply(to_str_dayobs)

    # Convert UUID type into something to_json can deal with.
    prenights.index = prenights.index.map(str)
    for row in prenights.index:
        parent_visitseq_uuid = prenights.loc[row, "parent_visitseq_uuid"]
        if isinstance(parent_visitseq_uuid, UUID):
            prenights.loc[row, "parent_visitseq_uuid"] = str(parent_visitseq_uuid)

    table_json = prenights.to_json(orient="index", date_format="iso", indent=4)

    destination_dir_rp = (
        ResourcePath(destination)
        .join(telescope)
        .join(str(dayobs.year))
        .join(str(dayobs.month), forceDirectory=True)
    )
    destination_rp = destination_dir_rp.join(f"{telescope}_prenights_for_{dayobs.isoformat()}.json")
    destination_rp.write(table_json.encode())
    click.echo(destination_rp.geturl())


if __name__ == "__main__":
    vseqarchive()
