"""This module provides a subclass of the testing.postgresql.Postgresql
class that modifies it to use a local-only connection through a unix
socket instead of access through an IP address.

It is included in the main rubin_sim.sim_archive module instead of the
testing directory to make access from tutorial jupyter notebooks easier.
"""

import getpass

import testing.postgresql


class LocalOnlyPostgresql(testing.postgresql.Postgresql):
    """Manager for creation and cleanup of temporary postgresql databases
    configured to provide access only over a local UNIX socket.
    """

    DEFAULT_SETTINGS = dict(
        auto_start=2,
        base_dir=None,
        initdb=None,
        initdb_args="--auth-local=peer --auth-host=reject",
        postgres=None,
        postgres_args="-F -c logging_collector=off",
        pid=None,
        port=None,
        copy_data_from=None,
    )

    def dsn(self, **kwargs) -> dict:
        """Data Source Name parameters that can be used by methods
        in the parent class, testing.postgresql.Postgresql.
        """
        # Keys are those required by pg8000, the postgresql interface
        # module used by the testing.postgresql.Postgresql class.
        params = dict(kwargs)
        params.setdefault("unix_sock", f'{self.base_dir}/tmp/.s.PGSQL.{self.settings["port"]}')
        params.setdefault("user", getpass.getuser())
        params.setdefault("database", "test")
        return params

    def psycopg2_dsn(self, **kwargs) -> dict:
        """Data Source Name parameters that can be used by psycopg2."""
        params = dict(kwargs)
        params.setdefault("port", self.settings["port"])
        params.setdefault("host", f"{self.base_dir}/tmp")
        params.setdefault("user", getpass.getuser())
        params.setdefault("database", "test")
        return params

    def url(self, **kwargs) -> str:
        """Data Source Name string that can be used by sqlalchemy."""
        params = self.psycopg2_dsn(**kwargs)
        url = f"postgresql:///{params['database']}?host={params['host']}"
        return url
