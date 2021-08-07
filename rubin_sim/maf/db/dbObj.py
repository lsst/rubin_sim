import numpy
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy.engine import reflection, url
from sqlalchemy import (create_engine, MetaData, event, inspect)
import warnings
from io import BytesIO
str_cast = str


__all__ = ['DBObject']


def valueOfPi():
    """
    A function to return the value of pi.  This is needed for adding PI()
    to sqlite databases
    """
    return numpy.pi


def declareTrigFunctions(conn, connection_rec, connection_proxy):
    """
    A database event listener
    which will define the math functions necessary for evaluating the
    Haversine function in sqlite databases (where they are not otherwise
    defined)

    see:    http://docs.sqlalchemy.org/en/latest/core/events.html
    """

    conn.create_function("COS", 1, numpy.cos)
    conn.create_function("SIN", 1, numpy.sin)
    conn.create_function("ASIN", 1, numpy.arcsin)
    conn.create_function("SQRT", 1, numpy.sqrt)
    conn.create_function("POWER", 2, numpy.power)
    conn.create_function("PI", 0, valueOfPi)


class ChunkIterator(object):
    """Iterator for query chunks"""
    def __init__(self, dbobj, query, chunk_size, arbitrarySQL=False):
        self.dbobj = dbobj
        self.exec_query = dbobj.connection.session.execute(query)
        self.chunk_size = chunk_size

        # arbitrarySQL exists in case a CatalogDBObject calls
        # get_arbitrary_chunk_iterator; in that case, we need to
        # be able to tell this object to call _postprocess_arbitrary_results,
        # rather than _postprocess_results
        self.arbitrarySQL = arbitrarySQL

    def __iter__(self):
        return self

    def __next__(self):
        if self.chunk_size is None and not self.exec_query.closed:
            chunk = self.exec_query.fetchall()
            return self._postprocess_results(chunk)
        elif self.chunk_size is not None:
            chunk = self.exec_query.fetchmany(self.chunk_size)
            return self._postprocess_results(chunk)
        else:
            raise StopIteration

    def _postprocess_results(self, chunk):
        if len(chunk) == 0:
            raise StopIteration
        if self.arbitrarySQL:
            return self.dbobj._postprocess_arbitrary_results(chunk)
        else:
            return self.dbobj._postprocess_results(chunk)


class DBConnection(object):
    """
    This is a class that will hold the engine, session, and metadata for a
    DBObject.  This will allow multiple DBObjects to share the same
    sqlalchemy connection, when appropriate.
    """

    def __init__(self, database=None, driver=None, host=None, port=None, verbose=False):
        """
        @param [in] database is the name of the database file being connected to

        @param [in] driver is the dialect of the database (e.g. 'sqlite', 'mssql', etc.)

        @param [in] host is the URL of the remote host, if appropriate

        @param [in] port is the port on the remote host to connect to, if appropriate

        @param [in] verbose is a `bool` controlling sqlalchemy's verbosity
        """

        self._database = database
        self._driver = driver
        self._host = host
        self._port = port
        self._verbose = verbose

        self._validate_conn_params()
        self._connect_to_engine()

    def __del__(self):
        try:
            del self._metadata
        except AttributeError:
            pass

        try:
            del self._engine
        except AttributeError:
            pass

        try:
            del self._session
        except AttributeError:
            pass

    def _connect_to_engine(self):

        # Remove dbAuth things. Assume we are only connecting to a local database.
        dbUrl = url.URL.create(self._driver, database=self._database)

        self._engine = create_engine(dbUrl, echo=self._verbose)

        if self._engine.dialect.name == 'sqlite':
            event.listen(self._engine, 'checkout', declareTrigFunctions)

        self._session = scoped_session(sessionmaker(autoflush=True,
                                                    bind=self._engine))
        self._metadata = MetaData(bind=self._engine)

    def _validate_conn_params(self):
        """Validate connection parameters

        - Check if user passed dbAddress instead of an database. Convert and warn.
        - Check that required connection Parameters are present
        - Replace default host/port if driver is 'sqlite'
        """

        if self._database is None:
            raise AttributeError("Cannot instantiate DBConnection; database is 'None'")

        if '//' in self._database:
            warnings.warn("Database name '%s' is invalid but looks like a dbAddress. "
                          "Attempting to convert to database, driver, host, "
                          "and port parameters. Any usernames and passwords are ignored and must "
                          "be in the db-auth.paf policy file. " % (self.database), FutureWarning)

            dbUrl = url.make_url(self._database)
            dialect = dbUrl.get_dialect()
            self._driver = dialect.name + '+' + dialect.driver if dialect.driver else dialect.name
            for key, value in dbUrl.translate_connect_args().items():
                if value is not None:
                    setattr(self, '_'+key, value)

        errMessage = "Please supply a 'driver' kwarg to the constructor or in class definition. "
        errMessage += "'driver' is formatted as dialect+driver, such as 'sqlite' or 'mssql+pymssql'."
        if not hasattr(self, '_driver'):
            raise AttributeError("%s has no attribute 'driver'. " % (self.__class__.__name__) + errMessage)
        elif self._driver is None:
            raise AttributeError("%s.driver is None. " % (self.__class__.__name__) + errMessage)

        errMessage = "Please supply a 'database' kwarg to the constructor or in class definition. "
        errMessage += " 'database' is the database name or the filename path if driver is 'sqlite'. "
        if not hasattr(self, '_database'):
            raise AttributeError("%s has no attribute 'database'. " % (self.__class__.__name__) + errMessage)
        elif self._database is None:
            raise AttributeError("%s.database is None. " % (self.__class__.__name__) + errMessage)

        if 'sqlite' in self._driver:
            # When passed sqlite database, override default host/port
            self._host = None
            self._port = None

    def __eq__(self, other):
        return (str(self._database) == str(other._database)) and \
               (str(self._driver) == str(other._driver)) and \
               (str(self._host) == str(other._host)) and \
               (str(self._port) == str(other._port))

    @property
    def engine(self):
        return self._engine

    @property
    def session(self):
        return self._session

    @property
    def metadata(self):
        return self._metadata

    @property
    def database(self):
        return self._database

    @property
    def driver(self):
        return self._driver

    @property
    def host(self):
        return self._host

    @property
    def port(self):
        return self._port

    @property
    def verbose(self):
        return self._verbose


class DBObject(object):

    def __init__(self, database=None, driver=None, host=None, port=None, verbose=False,
                 connection=None, cache_connection=True):
        """
        Initialize DBObject.

        @param [in] database is the name of the database file being connected to

        @param [in] driver is the dialect of the database (e.g. 'sqlite', 'mssql', etc.)

        @param [in] host is the URL of the remote host, if appropriate

        @param [in] port is the port on the remote host to connect to, if appropriate

        @param [in] verbose is a `bool` controlling sqlalchemy's verbosity (default False)

        @param [in] connection is an optional instance of DBConnection, in the event that
        this DBObject can share a database connection with another DBObject.  This is only
        necessary or even possible in a few specialized cases and should be used carefully.

        @param [in] cache_connection is a `bool`.  If True, DBObject will use a cache of
        DBConnections (if available) to get the connection to this database.
        """

        self.dtype = None
        # this is a cache for the query, so that any one query does not have to guess dtype multiple times

        if connection is None:
            # Explicit constructor to DBObject preferred
            kwargDict = dict(database=database,
                             driver=driver,
                             host=host,
                             port=port,
                             verbose=verbose)

            for key, value in kwargDict.items():
                if value is not None or not hasattr(self, key):
                    setattr(self, key, value)

            self.connection = self._get_connection(self.database, self.driver, self.host, self.port,
                                                   use_cache=cache_connection)

        else:
            self.connection = connection
            self.database = connection.database
            self.driver = connection.driver
            self.host = connection.host
            self.port = connection.port
            self.verbose = connection.verbose

    def _get_connection(self, database, driver, host, port, use_cache=True):
        """
        Search self._connection_cache (if it exists; it won't for DBObject, but
        will for CatalogDBObject) for a DBConnection matching the specified
        parameters.  If it exists, return it.  If not, open a connection to
        the specified database, add it to the cache, and return the connection.

        Parameters
        ----------
        database is the name of the database file being connected to

        driver is the dialect of the database (e.g. 'sqlite', 'mssql', etc.)

        host is the URL of the remote host, if appropriate

        port is the port on the remote host to connect to, if appropriate

        use_cache is a `bool` specifying whether or not we try to use the
        cache of database connections (you don't want to if opening many
        connections in many threads).
        """

        if use_cache and hasattr(self, '_connection_cache'):
            for conn in self._connection_cache:
                if str(conn.database) == str(database):
                    if str(conn.driver) == str(driver):
                        if str(conn.host) == str(host):
                            if str(conn.port) == str(port):
                                return conn

        conn = DBConnection(database=database, driver=driver, host=host, port=port)

        if use_cache and hasattr(self, '_connection_cache'):
            self._connection_cache.append(conn)

        return conn

    def get_table_names(self):
        """Return a list of the names of the tables (and views) in the database"""
        return [str(xx) for xx in inspect(self.connection.engine).get_table_names()] + \
               [str(xx) for xx in inspect(self.connection.engine).get_view_names()]

    def get_column_names(self, tableName=None):
        """
        Return a list of the names of the columns in the specified table.
        If no table is specified, return a dict of lists.  The dict will be keyed
        to the table names.  The lists will be of the column names in that table
        """
        tableNameList = self.get_table_names()
        if tableName is not None:
            if tableName not in tableNameList:
                return []
            return [str_cast(xx['name']) for xx in inspect(self.connection.engine).get_columns(tableName)]
        else:
            columnDict = {}
            for name in tableNameList:
                columnList = [str_cast(xx['name']) for xx in inspect(self.connection.engine).get_columns(name)]
                columnDict[name] = columnList
            return columnDict

    def _final_pass(self, results):
        """ Make final modifications to a set of data before returning it to the user

        **Parameters**

            * results : a structured array constructed from the result set from a query

        **Returns**

            * results : a potentially modified structured array.  The default is to do nothing.

        """
        return results

    def _convert_results_to_numpy_recarray_dbobj(self, results):
        if self.dtype is None:
            """
            Determine the dtype from the data.
            Store it in a global variable so we do not have to repeat on every chunk.
            """
            dataString = ''

            # We are going to detect the dtype by reading in a single row
            # of data with np.genfromtxt.  To do this, we must pass the
            # row as a string delimited by a specified character.  Here we
            # select a character that does not occur anywhere in the data.
            delimit_char_list = [',', ';', '|', ':', '/', '\\']
            delimit_char = None
            for cc in delimit_char_list:
                is_valid = True
                for xx in results[0]:
                    if cc in str(xx):
                        is_valid = False
                        break

                if is_valid:
                    delimit_char = cc
                    break

            if delimit_char is None:
                raise RuntimeError("DBObject could not detect the dtype of your return rows\n"
                                   "Please specify a dtype with the 'dtype' kwarg.")

            for xx in results[0]:
                if dataString != '':
                    dataString += delimit_char
                dataString += str(xx)
            names = [str_cast(ww) for ww in results[0].keys()]
            dataArr = numpy.genfromtxt(BytesIO(dataString.encode()), dtype=None,
                                       names=names, delimiter=delimit_char,
                                       encoding='utf-8')
            dt_list = []
            for name in dataArr.dtype.names:
                type_name = str(dataArr.dtype[name])
                sub_list = [name]
                if type_name.startswith('S') or type_name.startswith('|S'):
                    sub_list.append(str_cast)
                    sub_list.append(int(type_name.replace('S', '').replace('|', '')))
                else:
                    sub_list.append(dataArr.dtype[name])
                dt_list.append(tuple(sub_list))

            self.dtype = numpy.dtype(dt_list)

        if len(results) == 0:
            return numpy.recarray((0,), dtype=self.dtype)

        retresults = numpy.rec.fromrecords([tuple(xx) for xx in results], dtype=self.dtype)
        return retresults

    def _postprocess_results(self, results):
        """
        This wrapper exists so that a ChunkIterator built from a DBObject
        can have the same API as a ChunkIterator built from a CatalogDBObject
        """
        return self._postprocess_arbitrary_results(results)

    def _postprocess_arbitrary_results(self, results):

        if not isinstance(results, numpy.recarray):
            retresults = self._convert_results_to_numpy_recarray_dbobj(results)
        else:
            retresults = results

        return self._final_pass(retresults)

    def execute_arbitrary(self, query, dtype=None):
        """
        Executes an arbitrary query.  Returns a recarray of the results.

        dtype will be the dtype of the output recarray.  If it is None, then
        the code will guess the datatype and assign generic names to the columns
        """

        is_string = isinstance(query, str)

        if not is_string:
            raise RuntimeError("DBObject execute must be called with a string query")

        unacceptableCommands = ["delete", "drop", "insert", "update"]
        for badCommand in unacceptableCommands:
            if query.lower().find(badCommand.lower()) >= 0:
                raise RuntimeError("query made to DBObject execute contained %s " % badCommand)

        self.dtype = dtype
        retresults = self._postprocess_arbitrary_results(self.connection.session.execute(query).fetchall())
        return retresults

    def get_arbitrary_chunk_iterator(self, query, chunk_size=None, dtype=None):
        """
        This wrapper exists so that CatalogDBObjects can refer to
        get_arbitrary_chunk_iterator and DBObjects can refer to
        get_chunk_iterator
        """
        return self.get_chunk_iterator(query, chunk_size=chunk_size, dtype=dtype)

    def get_chunk_iterator(self, query, chunk_size=None, dtype=None):
        """
        Take an arbitrary, user-specified query and return a ChunkIterator that
        executes that query

        dtype will tell the ChunkIterator what datatype to expect for this query.
        This information gets passed to _postprocess_results.

        If 'None', then _postprocess_results will just guess the datatype
        and return generic names for the columns.
        """
        self.dtype = dtype
        return ChunkIterator(self, query, chunk_size, arbitrarySQL=True)
