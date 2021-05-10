import os
import sqlite3
import numpy
from rubin_sim.data import get_data_dir

__all__ = ["FieldsDatabase"]


class FieldsDatabase(object):

    FIELDS_DB = "Fields.db"
    """Internal file containing the standard 3.5 degree FOV survey field
       information."""

    def __init__(self):
        """Initialize the class.
        """
        self.db_name = self.FIELDS_DB
        self.connect = sqlite3.connect(os.path.join(get_data_dir(), 'site_models',
                                       self.db_name))

    def __del__(self):
        """Delete the class.
        """
        self.connect.close()

    def get_field_set(self, query):
        """Get a set of Field instances.

        Parameters
        ----------
        query : str
            The query for field retrieval.

        Returns
        -------
        set
            The collection of Field instances.
        """
        field_set = set()
        rows = self.get_rows(query)
        for row in rows:
            field_set.add(tuple(row))

        return field_set

    def get_opsim3_userregions(self, query, precision=2):
        """Get a formatted string of OpSim3 user regions.

        This function gets a formatted string of OpSim3 user regions suitable
        for an OpSim3 configuration file. The format looks like
        (RA,Dec,Width):

        userRegion = XXX.XX,YYY.YY,0.03
        ...

        The last column is unused in OpSim3. The precision argument can be
        used to control the formatting, but OpSim3 configuration files use 2
        digits as standard.

        Parameters
        ----------
        query : str
            The query for field retrieval.
        precision : int, optional
            The precision used for the RA and Dec columns. Default is 2.

        Returns
        -------
        str
            The OpSim3 user regions formatted string.
        """
        format_str = "userRegion = "\
                     "{{:.{0}f}},{{:.{0}f}},0.03".format(precision)
        rows = self.get_rows(query)
        result = []
        for row in rows:
            result.append(format_str.format(row[2], row[3]))
        return str(os.linesep.join(result))

    def get_ra_dec_arrays(self, query):
        """Retrieve lists of RA and Dec.

        Parameters
        ----------
        query : str
            The query for field retrieval.

        Returns
        -------
        numpy.array, numpy.array
            The arrays of RA and Dec.
        """
        rows = self.get_rows(query)
        ra = []
        dec = []
        for row in rows:
            ra.append(row[2])
            dec.append(row[3])

        return numpy.array(ra), numpy.array(dec)

    def get_id_ra_dec_arrays(self, query):
        """Retrieve lists of fieldId, RA and Dec.

        Parameters
        ----------
        query : str
            The query for field retrieval.

        Returns
        -------
        numpy.array, numpy.array, numpy.array
            The arrays of fieldId, RA and Dec.
        """
        rows = self.get_rows(query)
        fieldId = []
        ra = []
        dec = []
        for row in rows:
            fieldId.append(int(row[0]))
            ra.append(row[2])
            dec.append(row[3])

        return numpy.array(fieldId, dtype=int), numpy.array(ra), numpy.array(dec)

    def get_rows(self, query):
        """Get the rows from a query.

        This function hands back all rows from a query. This allows one to
        perform other operations on the information than those provided by
        this class.

        Parameters
        ----------
        query : str
            The query for field retrieval.

        Returns
        -------
        list
            The set of field information queried.
        """
        cursor = self.connect.cursor()
        cursor.execute(query)
        return cursor.fetchall()
