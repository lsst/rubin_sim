from itertools import zip_longest

__all__ = ["CUT_TYPEMAP", "FieldSelection"]

CUT_TYPEMAP = {
    "RA": "fieldRA",
    "Dec": "fieldDec",
    "GL": "fieldGL",
    "GB": "fieldGB",
    "EL": "fieldEL",
    "EB": "fieldEB"
}
"""Mapping of short names to field database column names."""

class FieldSelection(object):
    """Class for constructing SQL queries on the survey fields database.

    This class is for creating SQL queries to perform on the survey fields
    database. It does not actually perform the queries.
    """

    def base_select(self):
        """Return the base field query.

        Returns
        -------
        str
        """
        return "select * from Field"

    def combine_queries(self, *queries, **kwargs):
        """Combine a set of queries.

        Parameters
        ----------
        queries : str instances
            A set of queries to join via the given operators.
        combiners : tuple of str
            A set of logical operations (and, or etc.) to join the queries
            with. Defaults is an empty tuple. NOTE: A tuple with one logical
            operator must look like ('and',).
        order_by : str, optional
            Set the order by clause. Default is fieldId.

        Returns
        -------
        str:
            The fully combined query.
        """
        combiners = kwargs.get("combiners", ())
        if len(combiners) != len(queries) - 1:
            raise RuntimeError("Number of combiners must be one less than "
                               "number of queries!")

        order_by = kwargs.get("order_by", "fieldId")

        final_query = []
        final_query.append(self.base_select())
        final_query.append("where")
        for combine, query in zip_longest(combiners, queries):
            final_query.append(query)
            if combine is not None:
                final_query.append(combine)
        final_query.append("order by {}".format(order_by))

        return self.finish_query(" ".join(final_query))

    def finish_query(self, query):
        """Put a semicolon at the end of a query.

        Parameters
        ----------
        query : str
            The SQL query to finish.

        Returns
        -------
        str
            The finished SQl query.
        """
        return query + ";"

    def galactic_region(self, maxB, minB, endL, exclusion=False):
        """Create a galactic region.

        This function creates a sloping region around the galactic plane to
        either include or exclude fields.

        Parameters
        ----------
        maxB : float
            The maximum galactic latitude at the galactic longitude of zero.
        minB : float
            The minimum galactic latitude at the galactic longitude of endL.
        endL : float
            The galactic longitude for the end of the envelope region.
        exclusion : bool, optional
            Flag to construct the query as an exclusion. Default is False.

        Returns
        -------
        str
            The appropriate query.
        """
        region_select = ">" if exclusion else "<="
        band = maxB - minB
        sql = '(abs(fieldGB) {0} ({1} - ({2} * '\
              'abs(fieldGL)) / {3}))'.format(region_select, maxB, band, endL)

        return sql

    def get_all_fields(self):
        """Return query for all fields.

        Returns
        -------
        str
            The query for all the fields.
        """
        return self.finish_query(self.base_select())

    def select_region(self, region_type, start_value, end_value):
        """Create a simple bounded region.

        This function creates a bounded cut query based on the input values as
        bounds for a given region. If start_value < end_value, the cut looks
        like [start_value, end_value]. If start_value > end_value, the bounded
        cut is or'd between the following cuts: [start_value, 360] and
        [0, end_value].

        Parameters
        ----------
        region_type : str
            The name of the region to cut on.
        start_value : float
            The starting value (degrees) of the cut region.
        end_value : float
            The ending value (degrees) of the cut region.

        Returns
        -------
        str
            The appropriate query.
        """
        column_name = CUT_TYPEMAP[region_type]
        if end_value > start_value:
            sql = '{0} between {1} and {2}'.format(column_name, start_value,
                                                   end_value)
        else:
            sql = '({0} between {1} and 360 or '\
                  '{0} between 0 and {2})'.format(column_name, start_value,
                                                  end_value)

        return sql

    def select_user_regions(self, id_list):
        """Create a query for a list of fields.of

        This function creates a query focusing on field Ids. It is recommended
        not to use this with more than a dozen Ids.

        Parameters
        ----------
        id_list : list[int]
            A set of field Ids to construct a query for.query

        Returns
        -------
        str
            The appropriate query.
        """
        sql = []
        for fid in id_list:
            sql.append("fieldId={}".format(fid))
            sql.append("or")

        # Don't need last or
        del sql[-1]

        return " ".join(sql)
