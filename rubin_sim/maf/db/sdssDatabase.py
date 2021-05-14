from .database import Database
import numpy as np
import warnings

__all__ = ['SdssDatabase']

class SdssDatabase(Database):
    """Connect to the stripe 82 database"""
    def __init__(self, database=None, driver='sqlite', host=None, port=None,
                 dbTables={'clue.dbo.viewStripe82JoinAll':['viewStripe82JoinAll','id']},
                 defaultdbTables=None,
                 chunksize=1000000, **kwargs):
       super(SdssDatabase,self).__init__(database=database, driver=driver, port=port, host=host,
                                         dbTables=dbTables,defaultdbTables=defaultdbTables,
                                         chunksize=chunksize,**kwargs )



    def fetchMetricData(self, colnames, sqlconstraint, groupBy=None,
                        cleanNaNs=True, **kwargs):
        """Get data for metric"""
        table = self.tables['clue.dbo.viewStripe82JoinAll']
        # MSSQL doesn't seem to like double quotes?
        if sqlconstraint != sqlconstraint.replace('"', "'"):
            warnings.warn('Warning:  Replacing double quotes with single quotes in SQL where-clause. \
                           Double quotes are not defined in standard SQL.')
            sqlconstraint = sqlconstraint.replace('"', "'")
        data = table.query_columns_Array(chunk_size = self.chunksize,
                                         constraint = sqlconstraint,
                                         colnames = colnames,
                                         groupByCol = groupBy)
        # Toss columns with NaNs.
        if cleanNaNs:
            for col in colnames:
                good = np.where(np.isnan(data[col]) == False)
                data = data[good]
        return data

