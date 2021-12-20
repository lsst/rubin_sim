import os
import sqlite3
import sys
import numpy as np
import unittest
import warnings
import tempfile
import shutil
from rubin_sim.maf.db import DBObject
from rubin_sim.utils.CodeUtilities import sims_clean_up

ROOT = os.path.abspath(os.path.dirname(__file__))


class DBObjectTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Create a database with two tables of meaningless data to make sure that JOIN queries
        can be executed using DBObject
        """
        cls.scratch_dir = tempfile.mkdtemp(dir=ROOT, prefix="scratchSpace-")
        cls.db_name = os.path.join(cls.scratch_dir, "testDBObjectDB.db")
        if os.path.exists(cls.db_name):
            os.unlink(cls.db_name)

        conn = sqlite3.connect(cls.db_name)
        c = conn.cursor()
        try:
            c.execute("""CREATE TABLE intTable (id int, twice int, thrice int)""")
            conn.commit()
        except:
            raise RuntimeError("Error creating database.")

        for ii in range(100):
            ll = 2 * ii
            jj = 2 * ll
            kk = 3 * ll
            cmd = """INSERT INTO intTable VALUES (%s, %s, %s)""" % (ll, jj, kk)
            c.execute(cmd)

        conn.commit()

        c = conn.cursor()
        try:
            c.execute("""CREATE TABLE doubleTable (id int, sqrt float, log float)""")
            conn.commit()
        except:
            raise RuntimeError("Error creating database (double).")
        for ii in range(200):
            ll = ii + 1
            nn = np.sqrt(float(ll))
            mm = np.log(float(ll))

            cmd = """INSERT INTO doubleTable VALUES (%s, %s, %s)""" % (ll, nn, mm)
            c.execute(cmd)
        conn.commit()

        try:
            c.execute("""CREATE TABLE junkTable (id int, sqrt float, log float)""")
            conn.commit()
        except:
            raise RuntimeError("Error creating database (double).")
        for ii in range(200):
            ll = ii + 1
            nn = np.sqrt(float(ll))
            mm = np.log(float(ll))

            cmd = """INSERT INTO junkTable VALUES (%s, %s, %s)""" % (ll, nn, mm)
            c.execute(cmd)

        conn.commit()
        conn.close()

    @classmethod
    def tearDownClass(cls):
        sims_clean_up()
        if os.path.exists(cls.db_name):
            os.unlink(cls.db_name)
        if os.path.exists(cls.scratch_dir):
            shutil.rmtree(cls.scratch_dir)

    def setUp(self):
        self.driver = "sqlite"

    def testTableNames(self):
        """
        Test the method that returns the names of tables in a database
        """
        dbobj = DBObject(driver=self.driver, database=self.db_name)
        names = dbobj.get_table_names()
        self.assertEqual(len(names), 3)
        self.assertIn("doubleTable", names)
        self.assertIn("intTable", names)

    def testReadOnlyFilter(self):
        """
        Test that the filters we placed on queries made with execute_aribtrary()
        work
        """
        dbobj = DBObject(driver=self.driver, database=self.db_name)
        controlQuery = (
            "SELECT doubleTable.id, intTable.id, doubleTable.log, intTable.thrice "
        )
        controlQuery += "FROM doubleTable, intTable WHERE doubleTable.id = intTable.id"
        dbobj.execute_arbitrary(controlQuery)

        # make sure that execute_arbitrary only accepts strings
        query = ["a", "list"]
        self.assertRaises(RuntimeError, dbobj.execute_arbitrary, query)

        # check that our filter catches different capitalization permutations of the
        # verboten commands
        query = "DROP TABLE junkTable"
        self.assertRaises(RuntimeError, dbobj.execute_arbitrary, query)
        self.assertRaises(RuntimeError, dbobj.execute_arbitrary, query.lower())
        query = "DELETE FROM junkTable WHERE id=4"
        self.assertRaises(RuntimeError, dbobj.execute_arbitrary, query)
        self.assertRaises(RuntimeError, dbobj.execute_arbitrary, query.lower())
        query = "UPDATE junkTable SET sqrt=0.0, log=0.0 WHERE id=4"
        self.assertRaises(RuntimeError, dbobj.execute_arbitrary, query)
        self.assertRaises(RuntimeError, dbobj.execute_arbitrary, query.lower())
        query = "INSERT INTO junkTable VALUES (9999, 1.0, 1.0)"
        self.assertRaises(RuntimeError, dbobj.execute_arbitrary, query)
        self.assertRaises(RuntimeError, dbobj.execute_arbitrary, query.lower())

        query = "Drop Table junkTable"
        self.assertRaises(RuntimeError, dbobj.execute_arbitrary, query)
        query = "Delete FROM junkTable WHERE id=4"
        self.assertRaises(RuntimeError, dbobj.execute_arbitrary, query)
        query = "Update junkTable SET sqrt=0.0, log=0.0 WHERE id=4"
        self.assertRaises(RuntimeError, dbobj.execute_arbitrary, query)
        query = "Insert INTO junkTable VALUES (9999, 1.0, 1.0)"
        self.assertRaises(RuntimeError, dbobj.execute_arbitrary, query)

        query = "dRoP TaBlE junkTable"
        self.assertRaises(RuntimeError, dbobj.execute_arbitrary, query)
        query = "dElEtE FROM junkTable WHERE id=4"
        self.assertRaises(RuntimeError, dbobj.execute_arbitrary, query)
        query = "uPdAtE junkTable SET sqrt=0.0, log=0.0 WHERE id=4"
        self.assertRaises(RuntimeError, dbobj.execute_arbitrary, query)
        query = "iNsErT INTO junkTable VALUES (9999, 1.0, 1.0)"
        self.assertRaises(RuntimeError, dbobj.execute_arbitrary, query)

    def testColumnNames(self):
        """
        Test the method that returns the names of columns in a table
        """
        dbobj = DBObject(driver=self.driver, database=self.db_name)
        names = dbobj.get_column_names("doubleTable")
        self.assertEqual(len(names), 3)
        self.assertIn("id", names)
        self.assertIn("sqrt", names)
        self.assertIn("log", names)

        names = dbobj.get_column_names("intTable")
        self.assertEqual(len(names), 3)
        self.assertIn("id", names)
        self.assertIn("twice", names)
        self.assertIn("thrice", names)

        names = dbobj.get_column_names()
        keys = ["doubleTable", "intTable", "junkTable"]
        for kk in names:
            self.assertIn(kk, keys)

        self.assertEqual(len(names["doubleTable"]), 3)
        self.assertEqual(len(names["intTable"]), 3)
        self.assertIn("id", names["doubleTable"])
        self.assertIn("sqrt", names["doubleTable"])
        self.assertIn("log", names["doubleTable"])
        self.assertIn("id", names["intTable"])
        self.assertIn("twice", names["intTable"])
        self.assertIn("thrice", names["intTable"])

    def testSingleTableQuery(self):
        """
        Test a query on a single table (using chunk iterator)
        """
        dbobj = DBObject(driver=self.driver, database=self.db_name)
        query = "SELECT id, sqrt FROM doubleTable"
        results = dbobj.get_chunk_iterator(query)

        dtype = [("id", int), ("sqrt", float)]

        i = 1
        for chunk in results:
            for row in chunk:
                self.assertEqual(row[0], i)
                self.assertAlmostEqual(row[1], np.sqrt(i))
                self.assertEqual(dtype, row.dtype)
                i += 1

        self.assertEqual(i, 201)

    def testDtype(self):
        """
        Test that passing dtype to a query works

        (also test q query on a single table using .execute_arbitrary() directly
        """
        dbobj = DBObject(driver=self.driver, database=self.db_name)
        query = "SELECT id, log FROM doubleTable"
        dtype = [("id", int), ("log", float)]
        results = dbobj.execute_arbitrary(query, dtype=dtype)

        self.assertEqual(results.dtype, dtype)
        for xx in results:
            self.assertAlmostEqual(np.log(xx[0]), xx[1], 6)

        self.assertEqual(len(results), 200)

        results = dbobj.get_chunk_iterator(query, chunk_size=10, dtype=dtype)
        next(results)
        for chunk in results:
            self.assertEqual(chunk.dtype, dtype)

    def testJoin(self):
        """
        Test a join
        """
        dbobj = DBObject(driver=self.driver, database=self.db_name)
        query = "SELECT doubleTable.id, intTable.id, doubleTable.log, intTable.thrice "
        query += "FROM doubleTable, intTable WHERE doubleTable.id = intTable.id"
        results = dbobj.get_chunk_iterator(query, chunk_size=10)

        dtype = [("id", int), ("id_1", int), ("log", float), ("thrice", int)]

        i = 0
        for chunk in results:
            if i < 90:
                self.assertEqual(len(chunk), 10)
            for row in chunk:
                self.assertEqual(2 * (i + 1), row[0])
                self.assertEqual(row[0], row[1])
                self.assertAlmostEqual(np.log(row[0]), row[2], 6)
                self.assertEqual(3 * row[0], row[3])
                self.assertEqual(dtype, row.dtype)
                i += 1
        self.assertEqual(i, 99)
        # make sure that we found all the matches whe should have

        results = dbobj.execute_arbitrary(query)
        self.assertEqual(dtype, results.dtype)
        i = 0
        for row in results:
            self.assertEqual(2 * (i + 1), row[0])
            self.assertEqual(row[0], row[1])
            self.assertAlmostEqual(np.log(row[0]), row[2], 6)
            self.assertEqual(3 * row[0], row[3])
            i += 1
        self.assertEqual(i, 99)
        # make sure we found all the matches we should have

    def testMinMax(self):
        """
        Test queries on SQL functions by using the MIN and MAX functions
        """
        dbobj = DBObject(driver=self.driver, database=self.db_name)
        query = "SELECT MAX(thrice), MIN(thrice) FROM intTable"
        results = dbobj.execute_arbitrary(query)
        self.assertEqual(results[0][0], 594)
        self.assertEqual(results[0][1], 0)

        dtype = [("MAXthrice", int), ("MINthrice", int)]
        self.assertEqual(results.dtype, dtype)

    def testPassingConnection(self):
        """
        Repeat the test from testJoin, but with a DBObject whose connection was passed
        directly from another DBObject, to make sure that passing a connection works
        """
        dbobj_base = DBObject(driver=self.driver, database=self.db_name)
        dbobj = DBObject(connection=dbobj_base.connection)
        query = "SELECT doubleTable.id, intTable.id, doubleTable.log, intTable.thrice "
        query += "FROM doubleTable, intTable WHERE doubleTable.id = intTable.id"
        results = dbobj.get_chunk_iterator(query, chunk_size=10)

        dtype = [("id", int), ("id_1", int), ("log", float), ("thrice", int)]

        i = 0
        for chunk in results:
            if i < 90:
                self.assertEqual(len(chunk), 10)
            for row in chunk:
                self.assertEqual(2 * (i + 1), row[0])
                self.assertEqual(row[0], row[1])
                self.assertAlmostEqual(np.log(row[0]), row[2], 6)
                self.assertEqual(3 * row[0], row[3])
                self.assertEqual(dtype, row.dtype)
                i += 1
        self.assertEqual(i, 99)
        # make sure that we found all the matches whe should have

        results = dbobj.execute_arbitrary(query)
        self.assertEqual(dtype, results.dtype)
        i = 0
        for row in results:
            self.assertEqual(2 * (i + 1), row[0])
            self.assertEqual(row[0], row[1])
            self.assertAlmostEqual(np.log(row[0]), row[2], 6)
            self.assertEqual(3 * row[0], row[3])
            i += 1
        self.assertEqual(i, 99)
        # make sure we found all the matches we should have

    def testValidationErrors(self):
        """Test that appropriate errors and warnings are thrown when connecting"""

        # missing database
        self.assertRaises(AttributeError, DBObject, driver=self.driver)
        # missing driver
        self.assertRaises(AttributeError, DBObject, database=self.db_name)
        # missing host
        self.assertRaises(AttributeError, DBObject, driver="mssql+pymssql")
        # missing port
        self.assertRaises(
            AttributeError, DBObject, driver="mssql+pymssql", host="localhost"
        )

    def testDetectDtype(self):
        """
        Test that DBObject.execute_arbitrary can correctly detect the dtypes
        of the rows it is returning
        """
        db_name = os.path.join(self.scratch_dir, "testDBObject_dtype_DB.db")
        if os.path.exists(db_name):
            os.unlink(db_name)

        conn = sqlite3.connect(db_name)
        c = conn.cursor()
        try:
            c.execute("""CREATE TABLE testTable (id int, val real, sentence int)""")
            conn.commit()
        except:
            raise RuntimeError("Error creating database.")

        for ii in range(10):
            cmd = """INSERT INTO testTable VALUES (%d, %.5f, %s)""" % (
                ii,
                5.234 * ii,
                "'this, has; punctuation'",
            )
            c.execute(cmd)

        conn.commit()
        conn.close()

        db = DBObject(database=db_name, driver="sqlite")
        query = "SELECT id, val, sentence FROM testTable WHERE id%2 = 0"
        results = db.execute_arbitrary(query)

        np.testing.assert_array_equal(results["id"], np.arange(0, 9, 2, dtype=int))
        np.testing.assert_array_almost_equal(
            results["val"], 5.234 * np.arange(0, 9, 2), decimal=5
        )
        for sentence in results["sentence"]:
            self.assertEqual(sentence, "this, has; punctuation")

        self.assertEqual(str(results.dtype["id"]), "int64")
        self.assertEqual(str(results.dtype["val"]), "float64")
        if sys.version_info.major == 2:
            self.assertEqual(str(results.dtype["sentence"]), "|S22")
        else:
            self.assertEqual(str(results.dtype["sentence"]), "<U22")
        self.assertEqual(len(results.dtype), 3)

        # now test that it works when getting a ChunkIterator
        chunk_iter = db.get_arbitrary_chunk_iterator(query, chunk_size=3)
        ct = 0
        for chunk in chunk_iter:

            self.assertEqual(str(chunk.dtype["id"]), "int64")
            self.assertEqual(str(chunk.dtype["val"]), "float64")
            if sys.version_info.major == 2:
                self.assertEqual(str(results.dtype["sentence"]), "|S22")
            else:
                self.assertEqual(str(results.dtype["sentence"]), "<U22")
            self.assertEqual(len(chunk.dtype), 3)

            for line in chunk:
                ct += 1
                self.assertEqual(line["sentence"], "this, has; punctuation")
                self.assertAlmostEqual(line["val"], line["id"] * 5.234, 5)
                self.assertEqual(line["id"] % 2, 0)

        self.assertEqual(ct, 5)

        # test that doing a different query does not spoil dtype detection
        query = "SELECT id, sentence FROM testTable WHERE id%2 = 0"
        results = db.execute_arbitrary(query)
        self.assertGreater(len(results), 0)
        self.assertEqual(len(results.dtype.names), 2)
        self.assertEqual(str(results.dtype["id"]), "int64")
        if sys.version_info.major == 2:
            self.assertEqual(str(results.dtype["sentence"]), "|S22")
        else:
            self.assertEqual(str(results.dtype["sentence"]), "<U22")

        query = "SELECT id, val, sentence FROM testTable WHERE id%2 = 0"
        chunk_iter = db.get_arbitrary_chunk_iterator(query, chunk_size=3)
        ct = 0
        for chunk in chunk_iter:

            self.assertEqual(str(chunk.dtype["id"]), "int64")
            self.assertEqual(str(chunk.dtype["val"]), "float64")
            if sys.version_info.major == 2:
                self.assertEqual(str(results.dtype["sentence"]), "|S22")
            else:
                self.assertEqual(str(results.dtype["sentence"]), "<U22")
            self.assertEqual(len(chunk.dtype), 3)

            for line in chunk:
                ct += 1
                self.assertEqual(line["sentence"], "this, has; punctuation")
                self.assertAlmostEqual(line["val"], line["id"] * 5.234, 5)
                self.assertEqual(line["id"] % 2, 0)

        self.assertEqual(ct, 5)

        if os.path.exists(db_name):
            os.unlink(db_name)


if __name__ == "__main__":
    unittest.main()
