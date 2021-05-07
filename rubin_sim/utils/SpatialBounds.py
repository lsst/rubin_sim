"""
This file defines classes that control who ObservationMetaData describes
its field of view (i.e. is it a box in RA, Dec, is it a circle in RA, Dec....?)
"""

# Hopefully it will be extensible so that we can add different shapes in the
# future

import numpy as np
from six import with_metaclass

__all__ = ["SpatialBounds", "CircleBounds", "BoxBounds"]


class SpatialBoundsMetaClass(type):
    """
    Meta class for fieldOfView.  This class builds a registry of all
    valid fields of view so that fields can be instantiated from just a
    dictionary key.
    """

    # Largely this is being copied from the DBObjectMeta class in
    # dbConnection.py

    def __init__(cls, name, bases, dct):
        if not hasattr(cls, 'SBregistry'):
            cls.SBregistry = {}
        else:
            cls.SBregistry[cls.boundType] = cls

        return super(SpatialBoundsMetaClass, cls).__init__(name, bases, dct)


class SpatialBounds(with_metaclass(SpatialBoundsMetaClass, object)):
    """
    Classes inheriting from this class define spatial bounds on the objects
    contained within a catalog.  They also translate those bounds into
    constraints on SQL queries made by the query_columns method in
    CatalogDBobject (see dbConnnection.py)

    Daughter classes of this class need the following:

    self.boundType = a string by which the class is identified in the
    registry of FieldOfView classes

    __init__() that accepts (in this order) RA, Dec, and characteristic
    length.  Init should then construct the parameters defining the bound
    however is appropriate (e.g. setting self.RAmax and self.RAmin for a box)

    to_SQL() = a method that accepts RAcolname and DECcolname (strings denoting
    the names of the database columns containing RA and DEc) and which returns
    a string that characterizes the bound as an SQL 'WHERE' statement.
    """

    def __init__(self, *args):
        """
        Accepts a center point and a characteristic length defining the extent of
        the bounds

        @param[in] ra is the center RA in radians

        @param[in] dec is the center Dec in radians

        @param[in] length is either a single characteristic length (in radians)
        or a list of characteristic lengths defining the shape of the bound
        """

        raise NotImplementedError()

    def to_SQL(self, *args):
        """
        Accepts the names of the columns referring to RA and Dec in the database.
        Uses the stored RA, Dec, and length for this object to return an SQL
        query that only selects the region of RA and Dec desired

        @param[in] RAname a string; the name of the RA column in the database

        @param[in] DECname a string; the name of the Dec column in the database

        @returns a string; an SQL query that only selects the desired region in RA, Dec
        """

        raise NotImplementedError()

    def __eq__(self, other):
        for param in self.__dict__:
            if param not in other.__dict__:
                return False

            if self.__dict__[param] != other.__dict__[param]:
                return False

        for param in other.__dict__:
            if param not in self.__dict__:
                return False

        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    @classmethod
    def getSpatialBounds(self, name, *args, **kwargs):
        if name in self.SBregistry:
            return self.SBregistry[name](*args, **kwargs)
        else:
            raise RuntimeError("There is no SpatialBounds class keyed to %s" % name)


class CircleBounds(SpatialBounds):

    boundType = 'circle'

    def __init__(self, ra, dec, radius):
        """
        Accepts a center point and a characteristic length defining the extent of
        the bounds

        @param[in] ra is the center RA in radians

        @param[in] dec is the center Dec in radians

        @param[in] length is the radius of the field of view in radians
        """

        if not (isinstance(ra, float) or isinstance(ra, np.float64)):
            try:
                ra = float(ra)
            except:
                raise RuntimeError('In CircleBounds, ra must be a float; you have %s' % type(ra))

        if not (isinstance(dec, float) or isinstance(dec, np.float64)):
            try:
                dec = float(dec)
            except:
                raise RuntimeError('In CircleBounds, dec must be a float; you have %s' % type(dec))

        if not (isinstance(radius, float) or isinstance(radius, np.float64)):
            try:
                radius = float(radius)
            except:
                raise RuntimeError('In CircleBounds, radius must be a float; you have %s' % type(radius))

        self.RA = ra
        self.DEC = dec
        self.radius = radius

        self.RAdeg = np.degrees(ra)
        self.DECdeg = np.degrees(dec)
        self.radiusdeg = np.degrees(radius)

    def __eq__(self, other):
        return (type(self) == type(other)) and \
               (self.RA == other.RA) and \
               (self.DEC == other.DEC) and \
               (self.radius == other.radius) and \
               (self.RAdeg == other.RAdeg) and \
               (self.DECdeg == other.DECdeg) and \
               (self.radiusdeg == other.radiusdeg)

    def to_SQL(self, RAname, DECname):

        cosDec = np.cos(self.DEC)
        adjusted_radius = np.abs(np.degrees(np.arcsin(np.sin(self.radius)/cosDec)))

        if np.abs(cosDec) > 1.0e-20:
            RAmax = self.RAdeg + 2.0*adjusted_radius
            RAmin = self.RAdeg - 2.0*adjusted_radius
        else:
            # just in case, for some reason, we are looking at the poles
            RAmax = 361.0
            RAmin = -361.0

        if (np.isnan(RAmax) or np.isnan(RAmin) or
            RAmin<adjusted_radius or
            RAmax>360.0-adjusted_radius):

            RAmax = 361.0
            RAmin = -361.0

        DECmax = self.DECdeg + self.radiusdeg
        DECmin = self.DECdeg - self.radiusdeg

        # initially demand that all objects are within a box containing the circle
        # set from the DEC1=DEC2 and RA1=RA2 limits of the haversine function
        if RAmax-RAmin<361.0:
            bound = ("%s between %f and %f and %s between %f and %f "
                     % (RAname, RAmin, RAmax, DECname, DECmin, DECmax))
        else:
            bound = ('%s between %f and %f ' % (DECname, DECmin, DECmax))

        # then use the Haversine function to constrain the angular distance form boresite to be within
        # the desired radius.  See
        # http://en.wikipedia.org/wiki/Haversine_formula
        bound = bound + \
            ("and 2 * ASIN(SQRT( POWER(SIN(0.5*(%s - %s) * PI() / 180.0),2)" %
             (DECname, self.DECdeg))
        bound = bound + \
            ("+ COS(%s * PI() / 180.0) * COS(%s * PI() / 180.0) " %
             (DECname, self.DECdeg))
        bound = bound + \
            ("* POWER(SIN(0.5 * (%s - %s) * PI() / 180.0),2)))" %
             (RAname, self.RAdeg))
        bound = bound + (" < %s " % self.radius)

        return bound


class BoxBounds(SpatialBounds):

    boundType = 'box'

    def __init__(self, ra, dec, length):
        """
        Accepts a center point and a characteristic length defining the extent of
        the bounds

        @param[in] ra is the center RA in radians

        @param[in] dec is the center Dec in radians

        @param[in] length is either a single characteristic length (in radians)
        or a list of characteristic lengths defining the shape of the bound.
        If a single value, the field of view will be a square with side of 2 x length.
        If it is a list/tuple/array, the field of view will be a rectangle with side lengths
        RA = 2 x length[0] and Dec = 2 x length[1]
        """

        if not (isinstance(ra, float) or isinstance(ra, np.float64)):
            try:
                ra = float(ra)
            except:
                raise RuntimeError('In BoxBounds ra must be a float; you have %s' % type(ra))

        if not (isinstance(dec, float) or isinstance(dec, np.float64)):
            try:
                dec = float(dec)
            except:
                raise RuntimeError('In BoxBounds dec must be a float; you have %s' % type(dec))

        self.RA = ra
        self.DEC = dec

        self.RAdeg = np.degrees(ra)
        self.DECdeg = np.degrees(dec)

        try:
            if hasattr(length, '__len__'):
                if len(length) == 1:
                    lengthRAdeg = np.degrees(length[0])
                    lengthDECdeg = np.degrees(length[0])
                else:
                    lengthRAdeg = np.degrees(length[0])
                    lengthDECdeg = np.degrees(length[1])
            else:
                length = float(length)
                lengthRAdeg = np.degrees(length)
                lengthDECdeg = np.degrees(length)

        except:
            raise RuntimeError("BoxBounds is unsure how to handle length %s type: %s" % (
                str(length), type(length)))

        self.RAminDeg = self.RAdeg - lengthRAdeg
        self.RAmaxDeg = self.RAdeg + lengthRAdeg
        self.DECminDeg = self.DECdeg - lengthDECdeg
        self.DECmaxDeg = self.DECdeg + lengthDECdeg

        self.RAminDeg %= 360.0
        self.RAmaxDeg %= 360.0

    def __eq__(self, other):
        return (type(self) == type(other)) and \
               (self.RA == other.RA) and \
               (self.RAdeg == other.RAdeg) and \
               (self.DEC == other.DEC) and \
               (self.DECdeg == other.DECdeg) and \
               (self.RAminDeg == other.RAminDeg) and \
               (self.RAmaxDeg == other.RAmaxDeg) and \
               (self.DECminDeg == other.DECminDeg) and \
               (self.DECmaxDeg == other.DECmaxDeg)

    def to_SQL(self, RAname, DECname):
        # KSK:  I don't know exactly what we do here.  This is in code, but operating
        # on a database is it less confusing to work in degrees or radians?
        # (RAmin, RAmax, DECmin, DECmax) = map(math.radians,
        #                                     (RAmin, RAmax, DECmin, DECmax))

        # Special case where the whole region is selected
        if self.RAminDeg < 0 and self.RAmaxDeg > 360.:
            bound = "%s between %f and %f" % (
                DECname, self.DECminDeg, self.DECmaxDeg)
            return bound

        if self.RAminDeg > self.RAmaxDeg:
            bound = ("%s not between %f and %f and %s between %f and %f"
                     % (RAname, self.RAmaxDeg, self.RAminDeg,
                        DECname, self.DECminDeg, self.DECmaxDeg))
            bound += (" and %s+360.0 not between %f and %f" %
                      (RAname, self.RAmaxDeg, self.RAminDeg))
        else:
            bound = ("%s between %f and %f and %s between %f and %f"
                     % (RAname, self.RAminDeg, self.RAmaxDeg, DECname, self.DECminDeg, self.DECmaxDeg))

        return bound
