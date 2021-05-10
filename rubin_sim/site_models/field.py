import math

class Field(object):
    """Gather survey field information.

    Attributes
    ----------
    fid : int
        An identifier for the field.
    ra : float
        The right-ascension (degrees) of the field.
    dec : float
        The declination (degrees) of the field.
    gl : float
        The galactic longitude (degrees) of the field.
    gb : float
        The galactic latitude (degrees) of the field.
    el : float
        The ecliptic longitude (degrees) of the field.
    eb : float
        The ecliptic latitude (degrees) of the field.
    fov : float
        The field-of-view (degrees) of the field.
    """

    def __init__(self, fid, ra, dec, gl, gb, el, eb, fov):
        """Initialize class.

        Parameters
        ----------
        fid : int
            An indentifier for the field.
        ra : float
            The right-ascension (degrees) for the field.
        dec : float
            The declination (degrees) for the field.
        gl : float
            The galactic longitude (degrees) for the field.
        gb : float
            The galactic latitude (degrees) for the field.
        el : float
            The ecliptic longitude (degrees) for the field.
        eb : float
            The ecliptic latitude (degrees) for the field.
        fov : float
            The field-of-view (degrees) for the field.
        """
        self.fid = fid
        self.ra = ra
        self.dec = dec
        self.gl = gl
        self.gb = gb
        self.el = el
        self.eb = eb
        self.fov = fov

    def __str__(self):
        """The instance string representation.

        Returns
        -------
        str
        """
        return "Id: {}, RA: {}, Dec:{}, GL: {}, "\
               "GB: {}, EL: {}, EB: {}, FOV: {}".format(self.fid, self.ra,
                                                        self.dec, self.gl,
                                                        self.gb, self.el,
                                                        self.eb, self.fov)

    @property
    def ra_rad(self):
        """float : The right-ascension (radians) of the field.
        """
        return math.radians(self.ra)

    @property
    def dec_rad(self):
        """float : The declination (radians) of the field.
        """
        return math.radians(self.dec)

    @property
    def gl_rad(self):
        """float : The galactic longitude (radians) of the field.
        """
        return math.radians(self.gl)

    @property
    def gb_rad(self):
        """float : The galactic latitude (radians) of the field.
        """
        return math.radians(self.gb)

    @property
    def el_rad(self):
        """float : The ecliptic longitude (radians) of the field.
        """
        return math.radians(self.el)

    @property
    def eb_rad(self):
        """float : The ecliptic latitude (radians) of the field.
        """
        return math.radians(self.eb)

    @property
    def fov_rad(self):
        """float : The field-of-view (radians) of the field.
        """
        return math.radians(self.fov)

    @classmethod
    def from_db_row(cls, row):
        """Create instance from a database table row.

        Parameters
        ----------
        row : list
            The database row information to create the instance from.

        Returns
        -------
        :class:`.Field`
            The instance containing the database row information.
        """
        return cls(row[0], row[2], row[3], row[4], row[5], row[6], row[7],
                   row[1])
