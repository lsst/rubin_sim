__all__ = ("TimeHandler",)

from datetime import datetime, timedelta


class TimeHandler:
    """Keep track of simulation time information.

    This is a class tied to SOCS/Scheduler (OpSim).
    Its properties will be reevaluated in the future and this
    class may disappear.

    Attributes
    ----------
    _unix_start : datetime.datetime
        Holder for the start of the UNIX epoch
    initial_dt : datetime.datetime
        The date/time of the simulation start.
    current_dt : datetime.datetime
        The current simulation date/time.
    """

    def __init__(self, initial_date):
        """Initialize the class.

        Parameters
        ----------
            initial_date : str
                The inital date in the format of YYYY-MM-DD.
        """
        self._unix_start = datetime(1970, 1, 1)
        self.initial_dt = datetime.strptime(initial_date, "%Y-%m-%d")
        self.current_dt = self.initial_dt

    def _time_difference(self, datetime1, datetime2=None):
        """Calculate the difference in seconds between two times.

        This function calculates the difference in seconds between two given :class:`datetime` instances. If
        datetime2 is None, it is assumed to be UNIX epoch start.

        Parameters
        ----------
        datetime1 : datetime.datetime
            The first datetime instance.
        datetime2 : datetime.datetime
            The second datetime instance.

        Returns
        -------
        float
            The difference in seconds between the two datetime instances.
        """
        if datetime2 is None:
            datetime2 = self._unix_start
        return (datetime1 - datetime2).total_seconds()

    @property
    def initial_timestamp(self):
        """float: Return the UNIX timestamp for the initial date/time."""
        return self._time_difference(self.initial_dt)

    @property
    def current_timestamp(self):
        """float: Return the UNIX timestamp for the current date/time."""
        return self._time_difference(self.current_dt)

    @property
    def current_midnight_timestamp(self):
        """float: Return the UNIX timestamp of midnight for the current date."""
        midnight_dt = datetime(self.current_dt.year, self.current_dt.month, self.current_dt.day)
        return self._time_difference(midnight_dt)

    @property
    def next_midnight_timestamp(self):
        """float: Return the UNIX timestamp of midnight for the next day after current date."""
        midnight_dt = datetime(self.current_dt.year, self.current_dt.month, self.current_dt.day)
        midnight_dt += timedelta(**{"days": 1})
        return self._time_difference(midnight_dt)

    @property
    def time_since_start(self):
        """float: The number of seconds since the start date."""
        return self._time_difference(self.current_dt, self.initial_dt)

    def update_time(self, time_increment, time_units):
        """Update the currently held timestamp.

        This function updates the currently held time with the given increment and corresponding
        units.

        Parameters
        ----------
        time_increment : float
            The increment to adjust the current time.
        time_units : str
            The time unit for the increment value.
        """
        time_delta_dict = {time_units: time_increment}
        self.current_dt += timedelta(**time_delta_dict)

    @property
    def current_timestring(self):
        """str: Return the ISO-8601 representation of the current date/time."""
        return self.current_dt.isoformat()

    def has_time_elapsed(self, time_span):
        """Return a `bool` determining if the time span has elapsed.

        This function looks to see if the time elapsed (current_time - initial_time) in units of
        seconds is greater or less than the requested time span. It will return true if the time span
        is greater than or equal the elapsed time and false if less than the elapsed time.

        Parameters
        ----------
        time_span : float
            The requested time span in seconds.

        Returns
        -------
        bool
            True if the time elapsed is greater or False if less than the time span.
        """
        return time_span >= self._time_difference(self.current_dt, self.initial_dt)

    def future_datetime(self, time_increment, time_units, timestamp=None):
        """Return a future datetime object.

        This function adds the requested time increment to the current date/time to get a future date/time
        and returns a datetime object. An alternative timestamp can be supplied and the time increment will
        be applied to that instead. This function does not update the internal timestamp.

        Parameters
        ----------
        time_increment : float
            The increment to adjust the current time.
        time_units : str
            The time unit for the increment value.
        timestamp : float, optional
            An alternative timestamp to apply the time increment to.

        Returns
        -------
        datetime.datetime
            The datetime object for the future date/time.
        """
        if timestamp is not None:
            dt = datetime.utcfromtimestamp(timestamp)
        else:
            dt = self.current_dt
        time_delta_dict = {time_units: time_increment}
        return dt + timedelta(**time_delta_dict)

    def future_timestamp(self, time_increment, time_units, timestamp=None):
        """Return the UNIX timestamp for the future date/time.

        This function adds the requested time increment to the current date/time to get a future date/time
        and returns the UNIX timestamp for that date/time. It does not update the internal timestamp.

        Parameters
        ----------
        time_increment : float
            The increment to adjust the current time.
        time_units : str
            The time unit for the increment value.
        timestamp : float, optional
            An alternative timestamp to apply the time increment to.

        Returns
        -------
        float
            The future UNIX timestamp.
        """
        return self._time_difference(self.future_datetime(time_increment, time_units, timestamp=timestamp))

    def future_timestring(self, time_increment, time_units, timestamp=None):
        """Return the ISO-8601 representation of the future date/time.

        This function adds the requested time increment to the current date/time to get a future date/time
        and returns the ISO-8601 formatted string for that date/time. It does not update the internal
        timestamp.

        Parameters
        ----------
        time_increment : float
            The increment to adjust the current time.
        time_units : str
            The time unit for the increment value.
        timestamp : float, optional
            An alternative timestamp to apply the time increment to.

        Returns
        -------
        str
            The future date/time in ISO-8601.
        """
        return self.future_datetime(time_increment, time_units, timestamp=timestamp).isoformat()

    def time_since_given(self, timestamp):
        """Return the elapsed time (seconds).

        This function takes the given timestamp and calculates the elapsed time in seconds
        between it and the initial timestamp in the handler.

        Parameters
        ----------
        timestamp : float
            A UNIX timestamp

        Returns
        -------
        float
            The elapsed time (seconds) between the given
        """
        dt = datetime.utcfromtimestamp(timestamp)
        return self._time_difference(dt, self.initial_dt)

    def time_since_given_datetime(self, given_datetime, reverse=False):
        """Return the elapsed time (seconds).

        This function takes a given datetime object and calculates the elapsed time in seconds
        between it and the initial timestamp in the handler. If the given datetime is prior to
        the initial timestamp in the handler, use the reverse flag.

        Parameters
        ----------
        given_datetime : datetime
            The given timestamp.
        reverse : bool, optional
            Flag to make the difference in reverse. Default is False.

        Returns
        -------
        float
            The elapsed time (seconds) between the given timestamp and the initial timestamp
        """
        if reverse:
            return self._time_difference(self.initial_dt, given_datetime)
        else:
            return self._time_difference(given_datetime, self.initial_dt)
