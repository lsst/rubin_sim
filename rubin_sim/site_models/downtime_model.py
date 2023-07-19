__all__ = ("DowntimeModel",)

import warnings


class DowntimeModel:
    """Downtime estimates, both scheduled and unscheduled.

    Parameters
    ----------
    config: DowntimeModelConfig, optional
        A configuration class for the downtime model.
        This can be None, in which case the default DowntimeModelConfig is used.
        The user should set any non-default values for DowntimeModelConfig before
        configuration of the actual DowntimeModel.

    self.efd_requirements and self.target_requirements are also set.
    efd_requirements is a tuple: (list of str, float).
    This corresponds to the data columns required from the EFD and the amount of time history required.
    target_requirements is a list of str.
    This corresponds to the data columns required in the target dictionary passed when calculating the
    processed telemetry values.
    """

    def __init__(
        self,
        sched_down_col="scheduled_downtimes",
        unsched_down_col="unscheduled_downtimes",
        time_col="time",
    ):
        self.sched_down = sched_down_col
        self.unsched_down = unsched_down_col
        self.target_requirements = time_col

    def configure(self, config=None):
        warnings.warn("The configure method is deprecated.")

    def config_info(self):
        warnings.warn("The configure method is deprecated.")

    def __call__(self, efd_data, target_dict):
        """Calculate the sky coverage due to clouds.

        Parameters
        ----------
        efd_data: dict
            Dictionary of input telemetry, typically from the EFD.
            This must contain columns self.efd_requirements.
            (work in progress on handling time history).
        target_dict: dict
            Dictionary of target values over which to calculate the processed telemetry.
            (e.g. mapDict = {'ra': [], 'dec': [], 'altitude': [], 'azimuth': [], 'airmass': []})
            Here we use 'time', an astropy.time.Time, as we just need to know the time.

        Returns
        -------
        dict of bool, astropy.time.Time, astropy.time.Time
            Status of telescope (True = Down, False = Up) at time,
            time of expected end of downtime (~noon of the first available day),
            time of next scheduled downtime (~noon of the first available day).
        """
        # Check for downtime in scheduled downtimes.
        time = target_dict[self.target_requirements]
        next_start = efd_data[self.sched_down]["start"].searchsorted(time, side="right")
        next_end = efd_data[self.sched_down]["end"].searchsorted(time, side="right")
        if next_start > next_end:
            # Currently in a scheduled downtime.
            current_sched = efd_data[self.sched_down][next_end]
        else:
            # Not currently in a scheduled downtime.
            current_sched = None
        # This will be the next reported/expected downtime.
        next_sched = efd_data[self.sched_down][next_start]
        # Check for downtime in unscheduled downtimes.
        next_start = efd_data[self.unsched_down]["start"].searchsorted(time, side="right")
        next_end = efd_data[self.unsched_down]["end"].searchsorted(time, side="right")
        if next_start > next_end:
            # Currently in an unscheduled downtime.
            current_unsched = efd_data[self.unsched_down][next_end]
        else:
            current_unsched = None

        # Figure out what to report about current state.
        if current_sched is None and current_unsched is None:  # neither down
            status = False
            end_down = None
        else:  # we have a downtime from something ..
            if current_unsched is None:  # sched down only
                status = True
                end_down = current_sched["end"]
            elif current_sched is None:  # unsched down only
                status = True
                # should decide what to report on end of downtime here ..
                end_down = current_unsched["end"]
            else:  # both down ..
                status = True
                end_down = max(current_sched["end"], current_unsched["end"])
        return {"status": status, "end": end_down, "next": next_sched["start"]}
