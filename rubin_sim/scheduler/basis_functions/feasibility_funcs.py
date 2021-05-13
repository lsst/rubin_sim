import numpy as np
from rubin_sim.scheduler import features
import matplotlib.pylab as plt
from rubin_sim.scheduler.basis_functions import Base_basis_function
from rubin_sim.scheduler.utils import int_rounded


__all__ = ['Filter_loaded_basis_function', 'Time_to_twilight_basis_function',
           'Not_twilight_basis_function', 'Force_delay_basis_function',
           'Hour_Angle_limit_basis_function', 'Moon_down_basis_function',
           'Fraction_of_obs_basis_function', 'Clouded_out_basis_function',
           'Rising_more_basis_function', 'Soft_delay_basis_function',
           'Look_ahead_ddf_basis_function', 'Sun_alt_limit_basis_function',
           'Time_in_twilight_basis_function', 'Night_modulo_basis_function',
           'End_of_evening_basis_function', 'Time_to_scheduled_basis_function',
           'Limit_obs_pnight_basis_function']


class Filter_loaded_basis_function(Base_basis_function):
    """Check that the filter(s) needed are loaded

    Parameters
    ----------
    filternames : str or list of str
        The filternames that need to be mounted to execute.
    """
    def __init__(self, filternames='r'):
        super(Filter_loaded_basis_function, self).__init__()
        if type(filternames) is not list:
            filternames = [filternames]
        self.filternames = filternames

    def check_feasibility(self, conditions):

        for filtername in self.filternames:
            result = filtername in conditions.mounted_filters
            if result is False:
                return result
        return result


class Limit_obs_pnight_basis_function(Base_basis_function):
    """
    """
    def __init__(self, survey_str='', nlimit=100.):
        super(Limit_obs_pnight_basis_function, self).__init__()
        self.nlimit = nlimit
        self.survey_features['N_in_night'] = features.Survey_in_night(survey_str=survey_str)

    def check_feasibility(self, conditions):
        if self.survey_features['N_in_night'].feature >= self.nlimit:
            return False
        else:
            return True


class Night_modulo_basis_function(Base_basis_function):
    """Only return true on certain nights
    """
    def __init__(self, pattern=None):
        super(Night_modulo_basis_function, self).__init__()
        if pattern is None:
            pattern = [True, False]
        self.pattern = pattern
        self.mod_val = len(self.pattern)

    def check_feasibility(self, conditions):
        indx = int(conditions.night % self.mod_val)
        result = self.pattern[indx]
        return result


class Time_in_twilight_basis_function(Base_basis_function):
    """Make sure there is some time left in twilight.

    Parameters
    ----------
    time_needed : float (5)
        The time needed remaining in twilight (minutes)
    """
    def __init__(self, time_needed=5.):
        super(Time_in_twilight_basis_function, self).__init__()
        self.time_needed = time_needed/60./24.  # To days

    def check_feasibility(self, conditions):
        result = False
        time1 = conditions.sun_n18_setting - conditions.mjd
        time2 = conditions.sun_n12_rising - conditions.mjd

        if time1 > self.time_needed:
            result = True
        else:
            if conditions.sunAlt > np.radians(-18.):
                if time2 > self.time_needed:
                    result = True
        return result


class End_of_evening_basis_function(Base_basis_function):
    """Only let observations happen in a limited time before twilight
    """
    def __init__(self, time_remaining=30., alt_limit=18):
        super(End_of_evening_basis_function, self).__init__()
        self.time_remaining = int_rounded(time_remaining/60./24.)
        self.alt_limit = str(alt_limit)

    def check_feasibility(self, conditions):
        available_time = getattr(conditions, 'sun_n' + self.alt_limit + '_rising') - conditions.mjd
        result = int_rounded(available_time) < self.time_remaining
        return result


class Time_to_twilight_basis_function(Base_basis_function):
    """Make sure there is enough time before twilight. Useful
    if you want to check before starting a long sequence of observations.

    Parameters
    ----------
    time_needed : float (30.)
        The time needed to run a survey (mintues).
    alt_limit : int (18)
        The sun altitude limit to use. Must be 12 or 18
    """
    def __init__(self, time_needed=30., alt_limit=18):
        super(Time_to_twilight_basis_function, self).__init__()
        self.time_needed = time_needed/60./24.  # To days
        self.alt_limit = str(alt_limit)

    def check_feasibility(self, conditions):
        available_time = getattr(conditions, 'sun_n' + self.alt_limit + '_rising') - conditions.mjd
        result = available_time > self.time_needed
        return result


class Time_to_scheduled_basis_function(Base_basis_function):
    """Make sure there is enough time before next scheduled observation. Useful
    if you want to check before starting a long sequence of observations.

    Parameters
    ----------
    time_needed : float (30.)
        The time needed to run a survey (mintues).
    """
    def __init__(self, time_needed=30.):
        super(Time_to_scheduled_basis_function, self).__init__()
        self.time_needed = time_needed/60./24.  # To days

    def check_feasibility(self, conditions):
        if len(conditions.scheduled_observations) == 0:
            return True

        available_time = np.min(conditions.scheduled_observations) - conditions.mjd
        result = available_time > self.time_needed
        return result


class Not_twilight_basis_function(Base_basis_function):
    def __init__(self, sun_alt_limit=-18):
        """
        # Should be -18 or -12
        """
        self.sun_alt_limit = str(int(sun_alt_limit)).replace('-', 'n')
        super(Not_twilight_basis_function, self).__init__()

    def check_feasibility(self, conditions):
        result = True
        if conditions.mjd < getattr(conditions, 'sun_'+self.sun_alt_limit+'_setting'):
            result = False
        if conditions.mjd > getattr(conditions, 'sun_'+self.sun_alt_limit+'_rising'):
            result = False
        return result


class Force_delay_basis_function(Base_basis_function):
    """Keep a survey from executing to rapidly.

    Parameters
    ----------
    days_delay : float (2)
        The number of days to force a gap on.
    """
    def __init__(self, days_delay=2., survey_name=None):
        super(Force_delay_basis_function, self).__init__()
        self.days_delay = days_delay
        self.survey_name = survey_name
        self.survey_features['last_obs_self'] = features.Last_observation(survey_name=self.survey_name)

    def check_feasibility(self, conditions):
        result = True
        if conditions.mjd - self.survey_features['last_obs_self'].feature['mjd'] < self.days_delay:
            result = False
        return result


class Soft_delay_basis_function(Base_basis_function):
    """Like Force_delay, but go ahead and let things catch up if they fall far behind.

    Parameters
    ----------

    """
    def __init__(self, fractions=[0.000, 0.009, 0.017], delays=[0., 0.5, 1.5], survey_name=None):
        if len(fractions) != len(delays):
            raise ValueError('fractions and delays must be same length')
        super(Soft_delay_basis_function, self).__init__()
        self.delays = delays
        self.survey_name = survey_name
        self.survey_features['last_obs_self'] = features.Last_observation(survey_name=self.survey_name)
        self.fractions = fractions
        self.survey_features['Ntot'] = features.N_obs_survey()
        self.survey_features['N_survey'] = features.N_obs_survey(note=self.survey_name)

    def check_feasibility(self, conditions):
        result = True
        current_ratio = self.survey_features['N_survey'].feature / self.survey_features['Ntot'].feature
        indx = np.searchsorted(self.fractions, current_ratio)
        if indx == len(self.fractions):
            indx -= 1
        delay = self.delays[indx]
        if conditions.mjd - self.survey_features['last_obs_self'].feature['mjd'] < delay:
            result = False
        return result


class Hour_Angle_limit_basis_function(Base_basis_function):
    """Only execute a survey in limited hour angle ranges. Useful for
    limiting Deep Drilling Fields.

    Parameters
    ----------
    RA : float (0.)
        RA of the target (degrees).
    ha_limits : list of lists
        limits for what hour angles are acceptable (hours). e.g.,
        to give 4 hour window around RA=0, ha_limits=[[22,24], [0,2]]
    """
    def __init__(self, RA=0., ha_limits=None):
        super(Hour_Angle_limit_basis_function, self).__init__()
        self.ra_hours = RA/360.*24.
        self.HA_limits = np.array(ha_limits)

    def check_feasibility(self, conditions):
        target_HA = (conditions.lmst - self.ra_hours) % 24
        # Are we in any of the possible windows
        result = False
        for limit in self.HA_limits:
            lres = limit[0] <= target_HA < limit[1]
            result = result or lres

        return result


class Moon_down_basis_function(Base_basis_function):
    """Demand the moon is down """
    def check_feasibility(self, conditions):
        result = True
        if conditions.moonAlt > 0:
            result = False
        return result


class Fraction_of_obs_basis_function(Base_basis_function):
    """Limit the fraction of all observations that can be labled a certain
    survey name. Useful for keeping DDFs from exceeding a given fraction of the
    total survey.

    Parameters
    ----------
    frac_total : float
        The fraction of total observations that can be of this survey
    survey_name : str
        The name of the survey
    """
    def __init__(self, frac_total, survey_name=''):
        super(Fraction_of_obs_basis_function, self).__init__()
        self.survey_name = survey_name
        self.frac_total = frac_total
        self.survey_features['Ntot'] = features.N_obs_survey()
        self.survey_features['N_survey'] = features.N_obs_survey(note=self.survey_name)

    def check_feasibility(self, conditions):
        # If nothing has been observed, fine to go
        result = True
        if self.survey_features['Ntot'].feature == 0:
            return result
        ratio = self.survey_features['N_survey'].feature / self.survey_features['Ntot'].feature
        if ratio > self.frac_total:
            result = False
        return result


class Look_ahead_ddf_basis_function(Base_basis_function):
    """Look into the future to decide if it's a good time to observe or block.

    Parameters
    ----------
    frac_total : float
        The fraction of total observations that can be of this survey
    aggressive_fraction : float
        If the fraction of observations drops below ths value, be more aggressive in scheduling.
        e.g., do not wait for conditions to improve, execute as soon as possible.
    time_needed : float (30.)
        Estimate of the amount of time needed to execute DDF sequence (minutes).
    RA : float (0.)
        The RA of the DDF
    ha_limits : list of lists (None)
        limits for what hour angles are acceptable (hours). e.g.,
        to give 4 hour window around HA=0, ha_limits=[[22,24], [0,2]]
    survey_name : str ('')
        The name of the survey
    time_jump : float (44.)
        The amount of time to assume will jump ahead if another survey executes (minutes)
    sun_alt_limit : float (-18.)
        The limit to assume twilight starts (degrees)
    """
    def __init__(self, frac_total, aggressive_fraction, time_needed=30., RA=0.,
                 ha_limits=None, survey_name='', time_jump=44., sun_alt_limit=-18.):
        super(Look_ahead_ddf_basis_function, self).__init__()
        if aggressive_fraction > frac_total:
            raise ValueError('aggressive_fraction should be less than frac_total')
        self.survey_name = survey_name
        self.frac_total = frac_total
        self.ra_hours = RA/360.*24.
        self.HA_limits = np.array(ha_limits)
        self.sun_alt_limit = str(int(sun_alt_limit)).replace('-', 'n')
        self.time_jump = time_jump / 60. / 24.  # To days
        self.time_needed = time_needed / 60. / 24.  # To days
        self.aggressive_fraction = aggressive_fraction
        self.survey_features['Ntot'] = features.N_obs_survey()
        self.survey_features['N_survey'] = features.N_obs_survey(note=self.survey_name)

    def check_feasibility(self, conditions):
        result = True
        target_HA = (conditions.lmst - self.ra_hours) % 24
        ratio = self.survey_features['N_survey'].feature / self.survey_features['Ntot'].feature
        available_time = getattr(conditions, 'sun_' + self.sun_alt_limit + '_rising') - conditions.mjd
        # If it's more that self.time_jump to hour angle zero
        # See if there will be enough time to twilight in the future
        if (int_rounded(target_HA) > int_rounded(12)) & (int_rounded(target_HA) < int_rounded(24.-self.time_jump)):
            if int_rounded(available_time) > int_rounded(self.time_needed + self.time_jump):
                result = False
                # If we paused for better conditions, but the moon will rise, turn things back on.
                if int_rounded(conditions.moonAlt) < int_rounded(0):
                    if int_rounded(conditions.moonrise) > int_rounded(conditions.mjd):
                        if int_rounded(conditions.moonrise - conditions.mjd) > int_rounded(self.time_jump):
                            result = True
        # If the moon is up and will set soon, pause
        if int_rounded(conditions.moonAlt) > int_rounded(0):
            time_after_moonset = getattr(conditions, 'sun_' + self.sun_alt_limit + '_rising') - conditions.moonset
            if int_rounded(conditions.moonset) > int_rounded(self.time_jump):
                if int_rounded(time_after_moonset) > int_rounded(self.time_needed):
                    result = False

        # If the survey has fallen far behind, be agressive and observe anytime it's up.
        if int_rounded(ratio) < int_rounded(self.aggressive_fraction):
            result = True
        return result


class Clouded_out_basis_function(Base_basis_function):
    def __init__(self, cloud_limit=0.7):
        super(Clouded_out_basis_function, self).__init__()
        self.cloud_limit = cloud_limit

    def check_feasibility(self, conditions):
        result = True
        if conditions.bulk_cloud > self.cloud_limit:
            result = False
        return result


class Rising_more_basis_function(Base_basis_function):
    """Say a spot is not available if it will rise substatially before twilight.

    Parameters
    ----------
    RA : float
        The RA of the point in the sky (degrees)
    pad : float
        When to start observations if there's plenty of time before twilight (minutes)
    """
    def __init__(self, RA, pad=30.):
        super(Rising_more_basis_function, self).__init__()
        self.RA_hours = RA * 24 / 360.
        self.pad = pad/60.  # To hours

    def check_feasibility(self, conditions):
        result = True
        hour_angle = conditions.lmst - self.RA_hours
        # If it's rising, and twilight is well beyond when it crosses the meridian
        time_to_twi = (conditions.sun_n18_rising - conditions.mjd)*24.
        if (hour_angle < -self.pad) & (np.abs(hour_angle) < (time_to_twi - self.pad)):
            result = False
        return result


class Sun_alt_limit_basis_function(Base_basis_function):
    """Don't try unless the sun is below some limit
    """

    def __init__(self, alt_limit=-12.1):
        super(Sun_alt_limit_basis_function, self).__init__()
        self.alt_limit = np.radians(alt_limit)

    def check_feasibility(self, conditions):
        result = True
        if conditions.sunAlt > self.alt_limit:
            result = False
        return result


## XXX--TODO:  Can include checks to see if downtime is coming, clouds are coming, moon rising, or surveys in a higher tier 
# Have observations they want to execute soon.
