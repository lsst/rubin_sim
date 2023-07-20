__all__ = ("FieldSurvey",)

import numpy as np

from rubin_sim.utils import ra_dec2_hpid

from .dd_surveys import DeepDrillingSurvey


class FieldSurvey(DeepDrillingSurvey):
    def __init__(
        self,
        basis_functions,
        RA,
        dec,
        sequence="rgizy",
        nvis=[20, 10, 20, 26, 20],
        exptime=30.0,
        u_exptime=30.0,
        nexp=2,
        ignore_obs=None,
        survey_name="DD",
        reward_value=None,
        readtime=2.0,
        filter_change_time=120.0,
        nside=None,
        flush_pad=30.0,
        seed=42,
        detailers=None,
    ):
        super().__init__(
            basis_functions=basis_functions,
            RA=RA,
            dec=dec,
            sequence=sequence,
            nvis=nvis,
            exptime=exptime,
            u_exptime=u_exptime,
            nexp=nexp,
            ignore_obs=ignore_obs,
            survey_name=survey_name,
            reward_value=reward_value,
            readtime=readtime,
            filter_change_time=filter_change_time,
            nside=nside,
            flush_pad=flush_pad,
            seed=seed,
            detailers=detailers,
        )
        self.basis_weights = np.ones(len(basis_functions)) / len(basis_functions)
        self.ra_deg, self.dec_deg = RA, dec

    def calc_reward_function(self, conditions):
        self.reward_checked = True
        indx = ra_dec2_hpid(self.nside, self.ra_deg, self.dec_deg)
        if self._check_feasibility(conditions):
            self.reward = 0
            for bf, weight in zip(self.basis_functions, self.basis_weights):
                basis_value = bf(conditions, indx=indx)
                self.reward += basis_value * weight

            if not np.isscalar(self.reward):
                self.reward = np.sum(self.reward[indx])

                if np.any(np.isinf(self.reward)):
                    self.reward = np.inf
        else:
            # If not feasable, negative infinity reward
            self.reward = -np.inf
            return self.reward

        return self.reward
