import numpy as np

class TauObsMetricData:
    """Object to capture the calculated metric values for four categories of
    variability, distinguished by their characteristic variability timescales,
    and the corresponding interval between sequential observations required
    to detect the variability"""

    def __init__(self):
        self.tau_obs = np.array([2.0, 20.0, 73.0, 365.0])
        self.metric_values = np.zeros(len(self.tau_obs))
