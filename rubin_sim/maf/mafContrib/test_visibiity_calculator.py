import numpy as np
from astropy.time import Time
import calculate_lsst_field_visibility_astropy

ras = np.arange(290.0, 295.0, 1.0)
decs = np.arange(-15.0, -10.0, 1.0)

t_start = '2019-10-17'
t_end = '2019-10-20'

(total_time_visible, hrs_visible_per_night) = calculate_lsst_field_visibility_astropy.calculate_lsst_field_visibility(ras,decs,t_start,t_end,verbose=True)

print(total_time_visible)
