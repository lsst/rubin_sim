import os
from astropy.io import fits

def load_map_data(file_path):

    with fits.open(file_path) as hdul:
        map_data_table = hdul[1].data

    return map_data_table
