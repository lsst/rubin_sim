import os
import warnings


def get_oorb_data_dir():
    """Find where the oorb files should be installed
    """
    data_path = os.getenv('OORB_DATA')
    if data_path is None:
        # See if we are in a conda enviroment and can find it
        conda_dir = os.getenv('CONDA_PREFIX')
        if conda_dir is not None:
            data_path = os.path.join(conda_dir, 'share/openorb')
            if not os.path.isdir(data_path):
                data_path = None
    if data_path is None:
        warnings.warn('Failed to find path for oorb data files. No $OORB_DATA enviroement variable set, and they are not it usual conda spot')
    return data_path
