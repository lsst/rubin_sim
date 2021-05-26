from setuptools import setup

setup(name='rubin_sim',
      version='0.01',
      scripts=['bin/maf/addRun', 'bin/maf/glance_dir',
               'bin/maf/maf_night_report', 'bin/maf/scimaf_dir',
               'bin/maf/showMaf', 'bin/maf/generate_ss',
               'bin/maf/run_moving_calc', 'bin/maf/run_moving_fractions',
               'bin/maf/run_moving_join',
               'bin/movingObjects/makeLSSTobs', 'bin/movingObjects/addSeds',
               'bin/movingObjects/generateCoefficients',
               'bin/rs_download_data'],
      packages=['rubin_sim', ],
      )
