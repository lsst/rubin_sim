from setuptools import setup

setup(name='rubin_sim',
      version='0.01',
      scripts=['bin/rs_dowload_data', 'bin/maf/addRun', 'bin/maf/glance_dir',
               'bin/maf/maf_night_report', 'bin/maf/scimaf_dir', 'bin/maf/showMaf'],
      packages=['rubin_sim', ],
      )
