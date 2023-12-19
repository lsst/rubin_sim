# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from documenteer.conf.guide import *  # noqa: F403, import *

linkcheck_retries = 2

from .metric_list import make_metric_list

make_metric_list("maf-metric-list.rst")
