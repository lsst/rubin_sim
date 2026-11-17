# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from documenteer.conf.guide import *  # noqa: F403, import *

exclude_patterns = [*exclude_patterns, "issues/**"]  # noqa: F405

linkcheck_retries = 2
