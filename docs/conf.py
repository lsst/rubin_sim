# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from documenteer.conf.guide import *  # noqa: F403, import *

linkcheck_retries = 2

# Autodoc emits these annotation targets without an external inventory.
# The dataframe aliases have explicit schemas in clouds-api.rst.
nitpick_ignore = [
    ("py:class", "pandas.core.frame.DataFrame"),
    ("py:data", "typing.Any"),
]
