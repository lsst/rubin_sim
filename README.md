# rubin_sim
Scheduler, survey strategy analysis, and other simulation tools for Rubin Observatory.


[![pypi](https://img.shields.io/pypi/v/rubin-sim.svg)](https://pypi.org/project/rubin-sim/)
 [![Conda Version](https://img.shields.io/conda/vn/conda-forge/rubin-sim.svg)](https://anaconda.org/conda-forge/rubin-sim) <br>
[![Run CI](https://github.com/lsst/rubin_sim/actions/workflows/test_and_build.yaml/badge.svg)](https://github.com/lsst/rubin_sim/actions/workflows/test_and_build.yaml)
[![Build and Upload Docs](https://github.com/lsst/rubin_sim/actions/workflows/build_docs.yaml/badge.svg)](https://github.com/lsst/rubin_sim/actions/workflows/build_docs.yaml)
[![codecov](https://codecov.io/gh/lsst/rubin_sim/branch/main/graph/badge.svg?token=2BUBL8R9RH)](https://codecov.io/gh/lsst/rubin_sim)


[![DOI](https://zenodo.org/badge/365031715.svg)](https://zenodo.org/badge/latestdoi/365031715)


## rubin_sim ## 

The [Legacy Survey of Space and Time](http://www.lsst.org) (LSST)
is anticipated to encompass around 2 million observations spanning a decade,
averaging 800 visits per night. The `rubin_sim` package was built to help
understand the predicted performance of the LSST.

The `rubin_sim` package contains the following main modules: 
* `phot_utils` - provides synthetic photometry
using provided throughput curves based on current predicted performance.
* `skybrightness` incorporates the ESO
sky model, modified to match measured sky conditions at the LSST site,
including an addition of a model for twilight skybrightness. This is used
to generate the pre-calculated skybrightness data used in
[`rubin_scheduler.skybrightness_pre`](https://rubin-scheduler.lsst.io/skybrightness-pre.html).
* `moving_objects` provides a way to generate
synthetic observations of moving objects, based on how they would appear in
pointing databases ("opsims") created by
[`rubin_scheduler`](https://rubin-scheduler.lsst.io).
* `maf` the Metrics Analysis Framework, enabling efficient and
scientifically varied evaluation of the LSST survey strategy and progress
by providing a framework to enable these metrics to run in a
standardized way on opsim outputs.

More documentation for `rubin_sim` is available at 
[https://rubin-sim.lsst.io](https://rubin-sim.lsst.io), including installation instructions. 

### Getting Help ###

Questions about `rubin_sim` can be posted on the [sims slack channel](https://lsstc.slack.com/archives/C2LQ5JW9W), or on https://community.lsst.org/c/sci/survey_strategy/ (optionally, tag @yoachim and/or @ljones so we get notifications about it).
