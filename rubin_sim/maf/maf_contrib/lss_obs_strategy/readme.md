This folder contains the code that was used to produce results in [Awan+2016](https://arxiv.org/abs/1605.00555) and [COSEP](https://arxiv.org/abs/1708.04058v1)(Section 9.2). The overarching idea is to look at the artificial structure induced by the LSST observing strategy, and quantify the mitigation of the induced artifacts by(large, i.e., as large as the LSST field-of-view) translational dithers.

`coaddM5Aanalysis.py` looks at the non-uniformity in the 5$\sigma$ (point-source) coadded depth; some characteristic features in these non-uniformities are explored using `almPlots.py` which looks at the specific features in the coadded depth maps that give rise to spurious power on the specified $\ell$-range. See more in Section 4.1 in [Awan+2016](https://arxiv.org/abs/1605.00555).

`artificialStructureCalculation.py` calculates the artificial structure (i.e., density fluctuations) resulting from the observing strategy. It takes the coadded depth in each HEALPix pixel and propagates them to galaxy number counts using empirical galaxy luminosity functions (using `galaxyCountsMetric_extended.py`). The script allows various options, including:
1. Magnitude cuts: implemented within the script.
2. Dust extinction: implemented within the script.
3. Border masking: using `maskingAlgorithmGeneralized.py`
4. Photometric calibration errors: using `numObsMetric.py` and `galaxyCounts_withPixelCalibration.py` (which modulates the galaxy number counts in each HEALPix pixel depending on the zero-point in that pixel).
5. Poisson noise: implemented within the script.

Options 1-4 were implemented for both [Awan+2016](https://arxiv.org/abs/1605.00555) and [COSEP](https://arxiv.org/abs/1708.04058v1).

`os_bias_analysis.py` calculates the Figure-of-Metric for our purposes, as discussed in Section 9.2 in [COSEP](https://arxiv.org/abs/1708.04058v1). The module requires outputs from `artificialStructureCalculation.py` and theoretical galaxy power spectra with BAO (which, for the purposes here, are from Hu Zhan).

----
Helper modules include
- `newDitherStackers.py` which contains the dither stackers (only those that are not already incorporated in [sims_maf](https://github.com/lsst/sims_maf/blob/master/python/lsst/sims/maf/stackers/ditherStackers.py). DEPRECATED.
- `plotBundleMaps.py` which contains the function for plotting skymaps, power spectra, and cartview plots.
- `saveBundleData_npzFormat.py` which contains the function for saving data as npz files. DEPRECATED.
- `constantsForPipeline.py` which contains variables used in the code, including plot colors and power law constants for the galaxy luminosity functions for different redshift bins (based on mock catalogs from Padilla+).