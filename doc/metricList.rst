=================
Available metrics
=================
Core LSST MAF metrics
=====================
 
- `AbsMaxMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.simpleMetrics.AbsMaxMetric>`_ 
 	 Calculate the max of the absolute value of a simData column slice.
- `AbsMaxPercentMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.simpleMetrics.AbsMaxPercentMetric>`_ 
 	 Return the percent of the data which has the absolute value of the max value of the data.
- `AbsMeanMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.simpleMetrics.AbsMeanMetric>`_ 
 	 Calculate the mean of the absolute value of a simData column slice.
- `AbsMedianMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.simpleMetrics.AbsMedianMetric>`_ 
 	 Calculate the median of the absolute value of a simData column slice.
- `AccumulateCountMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.vectorMetrics.AccumulateCountMetric>`_ 
 	 Calculate the accumulated stat
- `AccumulateM5Metric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.vectorMetrics.AccumulateM5Metric>`_ 
 	 Calculate the accumulated stat
- `AccumulateMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.vectorMetrics.AccumulateMetric>`_ 
 	 Calculate the accumulated stat
- `AccumulateUniformityMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.vectorMetrics.AccumulateUniformityMetric>`_ 
 	 Make a 2D version of UniformityMetric
- `ActivityOverPeriodMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.moMetrics.ActivityOverPeriodMetric>`_ 
 	 Count fraction of object period we could identify activity for an SSobject.
- `ActivityOverTimeMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.moMetrics.ActivityOverTimeMetric>`_ 
 	 Count fraction of survey we could identify activity for an SSobject.
- `AreaSummaryMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.areaSummaryMetrics.AreaSummaryMetric>`_ 
 	 Find the min/max of a value in the best area. This is a handy substitute for when
- `AveSlewFracMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.slewMetrics.AveSlewFracMetric>`_ 
 	 Base class for the metrics.
- `BaseMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.baseMetric.BaseMetric>`_ 
 	 Base class for the metrics.
- `BaseMoMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.moMetrics.BaseMoMetric>`_ 
 	 Base class for the moving object metrics.
- `BinaryMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.simpleMetrics.BinaryMetric>`_ 
 	 Return 1 if there is data. 
- `BruteOSFMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.technicalMetrics.BruteOSFMetric>`_ 
 	 Assume I can't trust the slewtime or visittime colums.
- `CampaignLengthMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.seasonMetrics.CampaignLengthMetric>`_ 
 	 Calculate the number of seasons (roughly, years) a pointing is observed for.
- `ChipVendorMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.chipVendorMetric.ChipVendorMetric>`_ 
 	 See what happens if we have chips from different vendors
- `Coaddm5Metric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.simpleMetrics.Coaddm5Metric>`_ 
 	 Calculate the coadded m5 value at this gridpoint.
- `Color_AsteroidMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.moMetrics.Color_AsteroidMetric>`_ 
 	 This metric is appropriate for MBAs and NEOs, and other inner solar system objects.
- `CompletenessMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.technicalMetrics.CompletenessMetric>`_ 
 	 Compute the completeness and joint completeness 
- `CountExplimMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.simpleMetrics.CountExplimMetric>`_ 
 	 Count the number of x second visits.  Useful for rejecting very short exposures
- `CountMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.simpleMetrics.CountMetric>`_ 
 	 Count the length of a simData column slice. 
- `CountRatioMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.simpleMetrics.CountRatioMetric>`_ 
 	 Count the length of a simData column slice, then divide by 'normVal'. 
- `CountSubsetMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.simpleMetrics.CountSubsetMetric>`_ 
 	 Count the length of a simData column slice which matches 'subset'. 
- `CountUniqueMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.simpleMetrics.CountUniqueMetric>`_ 
 	 Return the number of unique values.
- `CrowdingM5Metric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.crowdingMetric.CrowdingM5Metric>`_ 
 	 Return the magnitude at which the photometric error exceeds crowding_error threshold.
- `CrowdingMagUncertMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.crowdingMetric.CrowdingMagUncertMetric>`_ 
 	 Given a stellar magnitude, calculate the mean uncertainty on the magnitude from crowding.
- `DcrPrecisionMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.dcrMetric.DcrPrecisionMetric>`_ 
 	 Determine how precise a DCR correction could be made
- `DiscoveryMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.moMetrics.DiscoveryMetric>`_ 
 	 Identify the discovery opportunities for an SSobject.
- `Discovery_DistanceMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.moMetrics.Discovery_DistanceMetric>`_ 
 	 Returns the distance of the i-th discovery track of an SSobject.
- `Discovery_EcLonLatMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.moMetrics.Discovery_EcLonLatMetric>`_ 
 	 Returns the ecliptic lon/lat and solar elong of the i-th discovery track of an SSobject.
- `Discovery_N_ChancesMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.moMetrics.Discovery_N_ChancesMetric>`_ 
 	 Calculate total number of discovery opportunities for an SSobject.
- `Discovery_N_ObsMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.moMetrics.Discovery_N_ObsMetric>`_ 
 	 Calculates the number of observations in the i-th discovery track of an SSobject.
- `Discovery_RADecMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.moMetrics.Discovery_RADecMetric>`_ 
 	 Returns the RA/Dec of the i-th discovery track of an SSobject.
- `Discovery_TimeMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.moMetrics.Discovery_TimeMetric>`_ 
 	 Returns the time of the i-th discovery track of an SSobject.
- `Discovery_VelocityMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.moMetrics.Discovery_VelocityMetric>`_ 
 	 Returns the sky velocity of the i-th discovery track of an SSobject.
- `ExgalM5 <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.exgalM5.ExgalM5>`_ 
 	 Calculate co-added five-sigma limiting depth after dust extinction.
- `ExgalM5_with_cuts <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.weakLensingSystematicsMetric.ExgalM5_with_cuts>`_ 
 	 Calculate co-added five-sigma limiting depth, but apply dust extinction and depth cuts.
- `FftMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.fftMetric.FftMetric>`_ 
 	 Calculate a truncated FFT of the exposure times.
- `FilterColorsMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.technicalMetrics.FilterColorsMetric>`_ 
 	 Calculate an RGBA value that accounts for the filters used up to time t0.
- `FracAboveMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.simpleMetrics.FracAboveMetric>`_ 
 	 Find the fraction of data values above a given value.
- `FracBelowMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.simpleMetrics.FracBelowMetric>`_ 
 	 Find the fraction of data values below a given value.
- `FullRangeAngleMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.simpleMetrics.FullRangeAngleMetric>`_ 
 	 Calculate the full range of an angular (degrees) simData column slice.
- `FullRangeMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.simpleMetrics.FullRangeMetric>`_ 
 	 Calculate the range of a simData column slice.
- `HighVelocityMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.moMetrics.HighVelocityMetric>`_ 
 	 Count number of times an SSobject appears trailed.
- `HighVelocityNightsMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.moMetrics.HighVelocityNightsMetric>`_ 
 	 Count the number of discovery opportunities (via trailing) for an SSobject.
- `HistogramM5Metric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.vectorMetrics.HistogramM5Metric>`_ 
 	 Calculate the coadded depth for each bin (e.g., per night).
- `HistogramMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.vectorMetrics.HistogramMetric>`_ 
 	 A wrapper to stats.binned_statistic
- `HourglassMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.hourglassMetric.HourglassMetric>`_ 
 	 Plot the filters used as a function of time. Must be used with the Hourglass Slicer.
- `IdentityMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.summaryMetrics.IdentityMetric>`_ 
 	 Return the metric value itself .. this is primarily useful as a summary statistic for UniSlicer metrics.
- `InstantaneousColorMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.moMetrics.InstantaneousColorMetric>`_ 
 	 Identify SSobjects which could have observations suitable to determine colors.
- `InterNightGapsMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.cadenceMetrics.InterNightGapsMetric>`_ 
 	 Calculate the gap between consecutive observations in different nights, in days.
- `IntraNightGapsMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.cadenceMetrics.IntraNightGapsMetric>`_ 
 	 Calculate the gap between consecutive observations within a night, in hours.
- `KnownObjectsMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.moMetrics.KnownObjectsMetric>`_ 
 	 Identify SSobjects which could be classified as 'previously known' based on their peak V magnitude.
- `KuiperMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.kuiperMetrics.KuiperMetric>`_ 
 	 Find the Kuiper V statistic for a distribution, useful for angles.
- `LightcurveColor_OuterMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.moMetrics.LightcurveColor_OuterMetric>`_ 
 	 This metric is appropriate for outer solar system objects, such as TNOs and SDOs.
- `LightcurveInversion_AsteroidMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.moMetrics.LightcurveInversion_AsteroidMetric>`_ 
 	 This metric is generally applicable to NEOs and MBAs - inner solar system objects.
- `LongGapAGNMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.longGapAGNMetric.LongGapAGNMetric>`_ 
 	 max delta-t and average of the top-10 longest gaps.
- `MagicDiscoveryMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.moMetrics.MagicDiscoveryMetric>`_ 
 	 Count the number of discovery opportunities with very good software for an SSobject.
- `MaxGapMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.tgaps.MaxGapMetric>`_ 
 	 Find the maximum gap in between observations.
- `MaxMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.simpleMetrics.MaxMetric>`_ 
 	 Calculate the maximum of a simData column slice.
- `MaxPercentMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.simpleMetrics.MaxPercentMetric>`_ 
 	 Return the percent of the data which has the maximum value.
- `MaxStateChangesWithinMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.technicalMetrics.MaxStateChangesWithinMetric>`_ 
 	 Compute the maximum number of changes of state that occur within a given timespan.
- `MeanAngleMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.simpleMetrics.MeanAngleMetric>`_ 
 	 Calculate the mean of an angular (degree) simData column slice.
- `MeanCampaignFrequencyMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.seasonMetrics.MeanCampaignFrequencyMetric>`_ 
 	 Calculate the mean separation between nights, within a season - then the mean over the campaign.
- `MeanMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.simpleMetrics.MeanMetric>`_ 
 	 Calculate the mean of a simData column slice.
- `MeanValueAtHMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.moSummaryMetrics.MeanValueAtHMetric>`_ 
 	 Return the mean value of a metric at a given H.
- `MedianMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.simpleMetrics.MedianMetric>`_ 
 	 Calculate the median of a simData column slice.
- `MetricRegistry <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.baseMetric.MetricRegistry>`_ 
 	 Meta class for metrics, to build a registry of metric classes.
- `MinMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.simpleMetrics.MinMetric>`_ 
 	 Calculate the minimum of a simData column slice.
- `MinTimeBetweenStatesMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.technicalMetrics.MinTimeBetweenStatesMetric>`_ 
 	 Compute the minimum time between changes of state in a column value.
- `MoCompletenessAtTimeMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.moSummaryMetrics.MoCompletenessAtTimeMetric>`_ 
 	 Calculate the completeness (relative to the entire population) <= a given H as a function of time,
- `MoCompletenessMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.moSummaryMetrics.MoCompletenessMetric>`_ 
 	 Calculate the fraction of the population that meets ``threshold`` value or higher.
- `NChangesMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.technicalMetrics.NChangesMetric>`_ 
 	 Compute the number of times a column value changes.
- `NNightsMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.moMetrics.NNightsMetric>`_ 
 	 Count the number of distinct nights an SSobject is observed.
- `NObsMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.moMetrics.NObsMetric>`_ 
 	 Count the total number of observations where an SSobject was 'visible'.
- `NObsNoSinglesMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.moMetrics.NObsNoSinglesMetric>`_ 
 	 Count the number of observations for an SSobject, without singles.
- `NRevisitsMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.cadenceMetrics.NRevisitsMetric>`_ 
 	 Calculate the number of consecutive visits with time differences less than dT.
- `NStateChangesFasterThanMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.technicalMetrics.NStateChangesFasterThanMetric>`_ 
 	 Compute the number of changes of state that happen faster than 'cutoff'.
- `NVisitsPerNightMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.tgaps.NVisitsPerNightMetric>`_ 
 	 Histogram the number of visits in each night.
- `NgalScaleMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.scalingMetrics.NgalScaleMetric>`_ 
 	 Approximate number of galaxies, scaled by median seeing.
- `NightPointingMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.nightPointingMetric.NightPointingMetric>`_ 
 	 Gather relevant information for a night to plot.
- `NightgapsMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.tgaps.NightgapsMetric>`_ 
 	 Histogram the number of nights between observations.
- `NlcPointsMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.scalingMetrics.NlcPointsMetric>`_ 
 	 Number of points in stellar light curves
- `NormalizeMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.summaryMetrics.NormalizeMetric>`_ 
 	 Return a metric values divided by 'normVal'. Useful for turning summary statistics into fractions.
- `NoutliersNsigmaMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.simpleMetrics.NoutliersNsigmaMetric>`_ 
 	 Calculate the # of visits less than nSigma below the mean (nSigma<0) or
- `NstarsMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.crowdingMetric.NstarsMetric>`_ 
 	 Return the number of stars visible above some uncertainty limit,
- `ObsArcMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.moMetrics.ObsArcMetric>`_ 
 	 Calculate the difference between the first and last observation of an SSobject.
- `OpenShutterFractionMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.technicalMetrics.OpenShutterFractionMetric>`_ 
 	 Compute the fraction of time the shutter is open compared to the total time spent observing.
- `OptimalM5Metric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.optimalM5Metric.OptimalM5Metric>`_ 
 	 Compare the co-added depth of the survey to one where
- `PairFractionMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.visitGroupsMetric.PairFractionMetric>`_ 
 	 What fraction of observations are part of a pair.
- `PairMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.pairMetric.PairMetric>`_ 
 	 Count the number of pairs that could be used for Solar System object detection
- `ParallaxCoverageMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.calibrationMetrics.ParallaxCoverageMetric>`_ 
 	 Check how well the parallax factor is distributed. Subtracts the weighted mean position of the
- `ParallaxDcrDegenMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.calibrationMetrics.ParallaxDcrDegenMetric>`_ 
 	 Use the full parallax and DCR displacement vectors to find if they are degenerate.
- `ParallaxMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.calibrationMetrics.ParallaxMetric>`_ 
 	 Calculate the uncertainty in a parallax measurement given a series of observations.
- `PassMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.simpleMetrics.PassMetric>`_ 
 	 Just pass the entire array through
- `PeakVMagMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.moMetrics.PeakVMagMetric>`_ 
 	 Pull out the peak V magnitude of all observations of the SSobject.
- `PercentileMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.simpleMetrics.PercentileMetric>`_ 
 	 Find the value of a column at a given percentile.
- `PeriodicDetectMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.periodicDetectMetric.PeriodicDetectMetric>`_ 
 	 Determine if we would be able to classify an object as periodic/non-uniform, using an F-test
- `PeriodicQualityMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.phaseGapMetric.PeriodicQualityMetric>`_ 
 	 Base class for the metrics.
- `PhaseGapMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.phaseGapMetric.PhaseGapMetric>`_ 
 	 Measure the maximum gap in phase coverage for observations of periodic variables.
- `ProperMotionMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.calibrationMetrics.ProperMotionMetric>`_ 
 	 Calculate the uncertainty in the returned proper motion.
- `RadiusObsMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.calibrationMetrics.RadiusObsMetric>`_ 
 	 find the radius in the focal plane. returns things in degrees.
- `RapidRevisitMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.cadenceMetrics.RapidRevisitMetric>`_ 
 	 Base class for the metrics.
- `RapidRevisitUniformityMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.cadenceMetrics.RapidRevisitUniformityMetric>`_ 
 	 Calculate uniformity of time between consecutive visits on short timescales (for RAV1).
- `RmsAngleMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.simpleMetrics.RmsAngleMetric>`_ 
 	 Calculate the standard deviation of an angular (degrees) simData column slice.
- `RmsMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.simpleMetrics.RmsMetric>`_ 
 	 Calculate the standard deviation of a simData column slice.
- `RobustRmsMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.simpleMetrics.RobustRmsMetric>`_ 
 	 Use the inter-quartile range of the data to estimate the RMS.  
- `SNCadenceMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.snCadenceMetric.SNCadenceMetric>`_ 
 	 Metric to estimate the redshift limit for faint supernovae (x1,color) = (-2.0,0.2)
- `SNNSNMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.snNSNMetric.SNNSNMetric>`_ 
 	 Estimate (nSN,zlim) of type Ia supernovae.
- `SNSLMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.snSLMetric.SNSLMetric>`_ 
 	 Calculate  the number of expected well-measured strongly lensed SN (per dataslice).
- `SNSNRMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.snSNRMetric.SNSNRMetric>`_ 
 	 Metric to estimate the detection rate for faint supernovae (x1,color) = (-2.0,0.2)
- `SeasonLengthMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.seasonMetrics.SeasonLengthMetric>`_ 
 	 Calculate the length of LSST seasons, in days.
- `SlewContributionMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.slewMetrics.SlewContributionMetric>`_ 
 	 Base class for the metrics.
- `StarDensityMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.starDensity.StarDensityMetric>`_ 
 	 Interpolate the stellar luminosity function to return the number of
- `StaticProbesFoMEmulatorMetricSimple <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.summaryMetrics.StaticProbesFoMEmulatorMetricSimple>`_ 
 	 This calculates the Figure of Merit for the combined
- `StringCountMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.stringCountMetric.StringCountMetric>`_ 
 	 Count up the number of times each string appears in a column.
- `SumMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.simpleMetrics.SumMetric>`_ 
 	 Calculate the sum of a simData column slice.
- `TableFractionMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.summaryMetrics.TableFractionMetric>`_ 
 	 Count the completeness (for many fields) and summarize how many fields have given completeness levels
- `TdcMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.seasonMetrics.TdcMetric>`_ 
 	 Calculate the Time Delay Challenge metric, as described in Liao et al 2015
- `TeffMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.technicalMetrics.TeffMetric>`_ 
 	 Effective time equivalent for a given set of visits.
- `TemplateExistsMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.cadenceMetrics.TemplateExistsMetric>`_ 
 	 Calculate the fraction of images with a previous template image of desired quality.
- `TgapsMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.tgaps.TgapsMetric>`_ 
 	 Histogram the times of the gaps between observations.
- `TgapsPercentMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.tgaps.TgapsPercentMetric>`_ 
 	 Compute the fraction of the time gaps between observations that occur in a given time range.
- `TotalPowerMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.summaryMetrics.TotalPowerMetric>`_ 
 	 Calculate the total power in the angular power spectrum between lmin/lmax.
- `TransientMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.transientMetrics.TransientMetric>`_ 
 	 Calculate what fraction of the transients would be detected. Best paired with a spatial slicer.
- `UniformityMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.cadenceMetrics.UniformityMetric>`_ 
 	 Calculate how uniformly the observations are spaced in time.
- `UniqueRatioMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.simpleMetrics.UniqueRatioMetric>`_ 
 	 Return the number of unique values divided by the total number of values.
- `UseMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.useMetrics.UseMetric>`_ 
 	 Metric to classify visits by type of visits
- `ValueAtHMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.moSummaryMetrics.ValueAtHMetric>`_ 
 	 Return the metric value at a given H value.
- `VisitGapMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.cadenceMetrics.VisitGapMetric>`_ 
 	 Calculate the gap between any consecutive observations, in hours, regardless of night boundaries.
- `VisitGroupsMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.visitGroupsMetric.VisitGroupsMetric>`_ 
 	 Count the number of visits per night within deltaTmin and deltaTmax.
- `WeakLensingNvisits <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.weakLensingSystematicsMetric.WeakLensingNvisits>`_ 
 	 A proxy metric for WL systematics. Higher values indicate better systematics mitigation.
- `YearCoverageMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.coverageMetric.YearCoverageMetric>`_ 
 	 Count the number of bins covered by nightCol -- default bins are 'years'.
- `ZeropointMetric <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.summaryMetrics.ZeropointMetric>`_ 
 	 Return a metric values with the addition of 'zp'. Useful for altering the zeropoint for summary statistics.
- `fOArea <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.summaryMetrics.fOArea>`_ 
 	 Metrics based on a specified number of visits, but returning AREA related to Nvisits:
- `fONv <source/rubin_sim.maf.metrics.html#rubin_sim.maf.metrics.summaryMetrics.fONv>`_ 
 	 Metrics based on a specified area, but returning NVISITS related to area:
 
Contributed mafContrib metrics
==============================
 
- `AngularSpreadMetric <source/rubin_sim.maf.mafContrib.html#rubin_sim.maf.mafContrib.angularSpread.AngularSpreadMetric>`_ 
 	 Compute the angular spread statistic which measures uniformity of a distribution angles accounting for 2pi periodicity.
- `GRBTransientMetric <source/rubin_sim.maf.mafContrib.html#rubin_sim.maf.mafContrib.GRBTransientMetric.GRBTransientMetric>`_ 
 	 Detections for on-axis GRB afterglows decaying as 
- `GW170817DetMetric <source/rubin_sim.maf.mafContrib.html#rubin_sim.maf.mafContrib.GW170817DetMetric.GW170817DetMetric>`_ 
 	 Wrapper metric class for GW170817-like kilonovae based on the
- `GalaxyCountsMetric <source/rubin_sim.maf.mafContrib.html#rubin_sim.maf.mafContrib.lssMetrics.GalaxyCountsMetric>`_ 
 	 Estimate the number of galaxies expected at a particular coadded depth.
- `KNePopMetric <source/rubin_sim.maf.mafContrib.html#rubin_sim.maf.mafContrib.kneMetrics.KNePopMetric>`_ 
 	 Base class for the metrics.
- `MicrolensingMetric <source/rubin_sim.maf.mafContrib.html#rubin_sim.maf.mafContrib.microlensingMetric.MicrolensingMetric>`_ 
 	 Quantifies detectability of Microlensing events.
- `NumObsMetric <source/rubin_sim.maf.mafContrib.html#rubin_sim.maf.mafContrib.LSSObsStrategy.numObsMetric.NumObsMetric>`_ 
 	 Calculate the number of observations per data slice.
- `PeriodDeviationMetric <source/rubin_sim.maf.mafContrib.html#rubin_sim.maf.mafContrib.varMetrics.PeriodDeviationMetric>`_ 
 	 Measure the percentage deviation of recovered periods for pure sine wave variability (in magnitude).
- `PeriodicMetric <source/rubin_sim.maf.mafContrib.html#rubin_sim.maf.mafContrib.PeriodicMetric.PeriodicMetric>`_ 
 	 From a set of observation times, uses code provided by Robert Siverd (LCOGT) to calculate the spectral window function.
- `PeriodicStarMetric <source/rubin_sim.maf.mafContrib.html#rubin_sim.maf.mafContrib.periodicStarMetric.PeriodicStarMetric>`_ 
 	 At each slicePoint, run a Monte Carlo simulation to see how well a periodic source can be fit.
- `RelRmsMetric <source/rubin_sim.maf.mafContrib.html#rubin_sim.maf.mafContrib.photPrecMetrics.RelRmsMetric>`_ 
 	 Relative scatter metric (RMS over median).
- `SEDSNMetric <source/rubin_sim.maf.mafContrib.html#rubin_sim.maf.mafContrib.photPrecMetrics.SEDSNMetric>`_ 
 	 Computes the S/Ns for a given SED.
- `SNMetric <source/rubin_sim.maf.mafContrib.html#rubin_sim.maf.mafContrib.photPrecMetrics.SNMetric>`_ 
 	 Calculate the signal to noise metric in a given filter for an object of a given magnitude.
- `StarCountMassMetric <source/rubin_sim.maf.mafContrib.html#rubin_sim.maf.mafContrib.StarCountMassMetric.StarCountMassMetric>`_ 
 	 Find the number of stars in a given field in the mass range fainter than magnitude 16 and bright enough to have noise less than 0.03 in a given band. M1 and M2 are the upper and lower limits of the mass range. 'band' is the band to be observed.
- `StarCountMetric <source/rubin_sim.maf.mafContrib.html#rubin_sim.maf.mafContrib.StarCountMetric.StarCountMetric>`_ 
 	 Find the number of stars in a given field between D1 and D2 in parsecs.
- `StaticProbesFoMEmulatorMetric <source/rubin_sim.maf.mafContrib.html#rubin_sim.maf.mafContrib.StaticProbesFoMSummaryMetric.StaticProbesFoMEmulatorMetric>`_ 
 	 This calculates the Figure of Merit for the combined
- `TdePopMetric <source/rubin_sim.maf.mafContrib.html#rubin_sim.maf.mafContrib.TDEsPopMetric.TdePopMetric>`_ 
 	 Base class for the metrics.
- `ThreshSEDSNMetric <source/rubin_sim.maf.mafContrib.html#rubin_sim.maf.mafContrib.photPrecMetrics.ThreshSEDSNMetric>`_ 
 	 Computes the metric whether the S/N is bigger than the threshold in all the bands for a given SED
- `TripletBandMetric <source/rubin_sim.maf.mafContrib.html#rubin_sim.maf.mafContrib.TripletMetric.TripletBandMetric>`_ 
 	 Find the number of 'triplets' of three images taken in the same band, based on user-selected minimum and maximum intervals (in hours),
- `TripletMetric <source/rubin_sim.maf.mafContrib.html#rubin_sim.maf.mafContrib.TripletMetric.TripletMetric>`_ 
 	 Find the number of 'triplets' of three images taken in any band, based on user-selected minimum and maximum intervals (in hours),
 
