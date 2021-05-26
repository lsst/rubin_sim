#################################################################################################
# Various things declared here to be imported when running artificialStructureCalculation.
# Makes the updates easier, since the constants/objects defined here are used by different
# functions, e.g. power law constants are called by GalaxyCountsMetric_extended as well as
# GalaxyCounts_withPixelCalibration. Similarly, plotColor dictionary helps maintain a consistent
# color scheme for different observing strategies.
#
# Included here: 
#  * power law constants from the i-band mock catalog; based on mocks from Padilla et al.
#  * normalization constant for galaxy counts from the mock catalogs.
#  * plotColor dictionary: colors for plotting results from different observing strategies.s
#
# Humna Awan
# humna.awan@rutgers.edu
#################################################################################################
#################################################################################################
# Power law constants for each z-bin based on N. D. Padilla et al.'s mock catalogs
# General power law form: 10**(a*m+b)
# Declare the dictionary for the power law constants
from collections import OrderedDict
powerLawConst_a = OrderedDict()
powerLawConst_b = OrderedDict()
# 0.<z<0.15
powerLawConst_a['0.<z<0.15'] = 0.166    # July 18 Catalog
powerLawConst_b['0.<z<0.15'] = -0.989    # July 18 Catalog

# 0.15<z<0.37
powerLawConst_a['0.15<z<0.37'] = 0.212     # July 18 Catalog
powerLawConst_b['0.15<z<0.37'] = -1.540    # July 18 Catalog

# 0.37<z<0.66
powerLawConst_a[' 0.37<z<0.66'] = 0.250     # July 18 Catalog
powerLawConst_b[' 0.37<z<0.66'] = -2.242    # July 18 Catalog

# 0.66<z<1.0
powerLawConst_a['0.66<z<1.0'] = 0.265     # July 18 Catalog
powerLawConst_b['0.66<z<1.0'] = -2.660    # July 18 Catalog

# 1.0<z<1.5
powerLawConst_a['1.0<z<1.5'] = 0.283     # July 18 Catalog
powerLawConst_b['1.0<z<1.5'] = -3.242    # July 18 Catalog

# 1.5<z<2.0
powerLawConst_a['1.5<z<2.0'] = 0.308     # July 18 Catalog
powerLawConst_b['1.5<z<2.0'] = -4.169    # July 18 Catalog

# 2.0<z<2.5
powerLawConst_a['2.0<z<2.5'] = 0.364    # July 18 Catalog
powerLawConst_b['2.0<z<2.5'] = -6.261    # July 18 Catalog

# 2.5<z<3.0
powerLawConst_a['2.5<z<3.0'] = 0.346    # July 18 Catalog
powerLawConst_b['2.5<z<3.0'] = -5.661    # July 18 Catalog

# 3.0<z<3.5
powerLawConst_a['3.0<z<3.5'] = 0.381    # July 18 Catalog
powerLawConst_b['3.0<z<3.5'] = -6.810    # July 18 Catalog

# 3.5<z<4.0
powerLawConst_a['3.5<z<4.0'] = 0.424    # July 18 Catalog
powerLawConst_b['3.5<z<4.0'] = -8.362    # July 18 Catalog

# 4.0<z<15.0
powerLawConst_a['4.0<z<15.0'] = 0.462    # July 18 Catalog
powerLawConst_b['4.0<z<15.0'] = -9.474    # July 18 Catalog

#################################################################################################
#################################################################################################
# Normalization to match the total galaxies from  mock catalogs to match up to CFHTLS counts for
# i<25.5 galaxy catalog. Using July 18-19 i-band mock catalogs.
normalizationConstant = 3.70330701802

#################################################################################################
#################################################################################################
# dictionary to hold colors for overplotted plots.
# coaddAnalysis and OSBiasAnalysis need an ordered dictionary; aritficialStructure just needs
# colors, in no specific order.
plotColor = OrderedDict()
plotColor['NoDither'] = [0., 0., 0.]  # black

plotColor['RandomDitherFieldPerVisit'] = [0.,0.,255/255.] # blue [255/255., 255/255.,   0.]  # yellow
plotColor['RandomDitherFieldPerNight'] = [255/255.,0.,255/255.]  # magenta
plotColor['RandomDitherPerNight'] = [255./255., 0.,0.] #[139/255., 0.,0.] # dark red

plotColor['RepulsiveRandomDitherFieldPerVisit'] = [255/255., 105/255., 180/255.] # hot pink
plotColor['RepulsiveRandomDitherFieldPerNight'] = [255/255., 255/255.,   0.]  # yellow
plotColor['RepulsiveRandomDitherPerNight'] = [147/255., 112/255., 219/255.] # medium purple

#plotColor['PentagonDiamondDitherPerSeason'] = [220/255.,  20/255.,  60/255.] # crimson
plotColor['PentagonDitherPerSeason'] = [75/255., 0., 130/255.] # indigo [255/255.,0.,255/255.]  # magenta [0., 1/255.,  34/255.]
#plotColor['SpiralDitherPerSeason'] = [4/255., 13/255.,  34/255.]

plotColor['FermatSpiralDitherFieldPerVisit'] = [124/255., 252/255.,   0.] # lawngreen
plotColor['FermatSpiralDitherFieldPerNight'] = [0, 206/255., 209/255.] # turqoise
plotColor['FermatSpiralDitherPerNight'] = [34/255., 139/255.,  34/255.] # forestgreen

plotColor['SequentialHexDitherFieldPerVisit'] = [ 0/255., 255/255., 127/255.] # spring green      [0.,0.,255/255.] # blue
plotColor['SequentialHexDitherFieldPerNight'] = [139/255., 0.,0.] # dark red
plotColor['SequentialHexDitherPerNight'] = [184/255., 134/255.,  11/255.] # dark goldenrod
