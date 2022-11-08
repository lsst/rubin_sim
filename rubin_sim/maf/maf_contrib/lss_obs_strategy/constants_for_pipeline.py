#################################################################################################
# Various things declared here to be imported when running artificialStructureCalculation.
# Makes the updates easier, since the constants/objects defined here are used by different
# functions, e.g. power law constants are called by GalaxyCountsMetric_extended as well as
# GalaxyCounts_withPixelCalibration. Similarly, plot_color dictionary helps maintain a consistent
# color scheme for different observing strategies.
#
# Included here:
#  * power law constants from the i-band mock catalog; based on mocks from Padilla et al.
#  * normalization constant for galaxy counts from the mock catalogs.
#  * plot_color dictionary: colors for plotting results from different observing strategies.s
#
# Humna Awan
# humna.awan@rutgers.edu
#################################################################################################
#################################################################################################
# Power law constants for each z-bin based on N. D. Padilla et al.'s mock catalogs
# General power law form: 10**(a*m+b)
# Declare the dictionary for the power law constants
from collections import OrderedDict

power_law_const_a = OrderedDict()
power_law_const_b = OrderedDict()
# 0.<z<0.15
power_law_const_a["0.<z<0.15"] = 0.166  # July 18 Catalog
power_law_const_b["0.<z<0.15"] = -0.989  # July 18 Catalog

# 0.15<z<0.37
power_law_const_a["0.15<z<0.37"] = 0.212  # July 18 Catalog
power_law_const_b["0.15<z<0.37"] = -1.540  # July 18 Catalog

# 0.37<z<0.66
power_law_const_a[" 0.37<z<0.66"] = 0.250  # July 18 Catalog
power_law_const_b[" 0.37<z<0.66"] = -2.242  # July 18 Catalog

# 0.66<z<1.0
power_law_const_a["0.66<z<1.0"] = 0.265  # July 18 Catalog
power_law_const_b["0.66<z<1.0"] = -2.660  # July 18 Catalog

# 1.0<z<1.5
power_law_const_a["1.0<z<1.5"] = 0.283  # July 18 Catalog
power_law_const_b["1.0<z<1.5"] = -3.242  # July 18 Catalog

# 1.5<z<2.0
power_law_const_a["1.5<z<2.0"] = 0.308  # July 18 Catalog
power_law_const_b["1.5<z<2.0"] = -4.169  # July 18 Catalog

# 2.0<z<2.5
power_law_const_a["2.0<z<2.5"] = 0.364  # July 18 Catalog
power_law_const_b["2.0<z<2.5"] = -6.261  # July 18 Catalog

# 2.5<z<3.0
power_law_const_a["2.5<z<3.0"] = 0.346  # July 18 Catalog
power_law_const_b["2.5<z<3.0"] = -5.661  # July 18 Catalog

# 3.0<z<3.5
power_law_const_a["3.0<z<3.5"] = 0.381  # July 18 Catalog
power_law_const_b["3.0<z<3.5"] = -6.810  # July 18 Catalog

# 3.5<z<4.0
power_law_const_a["3.5<z<4.0"] = 0.424  # July 18 Catalog
power_law_const_b["3.5<z<4.0"] = -8.362  # July 18 Catalog

# 4.0<z<15.0
power_law_const_a["4.0<z<15.0"] = 0.462  # July 18 Catalog
power_law_const_b["4.0<z<15.0"] = -9.474  # July 18 Catalog

#################################################################################################
#################################################################################################
# Normalization to match the total galaxies from  mock catalogs to match up to CFHTLS counts for
# i<25.5 galaxy catalog. Using July 18-19 i-band mock catalogs.
normalization_constant = 3.70330701802

#################################################################################################
#################################################################################################
# dictionary to hold colors for overplotted plots.
# coaddAnalysis and OSBiasAnalysis need an ordered dictionary; aritficialStructure just needs
# colors, in no specific order.
plot_color = OrderedDict()
plot_color["NoDither"] = [0.0, 0.0, 0.0]  # black

plot_color["RandomDitherFieldPerVisit"] = [
    0.0,
    0.0,
    255 / 255.0,
]  # blue [255/255., 255/255.,   0.]  # yellow
plot_color["RandomDitherFieldPerNight"] = [255 / 255.0, 0.0, 255 / 255.0]  # magenta
plot_color["RandomDitherPerNight"] = [
    255.0 / 255.0,
    0.0,
    0.0,
]  # [139/255., 0.,0.] # dark red

plot_color["RepulsiveRandomDitherFieldPerVisit"] = [
    255 / 255.0,
    105 / 255.0,
    180 / 255.0,
]  # hot pink
plot_color["RepulsiveRandomDitherFieldPerNight"] = [
    255 / 255.0,
    255 / 255.0,
    0.0,
]  # yellow
plot_color["RepulsiveRandomDitherPerNight"] = [
    147 / 255.0,
    112 / 255.0,
    219 / 255.0,
]  # medium purple

# plot_color['PentagonDiamondDitherPerSeason'] = [220/255.,  20/255.,  60/255.] # crimson
plot_color["PentagonDitherPerSeason"] = [
    75 / 255.0,
    0.0,
    130 / 255.0,
]  # indigo [255/255.,0.,255/255.]  # magenta [0., 1/255.,  34/255.]
# plot_color['SpiralDitherPerSeason'] = [4/255., 13/255.,  34/255.]

plot_color["FermatSpiralDitherFieldPerVisit"] = [
    124 / 255.0,
    252 / 255.0,
    0.0,
]  # lawngreen
plot_color["FermatSpiralDitherFieldPerNight"] = [
    0,
    206 / 255.0,
    209 / 255.0,
]  # turqoise
plot_color["FermatSpiralDitherPerNight"] = [
    34 / 255.0,
    139 / 255.0,
    34 / 255.0,
]  # forestgreen

plot_color["SequentialHexDitherFieldPerVisit"] = [
    0 / 255.0,
    255 / 255.0,
    127 / 255.0,
]  # spring green      [0.,0.,255/255.] # blue
plot_color["SequentialHexDitherFieldPerNight"] = [139 / 255.0, 0.0, 0.0]  # dark red
plot_color["SequentialHexDitherPerNight"] = [
    184 / 255.0,
    134 / 255.0,
    11 / 255.0,
]  # dark goldenrod
