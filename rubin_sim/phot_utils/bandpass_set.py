#
# LSST Data Management System
# Copyright 2008, 2009, 2010, 2011, 2012 LSST Corporation.
#
# This product includes software developed by the
# LSST Project (http://www.lsst.org/).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <http://www.lsstcorp.org/LegalNotices/>.
#

"""
  Questions or comments, email : ljones.uw@gmail.com


 The point of this class is mostly for convenience when dealing with
 sets of Seds and Bandpasses. Often this convenience is needed when
 dealing with these sets from the python interpreter (when figuring out
 if a group of SEDS looks appropriate, etc.)
 
 So, a lot of these functions actually deal with plotting. 
 Still, particularly in SedSet.py you may find the methods to calculate
 magnitudes or colors of a large group of seds 
 (with a set of Bandpasses, defined in this class) useful.
 
 Many of the functions defined here are useful for testing the set
 of LSST filters (i.e. do they meet the filter leak requirements?)
 or plotting the filters (i.e. plotFilters). 
"""

import os
import copy
import numpy as np
import matplotlib.pyplot as plt
from .bandpass import Bandpass
from .sed import Sed

# airmass of standard atmosphere
_std_x = 1.2

# wavelength range parameters for calculations.
WAVELEN_MIN = 300  # minimum wavelength for transmission/source (nm)
WAVELEN_MAX = 1200  # maximum wavelength for transmission/source (nm)
WAVELEN_STEP = 0.1  # step size in wavelength grid (nm)

# figure format to save output figures, if desired. (can choose 'png' or 'eps' or 'pdf' or a few others).
fig_format = "png"


class BandpassSet(object):
    """Set up a dictionary of a set of bandpasses (multi-filters).
    Run various engineering tests or visualizations."""

    def __init__(self):
        """Initialize the class but don't do anything yet."""
        return

    def set_bandpass_set(
        self, bp_dict, bp_dictlist=("u", "g", "r", "i", "z", "y"), verbose=True
    ):
        """Simply set throughputs from a pre-made dictionary."""
        if len(bp_dictlist) != len(list(bp_dict.keys())):
            bp_dict_list = list(bp_dict.keys())
        self.bandpass = copy.deepcopy(bp_dict)
        self.filterlist = copy.deepcopy(bp_dictlist)
        return

    def set_throughputs__single_files(
        self,
        filterlist=("u", "g", "r", "i", "z", "y"),
        rootdir="./",
        rootname="total_",
        rootsuffix=".dat",
        verbose=True,
    ):
        """Read bandpass set with filters in filterlist, from directory rootdir with base name rootname."""
        # Set up dictionary to hold bandpass information.
        bandpass = {}
        # Loop through filters:
        for f in filterlist:
            # Build full filename.
            filename = os.path.join(rootdir, rootname + f + rootsuffix)
            # read filter throughput and set up Sb/Phi and zeropoint
            if verbose:
                print("Reading throughput file %s" % (filename))
            # Initialize bandpass object.
            bandpass[f] = Bandpass()
            # Read the throughput curve, sampling onto grid of wavelen min/max/step.
            bandpass[f].read_throughput(
                filename,
                wavelen_min=WAVELEN_MIN,
                wavelen_max=WAVELEN_MAX,
                wavelen_step=WAVELEN_STEP,
            )
            # Calculate phi as well.
            bandpass[f].sb_tophi()
        # Set data in self.
        self.bandpass = bandpass
        self.filterlist = filterlist
        return

    def set_throughputs__component_files(
        self,
        filterlist=("u", "g", "r", "i", "z", "y"),
        all_filter_complist=(
            "detector.dat",
            "lens1.dat",
            "lens2.dat",
            "lens3.dat",
            "m1.dat",
            "m2.dat",
            "m3.dat",
            "atmos_std.dat",
        ),
        rootdir="./",
        verbose=True,
    ):
        """Read and build bandpass set from all_filter_complist, using data from directory rootdir.
        Note that with this method, every bandpass will be the same. The point is that then you can
        use this method, plus set up a different BandpassSet with values that are different for each filter
        and then multiply the two together using multiplyBandpassSets."""
        # Set up dictionary to hold final bandpass information.
        bandpass = {}
        # Loop through filters.
        # Set up full filenames in a list containing all elements of final throughput curve.
        complist = []
        # Join all 'all-filter' items.
        for cp in all_filter_complist:
            complist.append(os.path.join(rootdir, cp))
        for f in filterlist:
            if verbose:
                print("Reading throughput curves ", complist, " for filter ", f)
            # Initialize bandpass object.
            bandpass[f] = Bandpass()
            bandpass[f].read_throughputList(
                complist,
                wavelen_min=WAVELEN_MIN,
                wavelen_max=WAVELEN_MAX,
                wavelen_step=WAVELEN_STEP,
            )
            bandpass[f].sb_tophi()
        self.bandpass = bandpass
        self.filterlist = filterlist
        return

    def multiply_bandpass_sets(self, other_bp_set):
        """Multiply two bandpass sets together, filter by filter. Filterlists must match!
        Returns a new bandpassSet object."""
        if self.filterlist != other_bp_set.filterlist:
            raise Exception("The bandpassSet filter lists must match.")
        # Set up dictionary to hold new bandpass objects.
        new_bp_dict = {}
        for f in self.filterlist:
            wavelen, sb = self.bandpass[f].multiply_throughputs(
                other_bp_set.bandpass[f].wavelen, other_bp_set.bandpass[f].sb
            )
            new_bp_dict[f] = Bandpass(wavelen=wavelen, sb=sb)
        new_bp_set = BandpassSet()
        new_bp_set.set_bandpass_set(new_bp_dict, self.filterlist)
        return new_bp_set

    def write_phis(self, filename):
        """Write all phi values and wavelength to stdout"""
        # This is useful for getting a data file with only phi's, as requested by some science collaborations.
        file = open(filename, "w")
        # Print header.
        headerline = "#Wavelen(nm) "
        for filter in self.filterlist:
            headerline = headerline + "  phi_" + filter
        print(headerline, file=file)
        # print data
        for i in range(0, len(self.bandpass[self.filterlist[0]].wavelen), 1):
            outline = "%.2f " % (self.bandpass[self.filterlist[0]].wavelen[i])
            for f in self.filterlist:
                outline = outline + " %.6g " % (self.bandpass[f].phi[i])
            print(outline, file=file)
        file.close()
        return

    def write_photoz_throughputs(self, filename):
        """Write all throughputs in format AndyC needs for photoz"""
        file = open(filename, "w")
        for i, filter in enumerate(self.filterlist):
            file.write("%d NAME %d\n" % (len(filter.wavelen), i))
            j = 0
            for lam, thru in zip(filter.wavelen, filter.sb):
                file.write("%d %g %g\n" % (j, 10.0 * lam, thru))
                j = j + 1
        file.close()
        return

    def calc_filter_eff_wave(self, verbose=True):
        """Calculate the effective wavelengths for all filters."""
        # Set up dictionaries for effective wavelengths, as calculated for Transmission (sb) and Phi (phi).
        effsb = {}
        effphi = {}
        # Calculate values for each filter.
        for f in self.filterlist:
            effphi[f], effsb[f] = self.bandpass[f].calc_eff_wavelen()
        self.effsb = effsb
        self.effphi = effphi
        if verbose:
            print("Filter  Eff_Sb   Eff_phi")
            for f in self.filterlist:
                print(" %s      %.3f  %.3f" % (f, self.effsb[f], effphi[f]))
        return

    def calc_zero_points(self, gain=1.0, verbose=True):
        """Calculate the theoretical zeropoints for the bandpass, in AB magnitudes."""
        exptime = 15  # Default exposure time.
        effarea = (
            np.pi * (6.5 * 100 / 2.0) ** 2
        )  # Default effective area of primary mirror.
        zpt = {}
        print("Filter Zeropoint")
        for f in self.filterlist:
            zpt[f] = self.bandpass[f].calc_zp_t(
                expTime=exptime, effarea=effarea, gain=gain
            )
            print(" %s     %.3f" % (f, zpt[f]))
        return

    def calc_filter_edges(self, drop_peak=0.1, drop_percent=50, verbose=True):
        """Calculate the edges of each filter for Sb, at values of 'drop_*'.

        Values for drop_peak are X percent of max throughput, drop_percent is where the
        filter throughput drops to an absolute X percent value."""
        bandpass = self.bandpass
        filterlist = self.filterlist
        try:
            effsb = self.effsb
            effphi = self.effphi
        except AttributeError:
            self.calc_filter_eff_wave()
            effsb = self.effsb
            effphi = self.effphi
        # Set up dictionary for effective wavelengths and X% peak_drop wavelengths.
        drop_peak_blue = {}
        drop_peak_red = {}
        drop_perc_blue = {}
        drop_perc_red = {}
        maxthruput = {}
        # Calculate values for each filter.
        for f in filterlist:
            # Calculate minimum and maximum wavelengths for bandpass.
            minwavelen = bandpass[f].wavelen.min()
            maxwavelen = bandpass[f].wavelen.max()
            # Set defaults for dropoff points.
            drop_peak_blue[f] = maxwavelen
            drop_peak_red[f] = minwavelen
            drop_perc_blue[f] = maxwavelen
            drop_perc_red[f] = minwavelen
            # Find out what current wavelength grid is being used.
            wavelenstep = bandpass[f].wavelen[1] - bandpass[f].wavelen[0]
            # Find peak throughput.
            maxthruput[f] = bandpass[f].sb.max()
            # Calculate the values we're looking for (for the threshold for the 'drop')
            d_peak = maxthruput[f] * drop_peak / 100.0
            d_perc = drop_percent / 100.0  # given in %, must translate to fraction.
            # Find the nearest spot on the wavelength grid used for filter, for edge lookup.
            sbindex = np.where(abs(bandpass[f].wavelen - effsb[f]) < wavelenstep / 2.0)
            sbindex = sbindex[0][0]
            # Now find where Sb drops below 'drop_peak_thruput' of max for the first time.
            # Calculate wavelength where dropoff X percent of max level.
            # Start at effective wavelength, and walk outwards.
            for i in range(sbindex, len(bandpass[f].wavelen)):
                if bandpass[f].sb[i] <= d_peak:
                    drop_peak_red[f] = bandpass[f].wavelen[i]
                    break
            for i in range(sbindex, 0, -1):
                if bandpass[f].sb[i] <= d_peak:
                    drop_peak_blue[f] = bandpass[f].wavelen[i]
                    break
            # Calculate wavelength where dropoff X percent,  absolute value
            for i in range(sbindex, len(bandpass[f].wavelen)):
                if bandpass[f].sb[i] <= d_perc:
                    drop_perc_red[f] = bandpass[f].wavelen[i]
                    break
            for i in range(sbindex, 0, -1):
                if bandpass[f].sb[i] <= d_perc:
                    drop_perc_blue[f] = bandpass[f].wavelen[i]
                    break
        # Print output to screen.
        if verbose:
            print(
                "Filter  MaxThruput EffWavelen  %.3f%s_max(blue)  %.3f%s_max(red)  %.3f%s_abs(blue)  %.3f%s_abs(red)"
                % (drop_peak, "%", drop_peak, "%", drop_percent, "%", drop_percent, "%")
            )
            for f in self.filterlist:
                print(
                    "%4s   %10.4f %10.4f  %12.2f  %12.2f  %12.2f  %12.2f"
                    % (
                        f,
                        maxthruput[f],
                        effsb[f],
                        drop_peak_blue[f],
                        drop_peak_red[f],
                        drop_perc_blue[f],
                        drop_perc_red[f],
                    )
                )
        # Set values (dictionaries keyed by filterlist).
        self.drop_peak_red = drop_peak_red
        self.drop_peak_blue = drop_peak_blue
        self.drop_perc_red = drop_perc_red
        self.drop_perc_blue = drop_perc_blue
        return

    def calc_filter_leaks(
        self,
        ten_nm_limit=0.01,
        out_of_band_limit=0.05,
        filter_edges=0.1,
        extra_title=None,
        makeplot=True,
        savefig=False,
        figroot="bandpass",
    ):
        """Calculate throughput leaks beyond location where bandpass drops to filter_edges (%) of max throughput.


        According to SRD these leaks must be below 0.01% of peak value in any 10nm interval,
        and less than 0.05% of total transmission over all wavelengths beyond where thruput<0.1% of peak.
        Assumes wavelength is in nanometers! (because of nm requirement). Uses ten_nm_limit and out_of_band_limit
        to set specs. Note that the values given here should be in PERCENT (not fractions).
        Generates plots for each filter, as well as calculation of fleaks."""
        # Go through each filter, calculate filter leaks.
        filterlist = self.filterlist
        bandpass = self.bandpass
        # Make sure effective wavelengths defined.
        self.calc_filter_eff_wave(verbose=False)
        effsb = self.effsb
        # Look for the new FWHM definition for the 10nm filter leak definition
        if filter_edges == "FWHM":
            self.calc_filter_eff_wave(verbose=False)
            self.calc_filter_edges(drop_percent=0.50, verbose=False)
            # Calculate FWHM values.
            fwhm = {}
            for f in filterlist:
                fwhm[f] = self.drop_peak_red[f] - self.drop_peak_blue[f]
            # Adjust 'drop' edges to account for being FWHM from center.
            for f in filterlist:
                self.drop_peak_red[f] = self.effsb[f] + fwhm[f]
                self.drop_peak_blue[f] = self.effsb[f] - fwhm[f]
        # Otherwise, traditional % definition.
        else:
            self.calc_filter_edges(drop_peak=filter_edges, verbose=False)
        drop_peak_red = self.drop_peak_red
        drop_peak_blue = self.drop_peak_blue
        # Set up plot colors.
        colors = ("m", "b", "g", "y", "r", "k", "c")
        colorindex = 0
        for f in filterlist:
            print("=====")
            print("Analyzing %s filter" % (f))
            # find wavelength range in use.
            minwavelen = bandpass[f].wavelen.min()
            maxwavelen = bandpass[f].wavelen.max()
            # find out what current wavelength grid is being used
            wavelenstep = bandpass[f].wavelen[1] - bandpass[f].wavelen[0]
            # find the wavelength in the wavelength grid which is closest to effsb
            condition = abs(bandpass[f].wavelen - effsb[f]) < wavelenstep / 2.0
            waveleneffsb = bandpass[f].wavelen[condition]
            # calculate peak transmission
            peaktrans = bandpass[f].sb.max()
            # calculate total transmission withinin proper bandpass
            condition = (bandpass[f].wavelen > drop_peak_blue[f]) & (
                bandpass[f].wavelen < drop_peak_red[f]
            )
            temporary = bandpass[f].sb[condition]
            totaltrans = temporary.sum()
            # calculate total transmission outside drop_peak wavelengths of peak
            condition = (bandpass[f].wavelen >= drop_peak_red[f]) | (
                bandpass[f].wavelen <= drop_peak_blue[f]
            )
            temporary = bandpass[f].sb[condition]
            sumthruput_outside_bandpass = temporary.sum()
            print("Total transmission through filter: %s" % (totaltrans))
            print(
                "Transmission outside of filter edges (drop_peak): %f"
                % (sumthruput_outside_bandpass)
            )
            # Calculate percentage of out of band transmission to in-band transmission
            out_of_band_perc = sumthruput_outside_bandpass / totaltrans * 100.0
            print(
                "Ratio of total out-of-band to in-band transmission: %f%s"
                % (out_of_band_perc, "%")
            )
            infotext = "Out-of-band/in-band transmission %.3f%s" % (
                out_of_band_perc,
                "%",
            )
            if out_of_band_perc > out_of_band_limit:
                print(
                    " Does not meet SRD-This is more than %.4f%s of throughput outside the bandpass %s"
                    % (out_of_band_limit, "%", f)
                )
            else:
                print(
                    " Meets SRD - This is less than %.4f%s of total throughput outside bandpass"
                    % (out_of_band_limit, "%")
                )
            # calculate transmission in each 10nm interval.
            sb_10nm = np.zeros(len(bandpass[f].sb), dtype="float")
            gapsize_10nm = 10.0  # wavelen gap in nm
            meet_srd = True
            maxsb_10nm = 0.0
            maxwavelen_10nm = 0.0
            # Convert 10nm limit into actual value (and account for %)
            ten_nm_limit_value = ten_nm_limit * peaktrans / 100.0
            for i in range(0, len(sb_10nm), 1):
                # calculate 10 nm 'smoothed' transmission
                wavelen = bandpass[f].wavelen[i]
                condition = (
                    (bandpass[f].wavelen >= wavelen - gapsize_10nm / 2.0)
                    & (bandpass[f].wavelen < wavelen + gapsize_10nm / 2.0)
                    & (
                        (bandpass[f].wavelen <= drop_peak_blue[f])
                        | (bandpass[f].wavelen >= drop_peak_red[f])
                    )
                )
                sb_10nm[i] = bandpass[f].sb[condition].mean()
            condition = (bandpass[f].wavelen > drop_peak_blue[f]) & (
                bandpass[f].wavelen < drop_peak_red[f]
            )
            sb_10nm[condition] = 0
            # now check for violation of SRD
            if sb_10nm.max() > ten_nm_limit_value:
                meet_srd = False
                maxsb_10nm = sb_10nm.max()
                maxwavelen_10nm = bandpass[f].wavelen[
                    np.where(sb_10nm == sb_10nm.max())
                ]
            if meet_srd == False:
                print(
                    "Does not meet SRD - %s has at least one region not meeting the 10nm SRD filter leak requirement (max is %f%s of peak transmission at %.1f A)"
                    % (f, maxsb_10nm, "%", maxwavelen_10nm)
                )
            else:
                print("10nm limit within SRD.")
            if makeplot:
                # make plot for this filter
                plt.figure()
                # set colors for filter in plot
                color = colors[colorindex]
                colorindex = colorindex + 1
                if colorindex == len(colors):
                    colorindex = 0
                # Make lines on the plot.
                plt.plot(
                    bandpass[f].wavelen, bandpass[f].sb, color=color, linestyle="-"
                )
                plt.plot(bandpass[f].wavelen, sb_10nm, "r-", linewidth=2)
                plt.axvline(drop_peak_blue[f], color="b", linestyle=":")
                plt.axvline(drop_peak_red[f], color="b", linestyle=":")
                plt.axhline(ten_nm_limit_value, color="b", linestyle=":")
                legendstring = f + " filter thruput, 10nm average thruput in red\n"
                legendstring = legendstring + "  Peak throughput is %.1f%s\n" % (
                    peaktrans * 100.0,
                    "%",
                )
                legendstring = (
                    legendstring
                    + "  Total throughput (in band) is %.0f%s\n"
                    % (totaltrans * 100.0, "%")
                )
                legendstring = legendstring + "  " + infotext
                plt.figtext(0.25, 0.76, legendstring)
                plt.xlabel("Wavelength (nm)")
                plt.ylabel("Throughput (0-1)")
                plt.yscale("log")
                if extra_title != None:
                    titletext = extra_title + " " + f
                else:
                    titletext = f
                plt.title(titletext)
                plt.ylim(1e-6, 1)
                plt.xlim(xmin=300, xmax=1200)
                if savefig:
                    figname = figroot + "_" + f + "_fleak." + fig_format
                    plt.savefig(figname, format=fig_format)
        # end of loop through filters
        return

    def plot_filters(
        self,
        rootdir=".",
        throughput=True,
        phi=False,
        atmos=True,
        plotdropoffs=False,
        ploteffsb=True,
        compare=None,
        savefig=False,
        figroot="bandpass",
        xlim=(300, 1100),
        ylimthruput=(0, 1),
        ylimphi=(0, 0.002),
        filter_tags="normal",
        leg_tag=None,
        compare_tag=None,
        title=None,
        linestyle="-",
        linewidth=2,
        newfig=True,
    ):
        """Plot the filter throughputs and phi's, with limits xlim/ylimthruput/ylimphi.

        Optionally add comparison (another BandpassSet) throughput and phi curves.
        and show lines for % dropoffs ; filter_tags can be side or normal."""
        # check that all self variables are set up if needed
        bandpass = self.bandpass
        filterlist = self.filterlist
        try:
            self.effsb
            self.effphi
            if plotdropoffs:
                self.drop_peak_red
                self.drop_peak_blue
        except AttributeError:
            self.calc_filter_eff_wave(verbose=False)
            if plotdropoffs:
                self.calc_filter_edges(verbose=False)
        effsb = self.effsb
        effphi = self.effphi
        if plotdropoffs:
            drop_peak_red = self.drop_peak_red
            drop_peak_blue = self.drop_peak_blue
        # read files for atmosphere and optional comparison throughputs
        if atmos:
            atmosfile = os.path.join(rootdir, "atmos_std.dat")
            atmosphere = Bandpass()
            atmosphere.read_throughput(atmosfile)
        xatm = _std_x
        # set up colors for plot output
        colors = ("k", "b", "g", "y", "r", "m", "burlywood", "k")
        # colors = ('r', 'b', 'r', 'b', 'r', 'b', 'r', 'b')
        if throughput:
            if newfig:
                plt.figure()
            # plot throughputs
            colorindex = 0
            for f in filterlist:
                color = colors[colorindex]
                colorindex = colorindex + 1
                if colorindex == len(colors):
                    colorindex = 0
                plt.plot(
                    bandpass[f].wavelen,
                    bandpass[f].sb,
                    color=color,
                    linestyle=linestyle,
                    linewidth=linewidth,
                )
            # add effective wavelengths (optional)
            if ploteffsb:
                vertline = np.arange(0, 1.2, 0.1)
                temp = vertline * 0.0 + 1.0
                colorindex = 0
                for f in filterlist:
                    color = colors[colorindex]
                    colorindex = colorindex + 1
                    if colorindex == len(colors):
                        colorindex = 0
                    plt.plot(effsb[f] * temp, vertline, color=color, linestyle="-")
            # add dropoff limits if desired (usually only good with reduced x/y limits) (optional)
            if plotdropoffs:
                colorindex = 0
                for filter in filterlist:
                    color = colors[colorindex]
                    colorindex = colorindex + 1
                    if colorindex == len(colors):
                        colorindex = 0
                    plt.plot(
                        drop_peak_red[f] * temp, vertline, color=color, linestyle="--"
                    )
                    plt.plot(
                        drop_peak_blue[f] * temp, vertline, color=color, linestyle="--"
                    )
            # plot atmosphere (optional)
            if atmos:
                plt.plot(atmosphere.wavelen, atmosphere.sb, "k:")
            # plot comparison throughputs (optional)
            if compare != None:
                colorindex = 0
                for f in compare.filterlist:
                    color = colors[colorindex]
                    colorindex = colorindex + 1
                    if colorindex == len(colors):
                        colorindex = 0
                    plt.plot(
                        compare.bandpass[f].wavelen,
                        compare.bandpass[f].sb,
                        color=color,
                        linestyle="--",
                    )
            # add line legend (type of filter curves)
            legendtext = "%s = solid" % (leg_tag)
            if leg_tag == None:
                legendtext = ""
            if compare != None:
                if compare_tag != None:
                    legendtext = legendtext + "\n%s = dashed" % (compare_tag)
            if atmos:
                legendtext = legendtext + "\nAirmass %.1f" % (xatm)
            plt.figtext(0.15, 0.8, legendtext)
            # add names to filter throughputs
            if filter_tags == "side":
                xtags = np.zeros(len(filterlist), dtype=float)
                xtags = xtags + 0.15
                spacing = (0.8 - 0.1) / len(filterlist)
                ytags = np.arange(0.8, 0.1, -1 * spacing, dtype=float)
                ytags = ytags
            else:  # 'normal' tagging
                xtags = (0.16, 0.27, 0.42, 0.585, 0.68, 0.8, 0.8, 0.8)
                ytags = (0.73, 0.73, 0.73, 0.73, 0.73, 0.73, 0.69, 0.65)
            index = 0
            colorindex = 0
            for f in filterlist:
                plt.figtext(
                    xtags[index],
                    ytags[index],
                    f,
                    color=colors[colorindex],
                    va="top",
                    size="x-large",
                )
                index = index + 1
                colorindex = colorindex + 1
                if colorindex == len(colors):
                    colorindex = 0
            # set x/y limits
            plt.xlim(xmin=xlim[0], xmax=xlim[1])
            plt.ylim(ymin=ylimthruput[0], ymax=ylimthruput[1])
            plt.xlabel("Wavelength (nm)")
            plt.ylabel("Throughput (0-1)")
            plt.grid()
            if title != None:
                plt.title(title)
            if savefig:
                figname = figroot + "_thruputs." + fig_format
                plt.savefig(figname, format=fig_format)
        if phi:
            if newfig:
                plt.figure()
            # plot LSST 'phi' curves
            colorindex = 0
            for f in filterlist:
                color = colors[colorindex]
                colorindex = colorindex + 1
                if colorindex == len(colors):
                    colorindex = 0
                plt.plot(
                    bandpass[f].wavelen,
                    bandpass[f].phi,
                    color=color,
                    linestyle=linestyle,
                    linewidth=linewidth,
                )
            # add effective wavelengths for main filter set (optional)
            if ploteffsb:
                vertline = np.arange(0, 0.1, 0.01)
                temp = vertline * 0.0 + 1.0
                colorindex = 0
                for filter in filterlist:
                    color = colors[colorindex]
                    colorindex = colorindex + 1
                    if colorindex == len(colors):
                        colorindex = 0
                    plt.plot(effphi[f] * temp, vertline, color=color, linestyle="-")
            # plot comparison throughputs (optional)
            if compare != None:
                colorindex = 0
                for filter in compare.filterlist:
                    color = colors[colorindex]
                    colorindex = colorindex + 1
                    if colorindex == len(colors):
                        colorindex = 0
                    plt.plot(
                        compare.bandpass[f].wavelen,
                        compare.bandpass[f].phi,
                        color=color,
                        linestyle="--",
                    )
            # add line legend
            legendtext = "%s = solid" % (leg_tag)
            if leg_tag == None:
                legendtext = " "
            if compare != None:
                if compare_tag != None:
                    legendtext = legendtext + "\n%s = dashed" % (compare_tag)
            plt.figtext(0.15, 0.78, legendtext)
            # add name tags to filters
            if filter_tags == "side":
                xtags = np.zeros(len(filterlist), dtype=float)
                xtags = xtags + 0.15
                ytags = np.arange(len(filterlist), 0, -1.0, dtype=float)
                ytags = ytags * 0.04 + 0.35
            else:
                xtags = (0.17, 0.27, 0.42, 0.585, 0.677, 0.82, 0.82, 0.82)
                ytags = (0.63, 0.63, 0.63, 0.63, 0.63, 0.63, 0.60, 0.57)
            index = 0
            colorindex = 0
            for f in filterlist:
                plt.figtext(
                    xtags[index], ytags[index], f, color=colors[colorindex], va="top"
                )
                index = index + 1
                colorindex = colorindex + 1
                if colorindex == len(colors):
                    colorindex = 0
            # set x/y limits
            plt.xlim(xmin=xlim[0], xmax=xlim[1])
            plt.ylim(ymin=ylimphi[0], ymax=ylimphi[1])
            plt.xlabel("Wavelength (nm)")
            plt.ylabel("Phi")
            plt.grid()
            if title != None:
                plt.title(title)
            if savefig:
                figname = figroot + "_phi." + fig_format
                plt.savefig(figname, format=fig_format)
        return
