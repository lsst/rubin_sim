import numpy as np
import os
import glob
import healpy as hp
from rubin_sim.photUtils import Sed, Bandpass
from .twilightFunc import twilightFunc
from scipy.interpolate import InterpolatedUnivariateSpline, interp1d
from rubin_sim.data import get_data_dir

# Make backwards compatible with healpy
if hasattr(hp, 'get_interp_weights'):
    get_neighbours = hp.get_interp_weights
elif hasattr(hp, 'get_neighbours'):
    get_neighbours = hp.get_neighbours
else:
    print("Could not find appropriate healpy function for get_interp_weight or get_neighbours")


__all__ = ['id2intid', 'intid2id', 'loadSpecFiles', 'BaseSingleInterp', 'ScatteredStar', 'LowerAtm',
           'UpperAtm', 'MergedSpec', 'Airglow', 'TwilightInterp', 'MoonInterp',
           'ZodiacalInterp']


def id2intid(ids):
    """
    take an array of ids, and convert them to an integer id.
    Handy if you want to put things into a sparse array.
    """
    uids = np.unique(ids)
    order = np.argsort(ids)
    oids = ids[order]
    uintids = np.arange(np.size(uids), dtype=int)
    left = np.searchsorted(oids, uids)
    right = np.searchsorted(oids, uids, side='right')
    intids = np.empty(ids.size, dtype=int)
    for i in range(np.size(left)):
        intids[left[i]:right[i]] = uintids[i]
    result = intids*0
    result[order] = intids
    return result, uids, uintids


def intid2id(intids, uintids, uids, dtype=int):
    """
    convert an int back to an id
    """
    ids = np.zeros(np.size(intids))

    order = np.argsort(intids)
    ointids = intids[order]
    left = np.searchsorted(ointids, uintids, side='left')
    right = np.searchsorted(ointids, uintids, side='right')
    for i, (le, ri) in enumerate(zip(left, right)):
        ids[le:ri] = uids[i]
    result = np.zeros(np.size(intids), dtype=dtype)
    result[order] = ids

    return result


def loadSpecFiles(filenames, mags=False):
    """
    Load up the ESO spectra.

    The ESO npz files contain the following arrays:
    filterWave: The central wavelengths of the pre-computed magnitudes
    wave: wavelengths for the spectra
    spec: array of spectra and magnitudes along with the relevant variable inputs.  For example,
    airglow has dtype = [('airmass', '<f8'), ('solarFlux', '<f8'), ('spectra', '<f8', (17001,)),
                         ('mags', '<f8', (6,)]
    For each unique airmass and solarFlux value, there is a 17001 elements spectra and 6 magnitudes.
    """

    if len(filenames) == 1:
        temp = np.load(filenames[0])
        wave = temp['wave'].copy()
        filterWave = temp['filterWave'].copy()
        if mags:
            # don't copy the spectra to save memory space
            dt = np.dtype([(key, temp['spec'].dtype[i]) for
                           i, key in enumerate(temp['spec'].dtype.names) if key != 'spectra'])
            spec = np.zeros(temp['spec'].size, dtype=dt)
            for key in temp['spec'].dtype.names:
                if key != 'spectra':
                    spec[key] = temp['spec'][key].copy()
        else:
            spec = temp['spec'].copy()
    else:
        temp = np.load(filenames[0])
        wave = temp['wave'].copy()
        filterWave = temp['filterWave'].copy()
        if mags:
            # don't copy the spectra to save memory space
            dt = np.dtype([(key, temp['spec'].dtype[i]) for
                           i, key in enumerate(temp['spec'].dtype.names) if key != 'spectra'])
            spec = np.zeros(temp['spec'].size, dtype=dt)
            for key in temp['spec'].dtype.names:
                if key != 'spectra':
                    spec[key] = temp['spec'][key].copy()
        else:
            spec = temp['spec'].copy()
        for filename in filenames[1:]:
            temp = np.load(filename)
            if mags:
                # don't copy the spectra to save memory space
                dt = np.dtype([(key, temp['spec'].dtype[i]) for
                               i, key in enumerate(temp['spec'].dtype.names) if key != 'spectra'])
                tempspec = np.zeros(temp['spec'].size, dtype=dt)
                for key in temp['spec'].dtype.names:
                    if key != 'spectra':
                        tempspec[key] = temp['spec'][key].copy()
            else:
                tempspec = temp['spec']
            spec = np.append(spec, tempspec)
    return spec, wave, filterWave


class BaseSingleInterp(object):
    """
    Base class for sky components that only need to be interpolated on airmass
    """

    def __init__(self, compName=None, sortedOrder=['airmass', 'nightTimes'], mags=False):
        """
        mags: Rather than the full spectrum, return the LSST ugrizy magnitudes.
        """

        self.mags = mags

        dataDir = os.path.join(get_data_dir(), 'skybrightness', 'ESO_Spectra/'+compName)

        filenames = sorted(glob.glob(dataDir+'/*.npz'))
        self.spec, self.wave, self.filterWave = loadSpecFiles(filenames, mags=self.mags)

        # Take the log of the spectra in case we want to interp in log space.
        if not mags:
            self.logSpec = np.zeros(self.spec['spectra'].shape, dtype=float)
            good = np.where(self.spec['spectra'] != 0)
            self.logSpec[good] = np.log10(self.spec['spectra'][good])
            self.specSize = self.spec['spectra'][0].size
        else:
            self.specSize = 0

        # What order are the dimesions sorted by (from how the .npz was packaged)
        self.sortedOrder = sortedOrder
        self.dimDict = {}
        self.dimSizes = {}
        for dt in self.sortedOrder:
            self.dimDict[dt] = np.unique(self.spec[dt])
            self.dimSizes[dt] = np.size(np.unique(self.spec[dt]))

        # Set up and save the dict to order the filters once.
        self.filterNameDict = {'u': 0, 'g': 1, 'r': 2, 'i': 3, 'z': 4, 'y': 5}

    def __call__(self, intepPoints, filterNames=['u', 'g', 'r', 'i', 'z', 'y']):
        if self.mags:
            return self.interpMag(intepPoints, filterNames=filterNames)
        else:
            return self.interpSpec(intepPoints)

    def indxAndWeights(self, points, grid):
        """
        for given 1-D points, find the grid points on either side and return the weights
        assume grid is sorted
        """

        order = np.argsort(points)

        indxL = np.empty(points.size, dtype=int)
        indxR = np.empty(points.size, dtype=int)

        indxR[order] = np.searchsorted(grid, points[order])
        indxL = indxR-1

        # If points off the grid were requested, just use the edge grid point
        offGrid = np.where(indxR == grid.size)
        indxR[offGrid] = grid.size-1
        fullRange = grid[indxR]-grid[indxL]

        wL = np.zeros(fullRange.size, dtype=float)
        wR = np.ones(fullRange.size, dtype=float)

        good = np.where(fullRange != 0)
        wL[good] = (grid[indxR][good] - points[good])/fullRange[good]
        wR[good] = (points[good] - grid[indxL[good]])/fullRange[good]

        return indxR, indxL, wR, wL

    def _weighting(self, interpPoints, values):
        """
        given a list/array of airmass values, return a dict with the interpolated
        spectrum at each airmass and the wavelength array.

        Input interpPoints should be sorted
        """
        results = np.zeros((interpPoints.size, np.size(values[0])), dtype=float)

        inRange = np.where((interpPoints['airmass'] <= np.max(self.dimDict['airmass'])) &
                           (interpPoints['airmass'] >= np.min(self.dimDict['airmass'])))
        indxR, indxL, wR, wL = self.indxAndWeights(interpPoints['airmass'][inRange],
                                                   self.dimDict['airmass'])

        nextra = 1

        # XXX--should I use the log spectra?  Make a check and switch back and forth?
        results[inRange] = wR[:, np.newaxis]*values[indxR*nextra] + \
            wL[:, np.newaxis]*values[indxL*nextra]

        return results

    def interpSpec(self, interpPoints):
        result = self._weighting(interpPoints, self.logSpec)
        mask = np.where(result == 0.)
        result = 10.**result
        result[mask] = 0.
        return {'spec': result, 'wave': self.wave}

    def interpMag(self, interpPoints, filterNames=['u', 'g', 'r', 'i', 'z', 'y']):
        filterindx = [self.filterNameDict[key] for key in filterNames]
        result = self._weighting(interpPoints, self.spec['mags'][:, filterindx])
        mask = np.where(result == 0.)
        result = 10.**(-0.4*(result-np.log10(3631.)))
        result[mask] = 0.
        return {'spec': result, 'wave': self.filterWave}


class ScatteredStar(BaseSingleInterp):
    """
    Interpolate the spectra caused by scattered starlight.
    """

    def __init__(self, compName='ScatteredStarLight', mags=False):
        super(ScatteredStar, self).__init__(compName=compName, mags=mags)


class LowerAtm(BaseSingleInterp):
    """
    Interpolate the spectra caused by the lower atmosphere.
    """

    def __init__(self, compName='LowerAtm', mags=False):
        super(LowerAtm, self).__init__(compName=compName, mags=mags)


class UpperAtm(BaseSingleInterp):
    """
    Interpolate the spectra caused by the upper atmosphere.
    """

    def __init__(self, compName='UpperAtm', mags=False):
        super(UpperAtm, self).__init__(compName=compName, mags=mags)


class MergedSpec(BaseSingleInterp):
    """
    Interpolate the spectra caused by the sum of the scattered starlight, airglow, upper and lower atmosphere.
    """

    def __init__(self, compName='MergedSpec', mags=False):
        super(MergedSpec, self).__init__(compName=compName, mags=mags)


class Airglow(BaseSingleInterp):
    """
    Interpolate the spectra caused by airglow.
    """

    def __init__(self, compName='Airglow', sortedOrder=['airmass', 'solarFlux'], mags=False):
        super(Airglow, self).__init__(compName=compName, mags=mags, sortedOrder=sortedOrder)
        self.nSolarFlux = np.size(self.dimDict['solarFlux'])

    def _weighting(self, interpPoints, values):

        results = np.zeros((interpPoints.size, np.size(values[0])), dtype=float)
        # Only interpolate point that lie in the model grid
        inRange = np.where((interpPoints['airmass'] <= np.max(self.dimDict['airmass'])) &
                           (interpPoints['airmass'] >= np.min(self.dimDict['airmass'])) &
                           (interpPoints['solarFlux'] >= np.min(self.dimDict['solarFlux'])) &
                           (interpPoints['solarFlux'] <= np.max(self.dimDict['solarFlux'])))
        usePoints = interpPoints[inRange]
        amRightIndex, amLeftIndex, amRightW, amLeftW = self.indxAndWeights(usePoints['airmass'],
                                                                           self.dimDict['airmass'])

        sfRightIndex, sfLeftIndex, sfRightW, sfLeftW = self.indxAndWeights(usePoints['solarFlux'],
                                                                           self.dimDict['solarFlux'])

        for amIndex, amW in zip([amRightIndex, amLeftIndex], [amRightW, amLeftW]):
            for sfIndex, sfW in zip([sfRightIndex, sfLeftIndex], [sfRightW, sfLeftW]):
                results[inRange] += amW[:, np.newaxis]*sfW[:, np.newaxis] * \
                    values[amIndex*self.nSolarFlux+sfIndex]
        return results


class TwilightInterp(object):

    def __init__(self, mags=False, darkSkyMags=None, fitResults=None):
        """
        Read the Solar spectrum into a handy object and compute mags in different filters
        mags:  If true, only return the LSST filter magnitudes, otherwise return the full spectrum

        darkSkyMags = dict of the zenith dark sky values to be assumed. The twilight fits are
        done relative to the dark sky level.
        fitResults = dict of twilight parameters based on twilightFunc. Keys should be filter names.
        """

        if darkSkyMags is None:
            darkSkyMags = {'u': 22.8, 'g': 22.3, 'r': 21.2,
                           'i': 20.3, 'z': 19.3, 'y': 18.0,
                           'B': 22.35, 'G': 21.71, 'R': 21.3}

        self.mags = mags

        dataDir = os.path.join(get_data_dir(), 'skybrightness')

        solarSaved = np.load(os.path.join(dataDir, 'solarSpec/solarSpec.npz'))
        self.solarSpec = Sed(wavelen=solarSaved['wave'], flambda=solarSaved['spec'])
        solarSaved.close()

        canonFilters = {}
        fnames = ['blue_canon.csv', 'green_canon.csv', 'red_canon.csv']

        # Filter names, from bluest to reddest.
        self.filterNames = ['B', 'G', 'R']

        for fname, filterName in zip(fnames, self.filterNames):
            bpdata = np.genfromtxt(os.path.join(dataDir, 'Canon/', fname), delimiter=', ',
                                   dtype=list(zip(['wave', 'through'], [float]*2)))
            bpTemp = Bandpass()
            bpTemp.setBandpass(bpdata['wave'], bpdata['through'])
            canonFilters[filterName] = bpTemp

        # Tack on the LSST filters
        throughPath = os.path.join(get_data_dir(), 'throughputs', 'baseline')
        lsstKeys = ['u', 'g', 'r', 'i', 'z', 'y']
        for key in lsstKeys:
            bp = np.loadtxt(os.path.join(throughPath, 'total_'+key+'.dat'),
                            dtype=list(zip(['wave', 'trans'], [float]*2)))
            tempB = Bandpass()
            tempB.setBandpass(bp['wave'], bp['trans'])
            canonFilters[key] = tempB
            self.filterNames.append(key)

        # MAGIC NUMBERS from fitting the all-sky camera:
        # Code to generate values in sims_skybrightness/examples/fitTwiSlopesSimul.py
        # Which in turn uses twilight maps from sims_skybrightness/examples/buildTwilMaps.py
        # values are of the form:
        # 0: ratio of f^z_12 to f_dark^z
        # 1: slope of curve wrt sun alt
        # 2: airmass term (10^(arg[2]*(X-1)))
        # 3: azimuth term.
        # 4: zenith dark sky flux (erg/s/cm^2)

        # For z and y, just assuming the shape parameter fits are similar to the other bands.
        # Looks like the diode is not sensitive enough to detect faint sky.
        # Using the Patat et al 2006 I-band values for z and modeified a little for y as a temp fix.
        if fitResults is None:
            self.fitResults = {'B': [7.56765633e+00, 2.29798055e+01, 2.86879956e-01,
                                     3.01162143e-01, 2.58462036e-04],
                               'G': [2.38561156e+00, 2.29310648e+01, 2.97733083e-01,
                                     3.16403197e-01, 7.29660095e-04],
                               'R': [1.75498017e+00, 2.22011802e+01, 2.98619033e-01,
                                     3.28880254e-01, 3.24411056e-04],
                               'z': [2.29, 24.08, 0.3, 0.3, -666],
                               'y': [2.0, 24.08, 0.3, 0.3, -666]}

            # XXX-completely arbitrary fudge factor to make things brighter in the blue
            # Just copy the blue and say it's brighter.
            self.fitResults['u'] = [16., 2.29622121e+01, 2.85862729e-01,
                                    2.99902574e-01, 2.32325117e-04]
        else:
            self.fitResults = fitResults

        # Take out any filters that don't have fit results
        self.filterNames = [key for key in self.filterNames if key in self.fitResults]

        self.effWave = []
        self.solarMag = []
        for filterName in self.filterNames:
            self.effWave.append(canonFilters[filterName].calcEffWavelen()[0])
            self.solarMag.append(self.solarSpec.calcMag(canonFilters[filterName]))

        order = np.argsort(self.effWave)
        self.filterNames = np.array(self.filterNames)[order]
        self.effWave = np.array(self.effWave)[order]
        self.solarMag = np.array(self.solarMag)[order]

        # update the fit results to be zeropointed properly
        for key in self.fitResults:
            f0 = 10.**(-0.4*(darkSkyMags[key]-np.log10(3631.)))
            self.fitResults[key][-1] = f0

        self.solarWave = self.solarSpec.wavelen
        self.solarFlux = self.solarSpec.flambda
        # This one isn't as bad as the model grids, maybe we could get away with computing the magnitudes
        # in the __call__ each time.
        if mags:
            # Load up the LSST filters and convert the solarSpec.flabda and solarSpec.wavelen to fluxes
            throughPath = throughPath = os.path.join(get_data_dir(), 'throughputs', 'baseline')
            self.lsstFilterNames = ['u', 'g', 'r', 'i', 'z', 'y']
            self.lsstEquations = np.zeros((np.size(self.lsstFilterNames),
                                           np.size(self.fitResults['B'])), dtype=float)
            self.lsstEffWave = []

            fits = np.empty((np.size(self.effWave), np.size(self.fitResults['B'])), dtype=float)
            for i, fn in enumerate(self.filterNames):
                fits[i, :] = self.fitResults[fn]

            for filtername in self.lsstFilterNames:
                bp = np.loadtxt(os.path.join(throughPath, 'total_'+filtername+'.dat'),
                                dtype=list(zip(['wave', 'trans'], [float]*2)))
                tempB = Bandpass()
                tempB.setBandpass(bp['wave'], bp['trans'])
                self.lsstEffWave.append(tempB.calcEffWavelen()[0])
            # Loop through the parameters and interpolate to new eff wavelengths
            for i in np.arange(self.lsstEquations[0, :].size):
                interp = InterpolatedUnivariateSpline(self.effWave, fits[:, i])
                self.lsstEquations[:, i] = interp(self.lsstEffWave)
            # Set the dark sky flux
            for i, filterName in enumerate(self.lsstFilterNames):
                self.lsstEquations[i, -1] = 10.**(-0.4*(darkSkyMags[filterName]-np.log10(3631.)))

        self.filterNameDict = {'u': 0, 'g': 1, 'r': 2, 'i': 3, 'z': 4, 'y': 5}

    def printFitsUsed(self):
        """
        Print out the fit parameters being used
        """
        print('\\tablehead{\colhead{Filter} & \colhead{$r_{12/z}$} & \colhead{$a$ (1/radians)} & \colhead{$b$ (1/airmass)} & \colhead{$c$ (az term/airmass)} & \colhead{$f_z_dark$ (erg/s/cm$^2$)$\\times 10^8$} & \colhead{m$_z_dark$}}')
        for key in self.fitResults:
            numbers = ''
            for num in self.fitResults[key]:
                if num > .001:
                    numbers += ' & %.2f' % num
                else:
                    numbers += ' & %.2f' % (num*1e8)
            print(key, numbers, ' & ', '%.2f' % (-2.5*np.log10(self.fitResults[key][-1])+np.log10(3631.)))

    def __call__(self, intepPoints, filterNames=['u', 'g', 'r', 'i', 'z', 'y']):
        if self.mags:
            return self.interpMag(intepPoints, filterNames=filterNames)
        else:
            return self.interpSpec(intepPoints)

    def interpMag(self, interpPoints, maxAM=3.0,
                  limits=[np.radians(-5.), np.radians(-20.)],
                  filterNames=['u', 'g', 'r', 'i', 'z', 'y']):
        """
        Originally fit the twilight with a cutoff of sun altitude of -11 degrees. I think it can be safely
        extrapolated farther, but be warned you may be entering a regime where it breaks down.
        """
        npts = len(filterNames)
        result = np.zeros((np.size(interpPoints), npts), dtype=float)

        good = np.where((interpPoints['sunAlt'] >= np.min(limits)) &
                        (interpPoints['sunAlt'] <= np.max(limits)) &
                        (interpPoints['airmass'] <= maxAM) &
                        (interpPoints['airmass'] >= 1.))[0]

        for i, filterName in enumerate(filterNames):
            result[good, i] = twilightFunc(interpPoints[good],
                                           *self.lsstEquations[self.filterNameDict[filterName], :].tolist())

        return {'spec': result, 'wave': self.lsstEffWave}

    def interpSpec(self, interpPoints, maxAM=3.0,
                   limits=[np.radians(-5.), np.radians(-20.)]):
        """
        interpPoints should have airmass, azRelSun, and sunAlt.
        """

        npts = np.size(self.solarWave)
        result = np.zeros((np.size(interpPoints), npts), dtype=float)

        good = np.where((interpPoints['sunAlt'] >= np.min(limits)) &
                        (interpPoints['sunAlt'] <= np.max(limits)) &
                        (interpPoints['airmass'] <= maxAM) &
                        (interpPoints['airmass'] >= 1.))[0]

        # Compute the expected flux in each of the filters that we have fits for
        fluxes = []
        for filterName in self.filterNames:
            fluxes.append(twilightFunc(interpPoints[good], *self.fitResults[filterName]))
        fluxes = np.array(fluxes)

        # ratio of model flux to raw solar flux:
        yvals = fluxes.T/(10.**(-0.4*(self.solarMag-np.log10(3631.))))

        # Find wavelengths bluer than cutoff
        blueRegion = np.where(self.solarWave < np.min(self.effWave))

        for i, yval in enumerate(yvals):
            interpF = interp1d(self.effWave, yval, bounds_error=False, fill_value=yval[-1])
            ratio = interpF(self.solarWave)
            interpBlue = InterpolatedUnivariateSpline(self.effWave, yval, k=1)
            ratio[blueRegion] = interpBlue(self.solarWave[blueRegion])
            result[good[i]] = self.solarFlux*ratio

        return {'spec': result, 'wave': self.solarWave}


class MoonInterp(BaseSingleInterp):
    """
    Read in the saved Lunar spectra and interpolate.
    """

    def __init__(self, compName='Moon', sortedOrder=['moonSunSep', 'moonAltitude', 'hpid'], mags=False):
        super(MoonInterp, self).__init__(compName=compName, sortedOrder=sortedOrder, mags=mags)
        # Magic number from when the templates were generated
        self.nside = 4

    def _weighting(self, interpPoints, values):
        """
        Weighting for the scattered moonlight.
        """

        result = np.zeros((interpPoints.size, np.size(values[0])), dtype=float)

        # Check that moonAltitude is in range, otherwise return zero array
        if np.max(interpPoints['moonAltitude']) < np.min(self.dimDict['moonAltitude']):
            return result

        # Find the neighboring healpixels
        hpids, hweights = get_neighbours(self.nside, np.pi/2.-interpPoints['alt'],
                                         interpPoints['azRelMoon'])

        badhp = np.in1d(hpids.ravel(), self.dimDict['hpid'], invert=True).reshape(hpids.shape)
        hweights[badhp] = 0.

        norm = np.sum(hweights, axis=0)
        good = np.where(norm != 0.)[0]
        hweights[:, good] = hweights[:, good]/norm[good]

        # Find the neighboring moonAltitude points in the grid
        rightMAs, leftMAs, maRightW, maLeftW = self.indxAndWeights(interpPoints['moonAltitude'],
                                                                   self.dimDict['moonAltitude'])

        # Find the neighboring moonSunSep points in the grid
        rightMss, leftMss, mssRightW, mssLeftW = self.indxAndWeights(interpPoints['moonSunSep'],
                                                                     self.dimDict['moonSunSep'])

        nhpid = self.dimDict['hpid'].size
        nMA = self.dimDict['moonAltitude'].size
        # Convert the hpid to an index.
        tmp = intid2id(hpids.ravel(), self.dimDict['hpid'],
                       np.arange(self.dimDict['hpid'].size))
        hpindx = tmp.reshape(hpids.shape)
        # loop though the hweights and the moonAltitude weights

        for hpid, hweight in zip(hpindx, hweights):
            for maid, maW in zip([rightMAs, leftMAs], [maRightW, maLeftW]):
                for mssid, mssW in zip([rightMss, leftMss], [mssRightW, mssLeftW]):
                    weight = hweight*maW*mssW
                    result += weight[:, np.newaxis]*values[mssid*nhpid*nMA+maid*nhpid+hpid]

        return result


class ZodiacalInterp(BaseSingleInterp):
    """
    Interpolate the zodiacal light based on the airmass and the healpix ID where
    the healpixels are in ecliptic coordinates, with the sun at ecliptic longitude zero
    """

    def __init__(self, compName='Zodiacal', sortedOrder=['airmass', 'hpid'], mags=False):
        super(ZodiacalInterp, self).__init__(compName=compName, sortedOrder=sortedOrder, mags=mags)
        self.nside = hp.npix2nside(np.size(np.where(self.spec['airmass'] ==
                                                    np.unique(self.spec['airmass'])[0])[0]))

    def _weighting(self, interpPoints, values):
        """
        interpPoints is a numpy array where interpolation is desired
        values are the model values.
        """
        result = np.zeros((interpPoints.size, np.size(values[0])), dtype=float)

        inRange = np.where((interpPoints['airmass'] <= np.max(self.dimDict['airmass'])) &
                           (interpPoints['airmass'] >= np.min(self.dimDict['airmass'])))
        usePoints = interpPoints[inRange]
        # Find the neighboring healpixels
        hpids, hweights = get_neighbours(self.nside, np.pi/2.-usePoints['altEclip'],
                                         usePoints['azEclipRelSun'])

        badhp = np.in1d(hpids.ravel(), self.dimDict['hpid'], invert=True).reshape(hpids.shape)
        hweights[badhp] = 0.

        norm = np.sum(hweights, axis=0)
        good = np.where(norm != 0.)[0]
        hweights[:, good] = hweights[:, good]/norm[good]

        amRightIndex, amLeftIndex, amRightW, amLeftW = self.indxAndWeights(usePoints['airmass'],
                                                                           self.dimDict['airmass'])

        nhpid = self.dimDict['hpid'].size
        # loop though the hweights and the airmass weights
        for hpid, hweight in zip(hpids, hweights):
            for amIndex, amW in zip([amRightIndex, amLeftIndex], [amRightW, amLeftW]):
                weight = hweight*amW
                result[inRange] += weight[:, np.newaxis]*values[amIndex*nhpid+hpid]

        return result
