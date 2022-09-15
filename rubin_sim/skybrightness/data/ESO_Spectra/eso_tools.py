import numpy as np
from astropy.io import fits
import os, subprocess
from lsst.sims.photUtils import Sed, Bandpass
import healpy as hp
from lsst.sims.utils import angular_separation

## Tools for calling and reading things from the ESO sky model.
# Downloaded and installed from XXX.
# Installing is a major headache, let's hope we don't have to do this again anytime soon

# Run this in the sm-01_mod2 direcory to regenerate the sims_skybrightness save files.


hPlank = 6.626068e-27  # erg s
cLight = 2.99792458e10  # cm/s


def read_eso_output(filename="output/radspec.fits"):
    """Read in the output generated by sm-01_mod2/bin/calcskymodel"""

    fitsfile = fits.open(filename)
    wave = fitsfile[1].data["lam"].copy() * 1e3  # Wavelength to nm
    header = fitsfile[0].header["comment"]
    spec = fitsfile[1].data["flux"].copy()
    # Convert spectra from ph/s/m2/micron/arcsec2 to erg/s/cm2/nm/arcsec2
    spec = spec / (100.0**2) * hPlank * cLight / (wave * 1e-7) / 1e3

    return spec, wave, header


def write_config(
    outfile="config/skymodel_etc.par",
    sm_h=2.64,
    sm_hmin=2.0,
    alt=90.0,
    alpha=0.0,
    rho=180.0,
    altmoon=-90.0,
    moondist=1.0,
    pres=744.0,
    ssa=0.97,
    calcds="N",
    o3column=1.0,
    moonscal=1.0,
    lon_ecl=135.0,
    lat_ecl=90.0,
    emis_str=0.2,
    temp_str=290.0,
    msolflux=130.0,
    season=0,
    time=0,
    vac_air="vac",
    pwv=2.5,
    rtcode="L",
    resol=60000,
    filepath="data",
    inc_moon="N",
    inc_star="N",
    inc_zodi="N",
    inc_therm="N",
    inc_molec="N",
    inc_upper="N",
    inc_glow="N",
):
    f = open(outfile, "w")

    print("sm_h = %f" % sm_h, file=f)
    print("sm_hmin = %f" % sm_hmin, file=f)
    print("alt = %f" % alt, file=f)
    print("alpha = %f" % alpha, file=f)
    print("rho = %f" % rho, file=f)
    print("altmoon = %f" % altmoon, file=f)
    print("moondist = %f" % moondist, file=f)
    print("pres = %f" % pres, file=f)
    print("ssa = %f" % ssa, file=f)
    print("calcds = %s" % calcds, file=f)
    print("o3column = %f" % o3column, file=f)
    print("moonscal = %f" % moonscal, file=f)
    print("lon_ecl = %f" % lon_ecl, file=f)
    print("lat_ecl = %f" % lat_ecl, file=f)
    print("emis_str = %f" % emis_str, file=f)
    print("temp_str = %f" % temp_str, file=f)
    print("msolflux = %f" % msolflux, file=f)
    print("season = %i" % season, file=f)
    print("time = %i" % time, file=f)
    print("vac_air = %s" % vac_air, file=f)
    print("pwv = %f" % pwv, file=f)
    print("rtcode = %s" % rtcode, file=f)
    print("resol = %i" % resol, file=f)
    print("filepath = %s" % filepath, file=f)

    inc = inc_moon + inc_star + inc_zodi + inc_therm + inc_molec + inc_upper + inc_glow

    print("incl = %s" % inc, file=f)

    f.close()


def call_calcskymodel():
    subprocess.run(["bin/calcskymodel"])


def spec2mags(spectra_list, wave):
    # Load LSST filters
    throughPath = os.getenv("LSST_THROUGHPUTS_BASELINE")
    keys = ["u", "g", "r", "i", "z", "y"]

    dtype = [("mags", "float", (6))]
    result = np.zeros(len(spectra_list), dtype=dtype)

    nfilt = len(keys)
    filters = {}
    for filtername in keys:
        bp = np.loadtxt(
            os.path.join(throughPath, "total_" + filtername + ".dat"),
            dtype=list(zip(["wave", "trans"], [float] * 2)),
        )
        tempB = Bandpass()
        tempB.setBandpass(bp["wave"], bp["trans"])
        filters[filtername] = tempB

    filterwave = np.array([filters[f].calcEffWavelen()[0] for f in keys])

    for i, spectrum in enumerate(spectra_list):
        tempSed = Sed()
        tempSed.setSED(wave, flambda=spectrum)
        for j, filtName in enumerate(keys):
            try:
                result["mags"][i][j] = tempSed.calcMag(filters[filtName])
            except:
                pass
    return result, filterwave


def generate_airglow(outDir=None):
    if outDir is None:
        dataDir = os.getenv("SIMS_SKYBRIGHTNESS_DATA_DIR")
        outDir = os.path.join(dataDir, "ESO_Spectra/Airglow")

    ams = np.array([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.5, 3.0])
    specs = []

    alts = 90.0 - np.degrees(np.arccos(1.0 / ams))
    airmasses = []
    solarFlux = []

    for alt, am in zip(alts, ams):
        write_config(alt=alt, inc_glow="Y")
        call_calcskymodel()
        spec, wave, header = read_eso_output()
        specs.append(spec)
        airmasses.append(am)
        # Not doing a range of these this time. I suppose I could.
        solarFlux.append(130)

    mags, filterwave = spec2mags(specs, wave)
    nwave = wave.size
    nspec = len(specs)

    dtype = [
        ("airmass", "float"),
        ("solarFlux", "float"),
        ("spectra", "float", (nwave)),
        ("mags", "float", (6)),
    ]
    spectra = np.zeros(nspec, dtype=dtype)
    spectra["airmass"] = airmasses
    spectra["solarFlux"] = solarFlux
    spectra["spectra"] = specs
    spectra["mags"] = mags["mags"]

    spectra.sort(order=["airmass", "solarFlux"])

    np.savez(
        os.path.join(outDir, "airglowSpectra.npz"),
        wave=wave,
        spec=spectra,
        filterWave=filterwave,
    )


def generate_loweratm(outDir=None):
    if outDir is None:
        dataDir = os.getenv("SIMS_SKYBRIGHTNESS_DATA_DIR")
        outDir = os.path.join(dataDir, "ESO_Spectra/LowerAtm")

    ams = np.array([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.5, 3.0])
    specs = []

    alts = 90.0 - np.degrees(np.arccos(1.0 / ams))
    airmasses = []
    nightTimes = []

    for alt, am in zip(alts, ams):
        write_config(alt=alt, inc_molec="Y")
        call_calcskymodel()
        spec, wave, header = read_eso_output()
        specs.append(spec)
        airmasses.append(am)
        # Not doing a range of these this time. I suppose I could.
        nightTimes.append(0)

    mags, filterwave = spec2mags(specs, wave)
    nwave = wave.size
    nspec = len(specs)

    dtype = [
        ("airmass", "float"),
        ("nightTimes", "float"),
        ("spectra", "float", (nwave)),
        ("mags", "float", (6)),
    ]
    spectra = np.zeros(nspec, dtype=dtype)
    spectra["airmass"] = airmasses
    spectra["nightTimes"] = nightTimes
    spectra["spectra"] = specs
    spectra["mags"] = mags["mags"]

    spectra.sort(order=["airmass", "nightTimes"])

    np.savez(
        os.path.join(outDir, "Spectra.npz"),
        wave=wave,
        spec=spectra,
        filterWave=filterwave,
    )


def merged_spec():
    dataDir = os.getenv("SIMS_SKYBRIGHTNESS_DATA_DIR")
    outDir = os.path.join(dataDir, "ESO_Spectra/MergedSpec")

    # A large number of the background components only depend on Airmass, so we can merge those together

    npzs = [
        "LowerAtm/Spectra.npz",
        "ScatteredStarLight/scatteredStarLight.npz",
        "UpperAtm/Spectra.npz",
    ]
    files = [os.path.join(dataDir, "ESO_Spectra", npz) for npz in npzs]
    temp = np.load(files[0])
    wave = temp["wave"].copy()
    spec = temp["spec"].copy()
    spec["spectra"] = spec["spectra"] * 0.0
    spec["mags"] = spec["mags"] * 0.0

    for filename in files:
        restored = np.load(filename)
        spec["spectra"] += restored["spec"]["spectra"]
        flux = 10.0 ** (-0.4 * (restored["spec"]["mags"] - np.log10(3631.0)))
        flux[np.where(restored["spec"]["mags"] == 0.0)] = 0.0
        spec["mags"] += flux

    spec["mags"] = -2.5 * np.log10(spec["mags"]) + np.log10(3631.0)

    np.savez(
        os.path.join(outDir, "mergedSpec.npz"),
        spec=spec,
        wave=wave,
        filterWave=temp["filterWave"],
    )


def generate_moon(outDir=None):
    if outDir is None:
        dataDir = os.getenv("SIMS_SKYBRIGHTNESS_DATA_DIR")
        outDir = os.path.join(dataDir, "ESO_Spectra/Moon")

    nside = 4
    hpids = np.arange(hp.nside2npix(nside))
    lat, az = hp.pix2ang(nside, hpids)
    alt = np.pi / 2.0 - lat
    airmass = 1.0 / np.cos(np.pi / 2.0 - alt)

    # Only need low airmass and then 1/2 to sky
    good = np.where((az >= 0) & (az <= np.pi) & (airmass <= 3.01) & (airmass >= 1.0))
    airmass = airmass[good]
    alt = np.degrees(alt[good])
    az = np.degrees(az[good])
    hpids = hpids[good]

    moonSunSeps = np.array(
        [0.0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180], dtype=float
    )
    moonAlts = np.array([-1.0, 0, 15, 30, 45, 60, 75, 90], dtype=float)

    specs = []
    final_moonSunSep = []
    final_hpid = []
    final_moonAlt = []

    for moonSunSep in moonSunSeps:
        for moonAlt in moonAlts:
            angDists = angular_separation(0.0, moonAlt, az, alt)
            for salt, saz, am, angDist, hpid in zip(alt, az, airmass, angDists, hpids):
                write_config(
                    alpha=moonSunSep,
                    alt=salt,
                    altmoon=moonAlt,
                    rho=angDist,
                    inc_moon="Y",
                )
                call_calcskymodel()
                spec, wave, header = read_eso_output()
                specs.append(spec)
                final_moonSunSep.append(moonSunSep)
                final_hpid.append(hpid)
                final_moonAlt.append(moonAlt)

    mags, filterwave = spec2mags(specs, wave)
    nwave = wave.size
    nspec = len(specs)

    dtype = [
        ("hpid", "int"),
        ("moonAltitude", "float"),
        ("moonSunSep", "float"),
        ("spectra", "float", (nwave)),
        ("mags", "float", (6)),
    ]
    spectra = np.zeros(nspec, dtype=dtype)
    spectra["hpid"] = final_hpid
    spectra["moonAltitude"] = final_moonAlt
    spectra["moonSunSep"] = final_moonSunSep
    spectra["spectra"] = specs
    spectra["mags"] = mags["mags"]

    spectra.sort(order=["moonSunSep", "moonAltitude", "hpid"])

    nbreak = 5
    indices = np.linspace(0, spectra.size, nbreak + 1, dtype=int)

    for i in np.arange(nbreak):
        np.savez(
            os.path.join(outDir, "moonSpectra_" + str(i) + ".npz"),
            wave=wave,
            spec=spectra[indices[i] : indices[i + 1]],
            filterWave=filterwave,
        )


def generate_scatteredStar(outDir=None):
    if outDir is None:
        dataDir = os.getenv("SIMS_SKYBRIGHTNESS_DATA_DIR")
        outDir = os.path.join(dataDir, "ESO_Spectra/ScatteredStarLight")

    ams = np.array([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.5, 3.0])
    specs = []

    alts = 90.0 - np.degrees(np.arccos(1.0 / ams))
    airmasses = []
    nightTimes = []

    for alt, am in zip(alts, ams):
        write_config(alt=alt, inc_star="Y")
        call_calcskymodel()
        spec, wave, header = read_eso_output()
        specs.append(spec)
        airmasses.append(am)
        # Not doing a range of these this time. I suppose I could.
        nightTimes.append(0)

    mags, filterwave = spec2mags(specs, wave)
    nwave = wave.size
    nspec = len(specs)

    dtype = [
        ("airmass", "float"),
        ("nightTimes", "float"),
        ("spectra", "float", (nwave)),
        ("mags", "float", (6)),
    ]
    spectra = np.zeros(nspec, dtype=dtype)
    spectra["airmass"] = airmasses
    spectra["nightTimes"] = nightTimes
    spectra["spectra"] = specs
    spectra["mags"] = mags["mags"]

    spectra.sort(order=["airmass", "nightTimes"])

    np.savez(
        os.path.join(outDir, "scatteredStarLight.npz"),
        wave=wave,
        spec=spectra,
        filterWave=filterwave,
    )


def generate_upperatm(outDir=None):
    if outDir is None:
        dataDir = os.getenv("SIMS_SKYBRIGHTNESS_DATA_DIR")
        outDir = os.path.join(dataDir, "ESO_Spectra/UpperAtm")

    ams = np.array([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.5, 3.0])
    specs = []

    alts = 90.0 - np.degrees(np.arccos(1.0 / ams))
    airmasses = []
    nightTimes = []

    for alt, am in zip(alts, ams):
        write_config(alt=alt, inc_upper="Y")
        call_calcskymodel()
        spec, wave, header = read_eso_output()
        specs.append(spec)
        airmasses.append(am)
        # Not doing a range of these this time. I suppose I could.
        nightTimes.append(0)

    mags, filterwave = spec2mags(specs, wave)
    nwave = wave.size
    nspec = len(specs)

    dtype = [
        ("airmass", "float"),
        ("nightTimes", "float"),
        ("spectra", "float", (nwave)),
        ("mags", "float", (6)),
    ]
    spectra = np.zeros(nspec, dtype=dtype)
    spectra["airmass"] = airmasses
    spectra["nightTimes"] = nightTimes
    spectra["spectra"] = specs
    spectra["mags"] = mags["mags"]

    spectra.sort(order=["airmass", "nightTimes"])

    np.savez(
        os.path.join(outDir, "Spectra.npz"),
        wave=wave,
        spec=spectra,
        filterWave=filterwave,
    )


def generate_zodi(outDir=None):
    if outDir is None:
        dataDir = os.getenv("SIMS_SKYBRIGHTNESS_DATA_DIR")
        outDir = os.path.join(dataDir, "ESO_Spectra/Zodiacal")

    ams = np.array([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.5, 3.0])
    specs = []
    final_hpid = []
    nside = 4
    hpids = np.arange(hp.nside2npix(nside))
    theta, phi = hp.pix2ang(nside, hpids)
    lon = np.degrees(phi)
    lon[np.where(lon > 180.0)] = lon[np.where(lon > 180.0)] - 360.0
    lat = np.degrees(theta) - 90.0

    alts = 90.0 - np.degrees(np.arccos(1.0 / ams))
    airmasses = []

    for la, lo, hpi in zip(lat, lon, hpids):
        for alt, am in zip(alts, ams):
            write_config(alt=alt, lon_ecl=lo, lat_ecl=la, inc_zodi="Y")
            call_calcskymodel()
            spec, wave, header = read_eso_output()
            specs.append(spec)
            airmasses.append(am)
            final_hpid.append(hpi)

    mags, filterwave = spec2mags(specs, wave)
    nwave = wave.size
    nspec = len(specs)

    dtype = [
        ("airmass", "float"),
        ("hpid", "int"),
        ("spectra", "float", (nwave)),
        ("mags", "float", (6)),
    ]
    spectra = np.zeros(nspec, dtype=dtype)
    spectra["airmass"] = airmasses
    spectra["hpid"] = final_hpid
    spectra["spectra"] = specs
    spectra["mags"] = mags["mags"]

    spectra.sort(order=["airmass", "hpid"])

    # span this over multiple files to store in github
    nbreak = 3
    indices = np.linspace(0, spectra.size, nbreak + 1, dtype=int)

    for i in np.arange(nbreak):
        np.savez(
            os.path.join(outDir, "zodiacalSpectra_" + str(i) + ".npz"),
            wave=wave,
            spec=spectra[indices[i] : indices[i + 1]],
            filterWave=filterwave,
        )


def recalc_mags():
    # XXX--todo, make a function that loads up all the spectra and recalculates the magnitudes
    pass


if __name__ == "__main__":
    generate_airglow()
    generate_loweratm()
    generate_moon()
    generate_scatteredStar()
    generate_upperatm()
    generate_zodi()
    merged_spec()
