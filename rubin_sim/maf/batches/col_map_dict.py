__all__ = ["getColMap", "ColMapDict"]


def getColMap(opsdb):
    """Get the colmap dictionary, if you already have a database object.

    Parameters
    ----------
    opsdb : rubin_sim.maf.db.Database or rubin_sim.maf.db.OpsimDatabase

    Returns
    -------
    dictionary
    """
    try:
        version = opsdb.opsimVersion
        version = "opsim" + version.lower()
    except AttributeError:
        version = "fbs"
    colmap = ColMapDict(version)
    return colmap


def ColMapDict(dictName=None):

    if dictName is None:
        dictName = "FBS"
    dictName = dictName.lower()

    if dictName == "fbs" or dictName == "opsimfbs":
        colMap = {}
        colMap["ra"] = "fieldRA"
        colMap["dec"] = "fieldDec"
        colMap["raDecDeg"] = True
        colMap["mjd"] = "observationStartMJD"
        colMap["exptime"] = "visitExposureTime"
        colMap["visittime"] = "visitTime"
        colMap["alt"] = "altitude"
        colMap["az"] = "azimuth"
        colMap["lst"] = "observationStartLST"
        colMap["filter"] = "filter"
        colMap["fiveSigmaDepth"] = "fiveSigmaDepth"
        colMap["night"] = "night"
        colMap["slewtime"] = "slewTime"
        colMap["slewdist"] = "slewDistance"
        colMap["seeingEff"] = "seeingFwhmEff"
        colMap["seeingGeom"] = "seeingFwhmGeom"
        colMap["skyBrightness"] = "skyBrightness"
        colMap["moonDistance"] = "moonDistance"
        colMap["fieldId"] = "fieldId"
        colMap["proposalId"] = "proposalId"
        colMap["slewactivities"] = {}
        colMap["metadataList"] = [
            "airmass",
            "normairmass",
            "seeingEff",
            "skyBrightness",
            "fiveSigmaDepth",
            "HA",
            "moonDistance",
            "solarElong",
            "saturation_mag",
        ]
        colMap["metadataAngleList"] = ["rotSkyPos"]
        colMap["note"] = "note"

    elif dictName == "opsimv4":
        colMap = {}
        colMap["ra"] = "fieldRA"
        colMap["dec"] = "fieldDec"
        colMap["raDecDeg"] = True
        colMap["mjd"] = "observationStartMJD"
        colMap["exptime"] = "visitExposureTime"
        colMap["visittime"] = "visitTime"
        colMap["alt"] = "altitude"
        colMap["az"] = "azimuth"
        colMap["lst"] = "observationStartLST"
        colMap["filter"] = "filter"
        colMap["fiveSigmaDepth"] = "fiveSigmaDepth"
        colMap["night"] = "night"
        colMap["slewtime"] = "slewTime"
        colMap["slewdist"] = "slewDistance"
        colMap["seeingEff"] = "seeingFwhmEff"
        colMap["seeingGeom"] = "seeingFwhmGeom"
        colMap["skyBrightness"] = "skyBrightness"
        colMap["moonDistance"] = "moonDistance"
        colMap["fieldId"] = "fieldId"
        colMap["proposalId"] = "proposalId"
        # slew speeds table
        colMap["slewSpeedsTable"] = "SlewMaxSpeeds"
        # slew states table
        colMap["slewStatesTable"] = "SlewFinalState"
        # slew activities list
        colMap["slewActivitiesTable"] = "SlewActivities"
        # Slew columns
        colMap["Dome Alt Speed"] = "domeAltSpeed"
        colMap["Dome Az Speed"] = "domeAzSpeed"
        colMap["Tel Alt Speed"] = "telAltSpeed"
        colMap["Tel Az Speed"] = "telAzSpeed"
        colMap["Rotator Speed"] = "rotatorSpeed"
        colMap["Tel Alt"] = "telAlt"
        colMap["Tel Az"] = "telAz"
        colMap["Rot Tel Pos"] = "rotTelPos"
        colMap["Dome Alt"] = "domeAlt"
        colMap["Dome Az"] = "domeAz"
        colMap["slewactivities"] = {
            "Dome Alt": "domalt",
            "Dome Az": "domaz",
            "Dome Settle": "domazsettle",
            "Tel Alt": "telalt",
            "Tel Az": "telaz",
            "Tel Rot": "telrot",
            "Tel Settle": "telsettle",
            "TelOptics CL": "telopticsclosedloop",
            "TelOptics OL": "telopticsopenloop",
            "Readout": "readout",
            "Filter": "filter",
        }
        colMap["metadataList"] = [
            "airmass",
            "normairmass",
            "seeingEff",
            "skyBrightness",
            "fiveSigmaDepth",
            "HA",
            "moonDistance",
            "solarElong",
        ]
        colMap["metadataAngleList"] = ["rotSkyPos"]

    elif dictName == "opsimv3":
        colMap = {}
        colMap["ra"] = "fieldRA"
        colMap["dec"] = "fieldDec"
        colMap["raDecDeg"] = False
        colMap["mjd"] = "expMJD"
        colMap["exptime"] = "visitExpTime"
        colMap["visittime"] = "visitTime"
        colMap["alt"] = "altitude"
        colMap["az"] = "azimuth"
        colMap["lst"] = "lst"
        colMap["filter"] = "filter"
        colMap["fiveSigmaDepth"] = "fiveSigmaDepth"
        colMap["night"] = "night"
        colMap["slewtime"] = "slewTime"
        colMap["slewdist"] = "slewDist"
        colMap["seeingEff"] = "FWHMeff"
        colMap["seeingGeom"] = "FWHMgeom"
        colMap["skyBrightness"] = "filtSkyBrightness"
        colMap["moonDistance"] = "dist2Moon"
        colMap["fieldId"] = "fieldID"
        colMap["proposalId"] = "propID"
        # slew speeds table
        colMap["slewSpeedsTable"] = "SlewMaxSpeeds"
        # slew states table
        colMap["slewStatesTable"] = "SlewStates"
        # Slew activities list
        colMap["slewActivitiesTable"] = "SlewActivities"
        colMap["Dome Alt Speed"] = "domeAltSpeed"
        colMap["Dome Az Speed"] = "domeAzSpeed"
        colMap["Tel Alt Speed"] = "telAltSpeed"
        colMap["Tel Az Speed"] = "telAzSpeed"
        colMap["Rotator Speed"] = "rotatorSpeed"
        colMap["Tel Alt"] = "telAlt"
        colMap["Tel Az"] = "telAz"
        colMap["Rot Tel Pos"] = "rotTelPos"
        colMap["Dome Alt"] = "domAlt"
        colMap["Dome Az"] = "domAz"
        colMap["slewactivities"] = {
            "Dome Alt": "DomAlt",
            "Dome Az": "DomAz",
            "Tel Alt": "TelAlt",
            "Tel Az": "TelAz",
            "Tel Rot": "Rotator",
            "Settle": "Settle",
            "TelOptics CL": "TelOpticsCL",
            "TelOptics OL": "TelOpticsOL",
            "Readout": "Readout",
            "Filter": "Filter",
        }
        colMap["metadataList"] = [
            "airmass",
            "normairmass",
            "seeingEff",
            "skyBrightness",
            "fiveSigmaDepth",
            "HA",
            "moonDistance",
            "solarElong",
        ]
        colMap["metadataAngleList"] = ["rotSkyPos"]

    elif dictName == "barebones":
        colMap = {}
        colMap["ra"] = "ra"
        colMap["dec"] = "dec"
        colMap["raDecDeg"] = True
        colMap["mjd"] = "mjd"
        colMap["exptime"] = "exptime"
        colMap["visittime"] = "exptime"
        colMap["alt"] = "alt"
        colMap["az"] = "az"
        colMap["filter"] = "filter"
        colMap["fiveSigmaDepth"] = "fivesigmadepth"
        colMap["night"] = "night"
        colMap["slewtime"] = "slewtime"
        colMap["slewdist"] = None
        colMap["seeingGeom"] = "seeing"
        colMap["seeingEff"] = "seeing"
        colMap["metadataList"] = [
            "airmass",
            "normairmass",
            "seeingEff",
            "skyBrightness",
            "fiveSigmaDepth",
            "HA",
        ]
        colMap["metadataAngleList"] = ["rotSkyPos"]

    else:
        raise ValueError(f"No built in column dict with name {dictMap}")

    return colMap
