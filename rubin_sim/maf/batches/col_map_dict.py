__all__ = ("col_map_dict",)


def col_map_dict(dict_name=None):
    if dict_name is None:
        dict_name = "FBS"
    dict_name = dict_name.lower()

    if dict_name == "fbs":
        col_map = {}
        col_map["ra"] = "fieldRA"
        col_map["dec"] = "fieldDec"
        col_map["raDecDeg"] = True
        col_map["mjd"] = "observationStartMJD"
        col_map["exptime"] = "visitExposureTime"
        col_map["visittime"] = "visitTime"
        col_map["alt"] = "altitude"
        col_map["az"] = "azimuth"
        col_map["lst"] = "observationStartLST"
        col_map["filter"] = "band"
        col_map["band"] = "band"
        col_map["fiveSigmaDepth"] = "fiveSigmaDepth"
        col_map["night"] = "night"
        col_map["slewtime"] = "slewTime"
        col_map["slewdist"] = "slewDistance"
        col_map["seeingEff"] = "seeingFwhmEff"
        col_map["seeingGeom"] = "seeingFwhmGeom"
        col_map["skyBrightness"] = "skyBrightness"
        col_map["moonDistance"] = "moonDistance"
        col_map["slewactivities"] = {}
        col_map["metadataList"] = [
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
        col_map["metadataAngleList"] = ["rotSkyPos"]
        col_map["scheduler_note"] = "scheduler_note"
        col_map["scheduler_note_root"] = "scheduler_note_root"

    elif dict_name == "consdb":
        col_map = {}
        col_map["ra"] = "s_ra"
        col_map["dec"] = "s_dec"
        col_map["raDecDeg"] = True
        col_map["mjd"] = "obs_start_mjd"
        col_map["exptime"] = "exp_time"
        col_map["visittime"] = "dark_time"
        col_map["alt"] = "altitude"
        col_map["az"] = "azimuth"
        col_map["filter"] = "band"
        col_map["band"] = "band"
        col_map["fiveSigmaDepth"] = "stats_mag_lim_median"
        col_map["night"] = "day_obs"
        col_map["slewtime"] = "slew_time"
        col_map["slewdist"] = "slew_distance"
        col_map["seeingEff"] = "fwhm_eff"
        col_map["seeingGeom"] = "fwhm_geom"
        col_map["skyBrightness"] = "sky_bg_median_mag"
        col_map["moonDistance"] = "moon_distance"
        col_map["slewactivities"] = {}
        col_map["metadataList"] = [
            "airmass",
            "normairmass",
            "fwhm_eff",
            "sky_bg_median_mag",
            "stats_mag_lim_median",
            "HA",
            "moon_distance",
            "solarElong",
            "saturation_mag",
        ]
        col_map["metadataAngleList"] = ["sky_rotation"]
        col_map["scheduler_note"] = "scheduler_note"
        col_map["scheduler_note_root"] = "scheduler_note_root"

    elif dict_name == "fbs_sim" or dict_name == "opsimfbs":
        col_map = {}
        col_map["ra"] = "fieldRA"
        col_map["dec"] = "fieldDec"
        col_map["raDecDeg"] = True
        col_map["mjd"] = "observationStartMJD"
        col_map["exptime"] = "visitExposureTime"
        col_map["visittime"] = "visitTime"
        col_map["alt"] = "altitude"
        col_map["az"] = "azimuth"
        col_map["lst"] = "observationStartLST"
        col_map["filter"] = "filter"
        col_map["fiveSigmaDepth"] = "fiveSigmaDepth"
        col_map["night"] = "night"
        col_map["slewtime"] = "slewTime"
        col_map["slewdist"] = "slewDistance"
        col_map["seeingEff"] = "seeingFwhmEff"
        col_map["seeingGeom"] = "seeingFwhmGeom"
        col_map["skyBrightness"] = "skyBrightness"
        col_map["moonDistance"] = "moonDistance"
        col_map["slewactivities"] = {}
        col_map["metadataList"] = [
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
        col_map["metadataAngleList"] = ["rotSkyPos"]
        col_map["scheduler_note"] = "scheduler_note"
        col_map["scheduler_note_root"] = "scheduler_note_root"

    elif dict_name == "opsimv4":
        col_map = {}
        col_map["ra"] = "fieldRA"
        col_map["dec"] = "fieldDec"
        col_map["raDecDeg"] = True
        col_map["mjd"] = "observationStartMJD"
        col_map["exptime"] = "visitExposureTime"
        col_map["visittime"] = "visitTime"
        col_map["alt"] = "altitude"
        col_map["az"] = "azimuth"
        col_map["lst"] = "observationStartLST"
        col_map["filter"] = "filter"
        col_map["fiveSigmaDepth"] = "fiveSigmaDepth"
        col_map["night"] = "night"
        col_map["slewtime"] = "slewTime"
        col_map["slewdist"] = "slewDistance"
        col_map["seeingEff"] = "seeingFwhmEff"
        col_map["seeingGeom"] = "seeingFwhmGeom"
        col_map["skyBrightness"] = "skyBrightness"
        col_map["moonDistance"] = "moonDistance"
        col_map["fieldId"] = "fieldId"
        col_map["proposalId"] = "proposalId"
        # slew speeds table
        col_map["slewSpeedsTable"] = "SlewMaxSpeeds"
        # slew states table
        col_map["slewStatesTable"] = "SlewFinalState"
        # slew activities list
        col_map["slewActivitiesTable"] = "SlewActivities"
        # Slew columns
        col_map["Dome Alt Speed"] = "domeAltSpeed"
        col_map["Dome Az Speed"] = "domeAzSpeed"
        col_map["Tel Alt Speed"] = "telAltSpeed"
        col_map["Tel Az Speed"] = "telAzSpeed"
        col_map["Rotator Speed"] = "rotatorSpeed"
        col_map["Tel Alt"] = "telAlt"
        col_map["Tel Az"] = "telAz"
        col_map["Rot Tel Pos"] = "rotTelPos"
        col_map["Dome Alt"] = "domeAlt"
        col_map["Dome Az"] = "domeAz"
        col_map["slewactivities"] = {
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
        col_map["metadataList"] = [
            "airmass",
            "normairmass",
            "seeingEff",
            "skyBrightness",
            "fiveSigmaDepth",
            "HA",
            "moonDistance",
            "solarElong",
        ]
        col_map["metadataAngleList"] = ["rotSkyPos"]

    elif dict_name == "opsimv3":
        col_map = {}
        col_map["ra"] = "fieldRA"
        col_map["dec"] = "fieldDec"
        col_map["raDecDeg"] = False
        col_map["mjd"] = "expMJD"
        col_map["exptime"] = "visitExpTime"
        col_map["visittime"] = "visitTime"
        col_map["alt"] = "altitude"
        col_map["az"] = "azimuth"
        col_map["lst"] = "lst"
        col_map["filter"] = "filter"
        col_map["fiveSigmaDepth"] = "fiveSigmaDepth"
        col_map["night"] = "night"
        col_map["slewtime"] = "slewTime"
        col_map["slewdist"] = "slewDist"
        col_map["seeingEff"] = "FWHMeff"
        col_map["seeingGeom"] = "FWHMgeom"
        col_map["skyBrightness"] = "filtSkyBrightness"
        col_map["moonDistance"] = "dist2Moon"
        col_map["fieldId"] = "fieldID"
        col_map["proposalId"] = "propID"
        # slew speeds table
        col_map["slewSpeedsTable"] = "SlewMaxSpeeds"
        # slew states table
        col_map["slewStatesTable"] = "SlewStates"
        # Slew activities list
        col_map["slewActivitiesTable"] = "SlewActivities"
        col_map["Dome Alt Speed"] = "domeAltSpeed"
        col_map["Dome Az Speed"] = "domeAzSpeed"
        col_map["Tel Alt Speed"] = "telAltSpeed"
        col_map["Tel Az Speed"] = "telAzSpeed"
        col_map["Rotator Speed"] = "rotatorSpeed"
        col_map["Tel Alt"] = "telAlt"
        col_map["Tel Az"] = "telAz"
        col_map["Rot Tel Pos"] = "rotTelPos"
        col_map["Dome Alt"] = "domAlt"
        col_map["Dome Az"] = "domAz"
        col_map["slewactivities"] = {
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
        col_map["metadataList"] = [
            "airmass",
            "normairmass",
            "seeingEff",
            "skyBrightness",
            "fiveSigmaDepth",
            "HA",
            "moonDistance",
            "solarElong",
        ]
        col_map["metadataAngleList"] = ["rotSkyPos"]

    elif dict_name == "barebones":
        col_map = {}
        col_map["ra"] = "ra"
        col_map["dec"] = "dec"
        col_map["raDecDeg"] = True
        col_map["mjd"] = "mjd"
        col_map["exptime"] = "exptime"
        col_map["visittime"] = "exptime"
        col_map["alt"] = "alt"
        col_map["az"] = "az"
        col_map["filter"] = "filter"
        col_map["fiveSigmaDepth"] = "fivesigmadepth"
        col_map["night"] = "night"
        col_map["slewtime"] = "slewtime"
        col_map["slewdist"] = None
        col_map["seeingGeom"] = "seeing"
        col_map["seeingEff"] = "seeing"
        col_map["metadataList"] = [
            "airmass",
            "normairmass",
            "seeingEff",
            "skyBrightness",
            "fiveSigmaDepth",
            "HA",
        ]
        col_map["metadataAngleList"] = ["rotSkyPos"]

    else:
        raise ValueError(f"No built in column dict with name {dict_name}")

    return col_map
