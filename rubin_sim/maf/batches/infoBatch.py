import rubin_sim.maf.batches as batches

__all__ = ["metadata_bundle_dicts"]


def metadata_bundle_dicts(
    allsky_slicer, wfd_slicer, opsim="opsim", colmap=batches.ColMapDict("FBS")
):

    # Set up the bundle dicts
    # Some of these metrics are reproduced in other scripts - srd and cadence
    bdict = {}

    for tag, slicer in zip(["All sky", "WFD"], [allsky_slicer, wfd_slicer]):
        fO = batches.fOBatch(
            colmap=colmap, runName=opsim, slicer=slicer, extraInfoLabel=tag
        )
        bdict.update(fO)
        astrometry = batches.astrometryBatch(
            colmap=colmap, runName=opsim, slicer=slicer, extraInfoLabel=tag
        )
        bdict.update(astrometry)
        rapidrevisit = batches.rapidRevisitBatch(
            colmap=colmap, runName=opsim, slicer=slicer, extraInfoLabel=tag
        )
        bdict.update(rapidrevisit)

    # Intranight (pairs/time)
    for tag, slicer in zip(["All sky", "WFD"], [allsky_slicer, wfd_slicer]):
        intranight = batches.intraNight(
            colmap, opsim, slicer=slicer, extraInfoLabel=tag
        )
        bdict.update(intranight)

    # Internight (nights between visits)
    for tag, slicer in zip(["All sky", "WFD"], [allsky_slicer, wfd_slicer]):
        internight = batches.interNight(
            colmap, opsim, slicer=slicer, extraInfoLabel=tag
        )
        bdict.update(internight)

    # Intraseason (length of season)
    for tag, slicer in zip(["All sky", "WFD"], [allsky_slicer, wfd_slicer]):
        season = batches.seasons(
            colmap=colmap, runName=opsim, slicer=slicer, extraInfoLabel=tag
        )
        bdict.update(season)

    # Run all metadata metrics, All and just WFD.
    for tag, slicer in zip(["All sky", "WFD"], [allsky_slicer, wfd_slicer]):
        bdict.update(
            batches.allMetadata(colmap, opsim, slicer=slicer, extraInfoLabel=tag)
        )

    # And run some metadata for the first year only - all sky
    bdict.update(batches.firstYearMetadata(colmap, opsim, slicer=allsky_slicer))

    # Nvisits + m5 maps + Teff maps, All and just WFD.
    for tag, slicer in zip(["All sky", "WFD"], [allsky_slicer, wfd_slicer]):
        bdict.update(
            batches.nvisitsM5Maps(colmap, opsim, slicer=slicer, extraInfoLabel=tag)
        )
        bdict.update(
            batches.tEffMetrics(colmap, opsim, slicer=slicer, extraInfoLabel=tag)
        )

    # And number of visits for the first year and then halfway through the survey
    bdict.update(
        batches.nvisitsM5Maps(
            colmap,
            opsim,
            slicer=allsky_slicer,
            extraInfoLabel="Yr 1",
            extraSql="night < 365.5",
            runLength=1,
        )
    )
    bdict.update(
        batches.nvisitsM5Maps(
            colmap,
            opsim,
            slicer=allsky_slicer,
            extraSql='night > 365*3.5 and night < 365*4.5 and note not like "%DD%"',
            extraInfoLabel="Yr 3.5",
            runLength=1,
        )
    )

    # Nvisits per proposal and per night.
    ### NEED MORE HERE
    # bdict.update(batches.nvisitsPerProp(opsdb, colmap, opsim,
    #                                    slicer=allsky_slicer))

    # NVisits alt/az LambertSkyMap (all filters, per filter)
    bdict.update(batches.altazLambert(colmap, opsim))

    # Slew metrics.
    bdict.update(batches.slewBasics(colmap, opsim))

    # Open shutter metrics.
    bdict.update(batches.openshutterFractions(colmap, opsim))

    # Per night and whole survey filter changes.
    bdict.update(batches.filtersPerNight(colmap, opsim, nights=1, extraSql=None))
    bdict.update(batches.filtersWholeSurvey(colmap, opsim, extraSql=None))

    return bdict
