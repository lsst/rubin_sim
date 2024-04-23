__all__ = ("info_bundle_dicts",)

import numpy as np

import rubin_sim.maf.batches as batches


def info_bundle_dicts(allsky_slicer, wfd_slicer, opsim="opsim", colmap=batches.col_map_dict()):
    # Set up the bundle dicts
    # Some of these metrics are reproduced in other scripts - srd and cadence
    bdict = {}

    for tag, slicer in zip(["All sky", "WFD"], [allsky_slicer, wfd_slicer]):
        fO = batches.fOBatch(colmap=colmap, run_name=opsim, slicer=slicer, extra_info=tag)
        bdict.update(fO)
        astrometry = batches.astrometryBatch(colmap=colmap, run_name=opsim, slicer=slicer, extra_info=tag)
        bdict.update(astrometry)
        rapidrevisit = batches.rapidRevisitBatch(colmap=colmap, run_name=opsim, slicer=slicer, extra_info=tag)
        bdict.update(rapidrevisit)

    # Intranight (pairs/time)
    for tag, slicer in zip(["All sky", "WFD"], [allsky_slicer, wfd_slicer]):
        intranight = batches.intraNight(colmap, opsim, slicer=slicer, extraInfoLabel=tag)
        bdict.update(intranight)

    # Internight (nights between visits)
    for tag, slicer in zip(["All sky", "WFD"], [allsky_slicer, wfd_slicer]):
        internight = batches.interNight(colmap, opsim, slicer=slicer, extraInfoLabel=tag)
        bdict.update(internight)

    # Intraseason (length of season)
    for tag, slicer in zip(["All sky", "WFD"], [allsky_slicer, wfd_slicer]):
        season = batches.seasons(colmap=colmap, runName=opsim, slicer=slicer, extraInfoLabel=tag)
        bdict.update(season)

    # Run all metadata metrics, All and just WFD.
    for tag, slicer in zip(["All sky", "WFD"], [allsky_slicer, wfd_slicer]):
        bdict.update(batches.allMetadata(colmap, opsim, slicer=slicer, extraInfoLabel=tag))

    # And run some metadata for the first year only - all sky
    bdict.update(batches.firstYearMetadata(colmap, opsim, slicer=allsky_slicer))

    # Nvisits + m5 maps + Teff maps, All and just WFD.
    for tag, slicer in zip(["All sky", "WFD"], [allsky_slicer, wfd_slicer]):
        bdict.update(batches.nvisitsM5Maps(colmap, opsim, slicer=slicer, extraInfoLabel=tag))
        bdict.update(batches.tEffMetrics(colmap, opsim, slicer=slicer, extraInfoLabel=tag))

    # And number of visits for the first year and halfway through the survey
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
            extraSql='night > 365*3.5 and night < 365*4.5 and scheduler_note not like "%DD%"',
            extraInfoLabel="Yr 3-4",
            runLength=1,
        )
    )

    # NVisits alt/az LambertSkyMap (all filters, per filter)
    bdict.update(batches.altazLambert(colmap, opsim))

    # Slew metrics.
    bdict.update(batches.slewBasics(colmap, opsim))

    # Open shutter metrics.
    bdict.update(batches.openshutterFractions(colmap, opsim))

    # Some basic nvisits per purpose metrics
    wfd_footprint_mask = np.where(wfd_slicer.mask, 0, 1)
    bdict.update(
        batches.nvisitsPerSubset(
            colmap,
            opsim,
            constraint="visitExposureTime > 19",
            footprintConstraint=wfd_footprint_mask,
            extraInfoLabel="WFD",
        )
    )
    dd_constraint = "scheduler_note like '%DD%'"
    bdict.update(
        batches.nvisitsPerSubset(
            colmap,
            opsim,
            constraint=dd_constraint,
            footprintConstraint=None,
            extraInfoLabel="DDF",
        )
    )
    bdict.update(
        batches.nvisitsPerSubset(
            colmap,
            opsim,
            constraint=None,
            footprintConstraint=None,
            extraInfoLabel="All visits",
        )
    )

    # Per night and whole survey filter changes.
    bdict.update(batches.filtersPerNight(colmap, opsim, nights=1, extraSql=None))
    bdict.update(batches.filtersWholeSurvey(colmap, opsim, extraSql=None))

    return bdict
