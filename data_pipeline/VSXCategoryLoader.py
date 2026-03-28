import re
import random
from collections import defaultdict

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord

from astroquery.mast import Catalogs
import lightkurve as lk


class VSXCategoryLoader:
    def __init__(self):
        """
        VSXCategoryLoader provides a starter curated dictionary that maps raw
        AAVSO VSX variable-star types into broader normalized families that are
        easier to use for TESS analysis and machine-learning pipelines.

        Core design idea
        ----------------
        The raw VSX type system is astrophysically meaningful but can be too
        granular or inconsistent for a first-pass research pipeline. This class
        adds a curated family layer and provides a family-driven loader:

            input  -> {"ECLIPSING": 100, "RRLYR": 50}
            output -> {
                         "ECLIPSING": [starRecord1, starRecord2, ...],
                         "RRLYR": [starRecord3, starRecord4, ...]
                      }

        Important note
        --------------
        loadCategories() is intentionally family-driven. It is designed to
        retrieve a requested number of variable stars for each normalized family.
        """

        # ------------------------------------------------------------
        # Raw VSX type -> normalized family
        # ------------------------------------------------------------
        self.vsxTypeToFamily = {
            # --------------------------------------------------------
            # ECLIPSING BINARIES
            # --------------------------------------------------------
            # Binary-star systems whose brightness variations are caused by
            # eclipses along our line of sight. These are among the most
            # important classes for TESS because they are strongly periodic,
            # often high signal-to-noise, and can resemble exoplanet transits.
            #
            # EA = Algol-type detached binary
            # EB = Beta Lyrae-type semi-detached binary
            # EW = W UMa-type contact binary
            # E  = generic eclipsing binary
            "EA": "ECLIPSING",
            "EB": "ECLIPSING",
            "EW": "ECLIPSING",
            "E": "ECLIPSING",

            # --------------------------------------------------------
            # ROTATIONAL / ELLIPSOIDAL VARIABLES
            # --------------------------------------------------------
            # Variables driven by rotation, star spots, magnetic structure,
            # or tidal distortion rather than eclipses. These are often
            # periodic or quasi-periodic and can look sinusoidal in light
            # curves, making them important comparison cases for pulsators.
            #
            # ELL = ellipsoidal variable
            # RS  = RS CVn active binary
            # BY  = BY Dra spotted dwarf
            # ACV = Alpha2 CVn magnetic variable
            # ROT = generic rotational variable
            "ELL": "ELLIPSOIDAL_ROT",
            "RS": "ELLIPSOIDAL_ROT",
            "BY": "ELLIPSOIDAL_ROT",
            "ACV": "ELLIPSOIDAL_ROT",
            "ROT": "ELLIPSOIDAL_ROT",

            # --------------------------------------------------------
            # RR LYRAE
            # --------------------------------------------------------
            # Old, low-mass radial pulsators on the horizontal branch.
            # These are classic benchmark stars for period-finding and
            # Galactic-structure work, and they often have clean TESS signals.
            #
            # RRAB = fundamental mode
            # RRC  = first overtone
            # RRD  = double-mode
            # RR   = generic RR Lyrae
            "RR": "RRLYR",
            "RRAB": "RRLYR",
            "RRC": "RRLYR",
            "RRD": "RRLYR",

            # --------------------------------------------------------
            # DELTA SCUTI / SX PHOENICIS
            # --------------------------------------------------------
            # Short-period pulsators that can show high-frequency and sometimes
            # multi-mode behavior. These are very useful for testing whether a
            # pipeline can recover shorter timescales reliably.
            #
            # DSCT  = Delta Scuti
            # HADS  = High-Amplitude Delta Scuti
            # SXPHE = SX Phoenicis
            "DSCT": "DSCT_SXPHE",
            "HADS": "DSCT_SXPHE",
            "SXPHE": "DSCT_SXPHE",

            # --------------------------------------------------------
            # CEPHEIDS
            # --------------------------------------------------------
            # Classical and related pulsators with longer periods and strong
            # astrophysical importance because of the period-luminosity relation.
            #
            # DCEP  = Classical Cepheid
            # DCEPS = short-period Cepheid subtype
            # CWA   = Type II Cepheid subtype
            # CWB   = Type II Cepheid subtype
            # ACEP  = Anomalous Cepheid
            "DCEP": "CEPHEID",
            "DCEPS": "CEPHEID",
            "CWA": "CEPHEID",
            "CWB": "CEPHEID",
            "ACEP": "CEPHEID",

            # --------------------------------------------------------
            # LONG-PERIOD VARIABLES
            # --------------------------------------------------------
            # Evolved giant and supergiant stars with long-timescale
            # variability. These can be regular, semi-regular, or irregular,
            # and they are important edge cases because TESS baselines may not
            # fully cover their variability cycles.
            #
            # M   = Mira
            # SRA = semiregular A
            # SRB = semiregular B
            # LB  = slow irregular red variable
            # LPV = generic long-period variable
            "M": "LONG_PERIOD",
            "SRA": "LONG_PERIOD",
            "SRB": "LONG_PERIOD",
            "LB": "LONG_PERIOD",
            "LPV": "LONG_PERIOD",

            # --------------------------------------------------------
            # YOUNG STELLAR OBJECTS
            # --------------------------------------------------------
            # Pre-main-sequence stars whose brightness variations may be caused
            # by accretion, occultation by disk material, magnetic spots, or
            # eruptive behavior. These are often hard ML cases because they may
            # be irregular rather than strictly periodic.
            #
            # TTS  = T Tauri star
            # CTTS = classical T Tauri
            # WTTS = weak-lined T Tauri
            # UXOR = UX Orionis type
            # FUOR = FU Orionis type
            "TTS": "YSO",
            "CTTS": "YSO",
            "WTTS": "YSO",
            "UXOR": "YSO",
            "FUOR": "YSO",

            # --------------------------------------------------------
            # CATACLYSMIC VARIABLES
            # --------------------------------------------------------
            # Interacting binaries with a white dwarf accreting matter.
            # These can show outbursts, eruptions, disk-instability behavior,
            # and more complex light-curve patterns than simple periodic stars.
            #
            # CV = generic cataclysmic variable
            # UG = dwarf nova
            # N  = nova
            "CV": "CV",
            "UG": "CV",
            "N": "CV",

            # --------------------------------------------------------
            # X-RAY BINARIES / HIGH-ENERGY SYSTEMS
            # --------------------------------------------------------
            # Systems involving compact objects such as neutron stars or black
            # holes, where optical variability may reflect accretion, orbital
            # modulation, disk physics, or irradiation. These are rare but
            # scientifically important complex cases.
            #
            # X    = generic X-ray source
            # HMXB = high-mass X-ray binary
            # LMXB = low-mass X-ray binary
            "X": "XRAY",
            "HMXB": "XRAY",
            "LMXB": "XRAY",
        }

        # ------------------------------------------------------------
        # Normalized family metadata
        # ------------------------------------------------------------
        self.familyMetadata = {
            "ECLIPSING": {
                "displayName": "Eclipsing Binaries",
                "physicalMechanism": "Geometric eclipse in a binary system",
                "signalType": "Strongly periodic and often non-sinusoidal",
                "recommendedAlgorithm": "BLS",
                "typicalDifficulty": "easy",
                "tessUseCase": "Excellent for eclipse and transit-like event detection",
                "mlNotes": "Distinct morphology and strong benchmark class"
            },
            "ELLIPSOIDAL_ROT": {
                "displayName": "Rotational / Ellipsoidal Variables",
                "physicalMechanism": "Rotation, spots, or tidal distortion",
                "signalType": "Periodic or quasi-periodic and often smooth",
                "recommendedAlgorithm": "LombScargle",
                "typicalDifficulty": "medium",
                "tessUseCase": "Good for smooth periodic recovery tests",
                "mlNotes": "Can overlap morphologically with pulsators"
            },
            "RRLYR": {
                "displayName": "RR Lyrae Stars",
                "physicalMechanism": "Radial pulsation of horizontal-branch stars",
                "signalType": "Periodic and often asymmetric",
                "recommendedAlgorithm": "LombScargle",
                "typicalDifficulty": "easy",
                "tessUseCase": "Excellent benchmark for period recovery",
                "mlNotes": "Highly recognizable and astrophysically important"
            },
            "DSCT_SXPHE": {
                "displayName": "Delta Scuti / SX Phoenicis",
                "physicalMechanism": "Short-period stellar pulsation",
                "signalType": "Periodic, high-frequency, sometimes multi-mode",
                "recommendedAlgorithm": "LombScargle",
                "typicalDifficulty": "medium",
                "tessUseCase": "Useful for high-frequency recovery studies",
                "mlNotes": "Often benefits from careful frequency-domain features"
            },
            "CEPHEID": {
                "displayName": "Cepheids",
                "physicalMechanism": "Coherent radial pulsation",
                "signalType": "Periodic and relatively regular",
                "recommendedAlgorithm": "LombScargle",
                "typicalDifficulty": "easy",
                "tessUseCase": "Strong classical pulsator benchmark",
                "mlNotes": "Historically important and morphologically structured"
            },
            "LONG_PERIOD": {
                "displayName": "Long-Period Variables",
                "physicalMechanism": "Pulsation and envelope dynamics in evolved stars",
                "signalType": "Long timescale, semi-regular, or irregular",
                "recommendedAlgorithm": "LombScargle",
                "typicalDifficulty": "hard",
                "tessUseCase": "Useful edge case because TESS baseline may be limited",
                "mlNotes": "Long periods may reduce completeness in short baselines"
            },
            "YSO": {
                "displayName": "Young Stellar Objects",
                "physicalMechanism": "Accretion, occultation, disk effects, magnetic activity",
                "signalType": "Irregular or semi-periodic",
                "recommendedAlgorithm": "ML / custom features",
                "typicalDifficulty": "hard",
                "tessUseCase": "Stress-test class for irregular variability",
                "mlNotes": "Important for non-periodic classification experiments"
            },
            "CV": {
                "displayName": "Cataclysmic Variables",
                "physicalMechanism": "Accretion onto a white dwarf",
                "signalType": "Irregular, eruptive, or hybrid periodic",
                "recommendedAlgorithm": "ML / event detection",
                "typicalDifficulty": "hard",
                "tessUseCase": "Useful for accretion-driven outburst behavior",
                "mlNotes": "Often does not fit simple periodic taxonomy"
            },
            "XRAY": {
                "displayName": "X-ray Binaries / High-Energy Systems",
                "physicalMechanism": "Accretion onto compact objects",
                "signalType": "Complex, noisy, or hybrid modulation",
                "recommendedAlgorithm": "ML / custom analysis",
                "typicalDifficulty": "hard",
                "tessUseCase": "Rare but scientifically valuable special class",
                "mlNotes": "Good rare-class challenge set"
            },
            "UNKNOWN": {
                "displayName": "Unknown / Unmapped",
                "physicalMechanism": "Unspecified",
                "signalType": "Unspecified",
                "recommendedAlgorithm": "Review manually",
                "typicalDifficulty": "unknown",
                "tessUseCase": "Requires inspection",
                "mlNotes": "Potential future expansion target"
            }
        }

        # ------------------------------------------------------------
        # Reverse mapping: family -> raw VSX types
        # ------------------------------------------------------------
        self.familyToVsxTypes = defaultdict(list)
        for vsxType, family in self.vsxTypeToFamily.items():
            self.familyToVsxTypes[family].append(vsxType)

        # ------------------------------------------------------------
        # Mock catalog used for offline structure testing
        # ------------------------------------------------------------
        self.mockCatalog = self._buildMockCatalog()

    # ============================================================
    # Internal helpers
    # ============================================================

    def _parseVsxType(self, vsxType):
        """
        Parse a possibly composite VSX type string.

        Example:
            'EA/DSCT|RRAB+ROT' -> ['EA', 'RRAB', 'ROT']

        Notes:
        - '|' often means uncertain alternative
        - '+' often means multiple variability behaviors
        - '/' often indicates compound notation; this starter mapper keeps
          the leading token before '/'
        """
        if not isinstance(vsxType, str):
            return []

        parts = re.split(r"[|+]", vsxType)
        return [
            part.strip().split("/")[0].strip()
            for part in re.split(r"[|+]", vsxType)
            if part.strip()
        ]

    def _mapVsxTypeToFamilies(self, vsxType):
        """
        Map a raw VSX type string to all matching normalized families.

        Returns:
            list[str | None]
            - Each position corresponds to a parsed VSX type
            - None indicates an unknown/unmapped type
        """
        parsedTypes = self._parseVsxType(vsxType)

        return [
            self.vsxTypeToFamily[parsedType]
            if parsedType in self.vsxTypeToFamily
            else None
            for parsedType in parsedTypes
        ]

    def _pickPrimaryFamily(self, families):
        """
        Pick a single primary family from a list of candidate families.

        This is useful for single-label ML datasets. The priority order here
        is a research choice and can be changed later if your project evolves.
        """
        priority = [
            "ECLIPSING",
            "RRLYR",
            "CEPHEID",
            "DSCT_SXPHE",
            "LONG_PERIOD",
            "ELLIPSOIDAL_ROT",
            "YSO",
            "CV",
            "XRAY",
            "UNKNOWN"
        ]

        for family in priority:
            if family in families:
                return family

        return "UNKNOWN"

    def _safeTableValue(self, row, columnName, defaultValue=None):
        """
        Safely extract a value from an Astropy row-like object.
        """
        try:
            return row[columnName]
        except Exception:
            return defaultValue

    def _buildMockCatalog(self):
        """
        Build a small mock catalog for structure testing.

        Each record mimics a minimal VSX-derived star record. In real usage,
        you would replace this with actual VSX records downloaded from your
        local file, CSV export, or a custom VSX query layer.
        """
        mockCatalog = []
        rng = random.Random(42)

        return [
            {
                "starId": f"{vsxType}_{i}",
                "starName": f"{vsxType}_STAR_{i}",
                "vsxType": vsxType,
                "family": family,
                "raDeg": 10.0 + rng.random() * 300.0,
                "decDeg": -70.0 + rng.random() * 140.0
            }
            for family, vsxTypes in self.familyToVsxTypes.items()
            for vsxType in vsxTypes
            for i in range(60)
        ]

    def _filterRecordsByVsxTypes(self, records, vsxTypes):
        """
        Return records whose raw VSX type belongs to the requested raw VSX set.

        This method supports both simple raw types like 'EA' and composite raw
        types such as 'EA|DSCT' by parsing the record's vsxType field.
        """
        vsxTypeSet = set(vsxTypes)
        return [
            record
            for record in records
            if any(
                parsedType in vsxTypeSet
                for parsedType in self._parseVsxType(record.get("vsxType", ""))
            )
        ]

    def _loadStarsForFamily(self, family, count, records):
        """
        Load a requested number of stars for one normalized family.

        Parameters
        ----------
        family : str
            Requested normalized family, such as 'ECLIPSING'
        count : int
            Number of stars requested
        records : list[dict]
            Source records to search

        Returns
        -------
        list[dict]
            A list of star records belonging to that family
        """
        if family not in self.familyToVsxTypes:
            return []

        rawVsxTypes = self.familyToVsxTypes[family]
        matchedRecords = self._filterRecordsByVsxTypes(records, rawVsxTypes)

        if len(matchedRecords) > count:
            matchedRecords = random.sample(matchedRecords, count)

        return [
            {**record, "family": family}
            for record in matchedRecords
        ]

    # ============================================================
    # Public metadata helpers
    # ============================================================

    def getSupportedFamilies(self):
        """
        Return the sorted list of supported normalized families.
        """
        return sorted(self.familyMetadata.keys())

    def getFamilyMetadata(self, family):
        """
        Return metadata for one normalized family.
        """
        return self.familyMetadata.get(family, self.familyMetadata["UNKNOWN"])

    def mapVsxTypeToPrimaryFamily(self, vsxType):
        """
        Convenience wrapper for raw VSX type -> single primary family.
        """
        families = self._mapVsxTypeToFamilies(vsxType)
        return self._pickPrimaryFamily(families)

    # ============================================================
    # Family-driven category loading
    # ============================================================

    def loadCategories(self, categoryRequests, records=None, useMock=True):
        """
        Load variable stars by requested normalized family.

        This is the main method whose contract is:

            input:
                {
                    "ECLIPSING": 100,
                    "RRLYR": 50
                }

            output:
                {
                    "ECLIPSING": [starRecord1, starRecord2, ...],
                    "RRLYR": [starRecord3, starRecord4, ...]
                }

        Parameters
        ----------
        categoryRequests : dict[str, int]
            Dictionary mapping normalized family name to the number of stars
            requested for that family.

        records : list[dict] or None
            Source star records to search. Required when useMock=False.

            Expected record schema:
            {
                "starId": ...,
                "starName": ...,
                "vsxType": ...,
                "raDeg": ...,
                "decDeg": ...
            }

        useMock : bool
            If True, search the internal mock catalog.
            If False, search the provided records list.

        Returns
        -------
        dict[str, list[dict]]
            A dictionary whose keys are requested families and whose values
            are lists of matching variable-star records.
        """
        if useMock:
            sourceRecords = self.mockCatalog
        else:
            if records is None:
                raise ValueError("When useMock=False, you must provide records.")
            sourceRecords = records

        result = {}

        for family, count in categoryRequests.items():
            if family not in self.familyToVsxTypes:
                print(f"[WARNING] Unsupported family requested: {family}")
                result[family] = []
                continue

            result[family] = self._loadStarsForFamily(
                family=family,
                count=count,
                records=sourceRecords
            )

        return result

    def buildBalancedCategoryDictionary(self, categoryDictionary, starsPerCategory, randomSeed=42):
        """
        Downsample each family list to a common size for balanced ML work.
        """
        rng = random.Random(randomSeed)
        return {
            family: (
                rng.sample(stars, starsPerCategory)
                if len(stars) > starsPerCategory
                else list(stars)
            )
            for family, stars in categoryDictionary.items()
        }

    # ============================================================
    # TIC crossmatch
    # ============================================================

    def crossmatchToTic(self, starRecord, radiusArcsec=5.0):
        """
        Crossmatch one star record to the TESS Input Catalog using RA/Dec.

        Returns
        -------
        dict or None
            A copy of the input record with TIC fields appended,
            or None if no TIC match is found.
        """
        raDeg = starRecord.get("raDeg")
        decDeg = starRecord.get("decDeg")

        if raDeg is None or decDeg is None:
            return None

        coord = SkyCoord(ra=raDeg * u.deg, dec=decDeg * u.deg, frame="icrs")

        try:
            ticTable = Catalogs.query_region(
                coord,
                radius=radiusArcsec * u.arcsec,
                catalog="TIC"
            )
        except Exception as exc:
            print(f"[WARNING] TIC crossmatch failed for {starRecord.get('starName')}: {exc}")
            return None

        if ticTable is None or len(ticTable) == 0:
            return None

        try:
            ticTable.sort("distance")
        except Exception:
            pass

        bestMatch = ticTable[0]

        matchedRecord = dict(starRecord)
        matchedRecord["ticId"] = self._safeTableValue(bestMatch, "ID")
        matchedRecord["ticRaDeg"] = self._safeTableValue(bestMatch, "ra")
        matchedRecord["ticDecDeg"] = self._safeTableValue(bestMatch, "dec")
        matchedRecord["ticTmag"] = self._safeTableValue(bestMatch, "Tmag")
        matchedRecord["ticDistanceArcmin"] = self._safeTableValue(bestMatch, "distance")

        return matchedRecord

    def crossmatchCategoryDictionaryToTic(self, categoryDictionary, radiusArcsec=5.0):
        """
        Crossmatch all stars in a family->stars dictionary to TIC.
        """
        result = {}

        for family, stars in categoryDictionary.items():
            result[family] = [
                matched
                for star in stars
                if (matched := self.crossmatchToTic(star, radiusArcsec)) is not None
            ]

        return result

    # ============================================================
    # Lightkurve integration
    # ============================================================

    def downloadTessLightCurve(
        self,
        ticId,
        author="SPOC",
        exptime=120,
        sector=None,
        fluxColumn="pdcsap_flux",
        stitch=True
    ):
        """
        Download a TESS light curve for a TIC target using Lightkurve.
        """
        targetName = f"TIC {ticId}"

        try:
            searchResult = lk.search_lightcurve(
                target=targetName,
                mission="TESS",
                author=author,
                exptime=exptime,
                sector=sector
            )
        except Exception as exc:
            print(f"[WARNING] search_lightcurve failed for {targetName}: {exc}")
            return None

        if searchResult is None or len(searchResult) == 0:
            return None

        try:
            if stitch:
                collection = searchResult.download_all()
                if collection is None or len(collection) == 0:
                    return None
                lightCurve = collection.stitch()
            else:
                lightCurve = searchResult[0].download()
        except Exception as exc:
            print(f"[WARNING] download failed for {targetName}: {exc}")
            return None

        if lightCurve is None:
            return None

        try:
            if hasattr(lightCurve, "select_flux"):
                lightCurve = lightCurve.select_flux(fluxColumn)
        except Exception:
            pass

        return lightCurve

    def preprocessLightCurve(
        self,
        lightCurve,
        doRemoveNans=True,
        doNormalize=True,
        doFlatten=False,
        flattenWindowLength=401,
        doRemoveOutliers=False,
        sigma=5.0,
        doBin=False,
        timeBinSize=0.01
    ):
        """
        Standard preprocessing helper for TESS light curves.

        Keep this configurable because different science cases benefit from
        different preprocessing choices.
        """
        if lightCurve is None:
            return None

        processed = lightCurve

        try:
            if doRemoveNans:
                processed = processed.remove_nans()

            if doNormalize:
                processed = processed.normalize()

            if doFlatten:
                processed = processed.flatten(window_length=flattenWindowLength)

            if doRemoveOutliers:
                processed = processed.remove_outliers(sigma=sigma)

            if doBin:
                processed = processed.bin(time_bin_size=timeBinSize)

            if doRemoveNans:
                processed = processed.remove_nans()

        except Exception as exc:
            print(f"[WARNING] preprocessLightCurve failed: {exc}")
            return None

        return processed

    def attachLightCurvesToCategoryDictionary(
        self,
        categoryDictionary,
        author="SPOC",
        exptime=120,
        sector=None,
        fluxColumn="pdcsap_flux",
        stitch=True,
        preprocess=True,
        preprocessConfig=None
    ):
        """
        Download and optionally preprocess TESS light curves for each TIC-matched
        star in a family->stars dictionary.
        """
        if preprocessConfig is None:
            preprocessConfig = {
                "doRemoveNans": True,
                "doNormalize": True,
                "doFlatten": False,
                "flattenWindowLength": 401,
                "doRemoveOutliers": False,
                "sigma": 5.0,
                "doBin": False,
                "timeBinSize": 0.01
            }

        result = {}

        for family, stars in categoryDictionary.items():
            enrichedStars = []

            for star in stars:
                ticId = star.get("ticId")
                if ticId is None:
                    continue

                lightCurve = self.downloadTessLightCurve(
                    ticId=ticId,
                    author=author,
                    exptime=exptime,
                    sector=sector,
                    fluxColumn=fluxColumn,
                    stitch=stitch
                )

                if lightCurve is None:
                    continue

                if preprocess:
                    lightCurve = self.preprocessLightCurve(lightCurve, **preprocessConfig)

                if lightCurve is None:
                    continue

                enrichedStar = dict(star)
                enrichedStar["lightCurve"] = lightCurve
                enrichedStars.append(enrichedStar)

            result[family] = enrichedStars

        return result

    # ============================================================
    # Feature helpers
    # ============================================================

    def buildFeatureSummary(self, lightCurve, maxCadences=None):
        """
        Build a simple scalar summary from a Lightkurve object.
        """
        if lightCurve is None:
            return {
                "timeArray": None,
                "fluxArray": None,
                "numCadences": 0,
                "meanFlux": np.nan,
                "stdFlux": np.nan,
                "amplitude": np.nan,
                "medianFlux": np.nan,
                "madFlux": np.nan
            }

        try:
            timeArray = np.asarray(lightCurve.time.value)
            fluxArray = np.asarray(lightCurve.flux.value)

            if maxCadences is not None:
                timeArray = timeArray[:maxCadences]
                fluxArray = fluxArray[:maxCadences]

            medianFlux = float(np.nanmedian(fluxArray))
            madFlux = float(np.nanmedian(np.abs(fluxArray - medianFlux)))

            return {
                "timeArray": timeArray,
                "fluxArray": fluxArray,
                "numCadences": len(fluxArray),
                "meanFlux": float(np.nanmean(fluxArray)),
                "stdFlux": float(np.nanstd(fluxArray)),
                "amplitude": float(np.nanmax(fluxArray) - np.nanmin(fluxArray)),
                "medianFlux": medianFlux,
                "madFlux": madFlux
            }

        except Exception as exc:
            print(f"[WARNING] buildFeatureSummary failed: {exc}")
            return {
                "timeArray": None,
                "fluxArray": None,
                "numCadences": 0,
                "meanFlux": np.nan,
                "stdFlux": np.nan,
                "amplitude": np.nan,
                "medianFlux": np.nan,
                "madFlux": np.nan
            }

    # ============================================================
    # ML dataset builders
    # ============================================================

    def buildMlDataset(
        self,
        categoryDictionary,
        includeLightCurveArrays=True,
        includeMetadata=True,
        maxCadences=None,
        singleLabel=True
    ):
        """
        Convert a family->stars dictionary into a pandas DataFrame suitable
        for downstream ML experiments.
        """
        rows = []

        for family, stars in categoryDictionary.items():
            for star in stars:
                row = {}

                vsxType = star.get("vsxType", "")
                familyList = self._mapVsxTypeToFamilies(vsxType)
                primaryFamily = self._pickPrimaryFamily(familyList) if singleLabel else familyList
                metadata = self.getFamilyMetadata(self._pickPrimaryFamily(familyList))

                if includeMetadata:
                    row["category"] = family
                    row["starId"] = star.get("starId")
                    row["starName"] = star.get("starName")
                    row["vsxType"] = vsxType
                    row["ticId"] = star.get("ticId")
                    row["raDeg"] = star.get("raDeg")
                    row["decDeg"] = star.get("decDeg")
                    row["ticRaDeg"] = star.get("ticRaDeg")
                    row["ticDecDeg"] = star.get("ticDecDeg")
                    row["ticTmag"] = star.get("ticTmag")
                    row["familyList"] = familyList
                    row["primaryFamily"] = primaryFamily
                    row["familyDisplayName"] = metadata["displayName"]
                    row["signalType"] = metadata["signalType"]
                    row["recommendedAlgorithm"] = metadata["recommendedAlgorithm"]
                    row["typicalDifficulty"] = metadata["typicalDifficulty"]

                if includeLightCurveArrays:
                    featureSummary = self.buildFeatureSummary(
                        lightCurve=star.get("lightCurve"),
                        maxCadences=maxCadences
                    )
                    row.update(featureSummary)

                rows.append(row)

        return pd.DataFrame(rows)

    def buildBalancedMlDataset(
        self,
        categoryDictionary,
        starsPerCategory,
        randomSeed=42,
        **buildMlDatasetKwargs
    ):
        """
        Build a balanced ML dataframe by first downsampling each requested
        family to the same number of stars.
        """
        balancedDictionary = self.buildBalancedCategoryDictionary(
            categoryDictionary=categoryDictionary,
            starsPerCategory=starsPerCategory,
            randomSeed=randomSeed
        )

        return self.buildMlDataset(
            categoryDictionary=balancedDictionary,
            **buildMlDatasetKwargs
        )
    

if __name__ == "__main__":
    loader = VSXCategoryLoader()

    categoryRequests = {
        "ECLIPSING": 20,
        "RRLYR": 20,
        "DSCT_SXPHE": 20
    }

    categoryDictionary = loader.loadCategories(
        categoryRequests=categoryRequests,
        useMock=True
    )

    for family, stars in categoryDictionary.items():
        print(family, len(stars))
        if stars:
            print(stars[0])

    # TIC crossmatch
    ticDictionary = loader.crossmatchCategoryDictionaryToTic(categoryDictionary, radiusArcsec=5.0)

    # Download light curves
    lightCurveDictionary = loader.attachLightCurvesToCategoryDictionary(
        ticDictionary,
        author="SPOC",
        exptime=120,
        fluxColumn="pdcsap_flux",
        stitch=True,
        preprocess=True
    )

    # Build balanced ML dataframe
    mlDataFrame = loader.buildBalancedMlDataset(
        lightCurveDictionary,
        starsPerCategory=10,
        includeLightCurveArrays=True,
        includeMetadata=True,
        maxCadences=5000,
        singleLabel=True
    )

    print(mlDataFrame.head())