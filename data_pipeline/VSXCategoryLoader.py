import sys
import os
import re
import math
import random
import time
import logging
import xml.etree.ElementTree as ET
from io import BytesIO
from collections import defaultdict

import requests
import numpy as np
import pandas as pd
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from astropy.io.votable import parse_single_table
import lightkurve as lk

class VSXCategoryLoader:
    def __init__(self, cacheFolder='VSXCache', refreshCache=False):
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
        retrieve a requested number of variable stars for each normalized family
        by querying the AAVSO VSX service directly.
        """
        self.logger = logging.getLogger("VSXCategoryLoader")
        self.cacheFolder = cacheFolder
        if not os.path.exists(self.cacheFolder):
            os.makedirs(self.cacheFolder)
        self.refreshCache = refreshCache

        self.vsxBaseUrl = "https://www.aavso.org/vsx/index.php"
        self.httpTimeoutSec = 600

        # Keep a reusable session for VSX traffic and automatically recover
        # from transient network/server failures.
        self.httpSession = requests.Session()
        retryPolicy = Retry(
            total=3,
            connect=3,
            read=3,
            status=3,
            backoff_factor=0.5,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(["GET"]),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retryPolicy)
        self.httpSession.mount("https://", adapter)
        self.httpSession.mount("http://", adapter)

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
                "mlNotes": "Distinct morphology and strong benchmark class",
            },
            "ELLIPSOIDAL_ROT": {
                "displayName": "Rotational / Ellipsoidal Variables",
                "physicalMechanism": "Rotation, spots, or tidal distortion",
                "signalType": "Periodic or quasi-periodic and often smooth",
                "recommendedAlgorithm": "LombScargle",
                "typicalDifficulty": "medium",
                "tessUseCase": "Good for smooth periodic recovery tests",
                "mlNotes": "Can overlap morphologically with pulsators",
            },
            "RRLYR": {
                "displayName": "RR Lyrae Stars",
                "physicalMechanism": "Radial pulsation of horizontal-branch stars",
                "signalType": "Periodic and often asymmetric",
                "recommendedAlgorithm": "LombScargle",
                "typicalDifficulty": "easy",
                "tessUseCase": "Excellent benchmark for period recovery",
                "mlNotes": "Highly recognizable and astrophysically important",
            },
            "DSCT_SXPHE": {
                "displayName": "Delta Scuti / SX Phoenicis",
                "physicalMechanism": "Short-period stellar pulsation",
                "signalType": "Periodic, high-frequency, sometimes multi-mode",
                "recommendedAlgorithm": "LombScargle",
                "typicalDifficulty": "medium",
                "tessUseCase": "Useful for high-frequency recovery studies",
                "mlNotes": "Often benefits from careful frequency-domain features",
            },
            "CEPHEID": {
                "displayName": "Cepheids",
                "physicalMechanism": "Coherent radial pulsation",
                "signalType": "Periodic and relatively regular",
                "recommendedAlgorithm": "LombScargle",
                "typicalDifficulty": "easy",
                "tessUseCase": "Strong classical pulsator benchmark",
                "mlNotes": "Historically important and morphologically structured",
            },
            "LONG_PERIOD": {
                "displayName": "Long-Period Variables",
                "physicalMechanism": "Pulsation and envelope dynamics in evolved stars",
                "signalType": "Long timescale, semi-regular, or irregular",
                "recommendedAlgorithm": "LombScargle",
                "typicalDifficulty": "hard",
                "tessUseCase": "Useful edge case because TESS baseline may be limited",
                "mlNotes": "Long periods may reduce completeness in short baselines",
            },
            "YSO": {
                "displayName": "Young Stellar Objects",
                "physicalMechanism": "Accretion, occultation, disk effects, magnetic activity",
                "signalType": "Irregular or semi-periodic",
                "recommendedAlgorithm": "ML / custom features",
                "typicalDifficulty": "hard",
                "tessUseCase": "Stress-test class for irregular variability",
                "mlNotes": "Important for non-periodic classification experiments",
            },
            "CV": {
                "displayName": "Cataclysmic Variables",
                "physicalMechanism": "Accretion onto a white dwarf",
                "signalType": "Irregular, eruptive, or hybrid periodic",
                "recommendedAlgorithm": "ML / event detection",
                "typicalDifficulty": "hard",
                "tessUseCase": "Useful for accretion-driven outburst behavior",
                "mlNotes": "Often does not fit simple periodic taxonomy",
            },
            "XRAY": {
                "displayName": "X-ray Binaries / High-Energy Systems",
                "physicalMechanism": "Accretion onto compact objects",
                "signalType": "Complex, noisy, or hybrid modulation",
                "recommendedAlgorithm": "ML / custom analysis",
                "typicalDifficulty": "hard",
                "tessUseCase": "Rare but scientifically valuable special class",
                "mlNotes": "Good rare-class challenge set",
            },
            # "UNKNOWN": {
            #     "displayName": "Unknown / Unmapped",
            #     "physicalMechanism": "Unspecified",
            #     "signalType": "Unspecified",
            #     "recommendedAlgorithm": "Review manually",
            #     "typicalDifficulty": "unknown",
            #     "tessUseCase": "Requires inspection",
            #     "mlNotes": "Potential future expansion target",
            # },
        }

        # ------------------------------------------------------------
        # Reverse mapping: family -> raw VSX types
        # ------------------------------------------------------------
        self.familyToVsxTypes = defaultdict(list)
        for vsxType, family in self.vsxTypeToFamily.items():
            self.familyToVsxTypes[family].append(vsxType)

    def close(self):
        """
        Explicitly close network resources held by this loader.
        """
        session = getattr(self, "httpSession", None)
        if session is not None:
            session.close()
            self.httpSession = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _httpGetVsxContent(self, params):
        """
        Perform one VSX GET request and always close the response object.

        Returns
        -------
        bytes
            Response body content.
        """
        time.sleep(5)
        vsxResponsePath = os.path.join(self.cacheFolder, f"vsx_response_{params['vtype']}.xml")
        if self.refreshCache or not os.path.exists(vsxResponsePath):
            self.logger.info("Connecting to VSX to download fresh data...")
            with self.httpSession.get(
                self.vsxBaseUrl,
                params=params,
                timeout=self.httpTimeoutSec,
            ) as response:
                response.raise_for_status()
                with open(vsxResponsePath, "wb") as f:
                    f.write(response.content)
                self.logger.info("VSX data downloaded and cached to %s", vsxResponsePath)
                return response.content
        else:
            self.logger.info("Loading VSX data from cache: %s", vsxResponsePath)
            with open(vsxResponsePath, "rb") as f:
                return f.read()

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
            "UNKNOWN",
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

    def _loadStarsForFamily(self, family, count):
        """
        Load a requested number of stars for one normalized family directly
        from AAVSO VSX.
        """
        if family not in self.familyToVsxTypes:
            return []

        rawVsxTypes = self.familyToVsxTypes[family]
        if not rawVsxTypes or count <= 0:
            return []

        perTypeTarget = max(1, math.ceil(count / len(rawVsxTypes)))

        combined = []
        for rawVsxType in rawVsxTypes:
            try:
                typeMatches = self._queryVsxByRawType(family, rawVsxType)
            except Exception as exc:
                self.logger.warning("VSX query failed for raw type %s: %s", rawVsxType, exc)
                continue

            if len(typeMatches) > perTypeTarget:
                typeMatches = random.sample(typeMatches, perTypeTarget)

            combined.extend(typeMatches)

            if len(self._deduplicateStars(combined)) >= count:
                break

        combined = self._deduplicateStars(combined)

        if len(combined) > count:
            combined = random.sample(combined, count)

        return combined

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
        knownFamilies = [family for family in families if family is not None]
        return self._pickPrimaryFamily(knownFamilies)

    # ============================================================
    # Family-driven category loading
    # ============================================================

    def loadCategories(self, categoryRequests):
        """
        Load variable stars by requested normalized family directly from VSX.

        Parameters
        ----------
        categoryRequests : dict[str, int]
            Example:
                {
                    "ECLIPSING": 100,
                    "RRLYR": 50
                }

        Returns
        -------
        dict[str, list[dict]]
            Dictionary mapping each requested family to a list of fetched stars.
        """
        result = {}

        for family, count in categoryRequests.items():
            if family not in self.familyToVsxTypes:
                self.logger.warning("Unsupported family requested: %s", family)
                result[family] = []
                continue

            result[family] = self._loadStarsForFamily(family=family, count=count)

        return result
    
    def _votableToRowDicts(self, responseContent):
        """
        Parse VSX VOTable response bytes into a list of row dictionaries.
        """
        # VSX responses can omit strict VOTable typing details for string fields,
        # which can cause Astropy to truncate values to one character. Parse the
        # XML table cells directly first to preserve full text values.
        try:
            root = ET.fromstring(responseContent)
            tableElement = root.find(".//{*}TABLE")
            if tableElement is not None:
                fieldElements = tableElement.findall("{*}FIELD")
                columnNames = [
                    fieldElement.get("name") or fieldElement.get("id") or fieldElement.get("ID")
                    for fieldElement in fieldElements
                    if fieldElement.get("name") or fieldElement.get("id") or fieldElement.get("ID")
                ]

                if columnNames:
                    rowElements = tableElement.findall(".//{*}TR")
                    rowDicts = []
                    for rowElement in rowElements:
                        rowValues = [
                            (cell.text or "").strip()
                            for cell in rowElement.findall("{*}TD")
                        ]
                        rowDicts.append({
                            columnName: (rowValues[index] if index < len(rowValues) else "")
                            for index, columnName in enumerate(columnNames)
                        })

                    if rowDicts:
                        return rowDicts
        except Exception as exc:
            self.logger.debug("XML VOTable parsing path failed, falling back to Astropy: %s", exc)

        # Fallback: keep the Astropy path for compatibility.
        table = parse_single_table(BytesIO(responseContent)).to_table(use_names_over_ids=True)
        columnNames = list(table.colnames)

        return [
            {
                columnName: row[columnName].item() if hasattr(row[columnName], "item") else row[columnName]
                for columnName in columnNames
            }
            for row in table
        ]

    def _queryVsxByRawType(self, family, rawVsxType):
        """
        Query AAVSO VSX for one raw variability type and return normalized records.
        """
        params = {
            "view": "query.votable",
            "vtype": rawVsxType,
        }

        responseContent = self._httpGetVsxContent(params)
        rowDicts = self._votableToRowDicts(responseContent)

        return [
            self._normalizeVsxRow(rowDict, rawVsxType, family)
            for rowDict in rowDicts
        ]
    
    def _deduplicateStars(self, stars):
        """
        Deduplicate combined results across multiple raw VSX types.
        """
        seenKeys = set()
        deduped = []

        for star in stars:
            key = (
                star.get("VSXId"),
                star.get("VSXName"),
                star.get("raDeg"),
                star.get("decDeg"),
            )
            if key in seenKeys:
                continue
            seenKeys.add(key)
            deduped.append(star)

        return deduped
    
    def _pickFirstExistingKey(self, rowDict, candidateKeys):
        lowerKeyMap = {key.lower(): key for key in rowDict.keys()}

        for candidate in candidateKeys:
            actualKey = lowerKeyMap.get(candidate.lower())
            if actualKey is not None:
                return rowDict.get(actualKey)

        return None
    
    def _safeFloat(self, value):
        try:
            if value is None or value == "":
                return None
            return float(value)
        except Exception:
            return None
    
    def _parseCoordsJ2000(self, coordText):
        """
        Parse VSX 'Coords(J2000)' field formatted like:
            '11.44133333,41.84172222'
        into:
            (raDeg, decDeg)
        """
        if isinstance(coordText, bytes):
            coordText = coordText.decode("utf-8", errors="ignore")

        if not isinstance(coordText, str):
            return None, None

        coordText = coordText.strip()
        if not coordText:
            return None, None

        # Primary expected format from VSX is: "ra,dec".
        if "," in coordText:
            parts = [part.strip() for part in coordText.split(",")]
        else:
            # Fallback for whitespace-delimited variants.
            parts = coordText.split()

        if len(parts) != 2:
            return None, None

        try:
            raDeg = float(parts[0])
            decDeg = float(parts[1])
            return raDeg, decDeg
        except ValueError:
            return None, None
    
    def _normalizeVsxRow(self, rowDict, requestedRawVsxType, family):
        """
        Normalize one VSX VOTable row into the internal record shape.
        """
        coordText = self._pickFirstExistingKey(
            rowDict,
            ["Coords(J2000)", "radec2000"]
        )
        raDeg, decDeg = self._parseCoordsJ2000(coordText)

        VSXId = self._pickFirstExistingKey(rowDict, ["AUID", "Name"])
        VSXName = self._pickFirstExistingKey(rowDict, ["Name"])
        vsxType = self._pickFirstExistingKey(rowDict, ["VarType"])
        period = self._safeFloat(self._pickFirstExistingKey(rowDict, ["Period"]))

        return {
            "VSXId": VSXId if VSXId not in [None, ""] else VSXName,
            "VSXName": VSXName,
            "VSXType": vsxType if vsxType not in [None, ""] else requestedRawVsxType,
            "period": period,
            "family": family,
            "raDeg": raDeg,
            "decDeg": decDeg,
            "rawRow": rowDict,
        }
    
if __name__ == "__main__":
    loader = VSXCategoryLoader()

    categoryRequests = {
        "ECLIPSING": 20,
        "RRLYR": 20,
        "DSCT_SXPHE": 20
    }

    categoryDictionary = loader.loadCategories(categoryRequests=categoryRequests)
    loader.close()