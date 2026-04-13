from astropy.coordinates import SkyCoord
from astropy import units as u
from astroquery.mast import Catalogs
import logging

class VSX2TESSConverter:
    """
    VSX2TESSConverter provides utilities to convert AAVSO VSX variable-star
    records into a format suitable for crossmatching with TESS Input Catalog
    (TIC) entries and downstream TESS analysis pipelines.

    This class is intended to take a VSX record (as returned by VSXCategoryLoader)
    and produce a record enriched with TIC information, including TIC ID, TESS
    magnitude, and angular separation from the VSX coordinates.
    """

    def __init__(self):
        self.logger = logging.getLogger("VSX2TESSConverter")

    def crossmatchToTic(self, starRecord, radiusArcsec=5.0, maxMatches=5):
        """
        Crossmatch AAVSO VSX variable-star record to the TESS Input Catalog using RA/Dec.

        The returned record includes a `ticMatches` list containing up to
        `maxMatches` candidates sorted by angular separation from the VSX target.
        Each candidate has: ticId, ticRaDeg, ticDecDeg, ticTmag, ticDistanceArcmin.

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
                catalog="TIC",
            )
        except Exception as exc:
            self.logger.warning(
                "TIC crossmatch failed for %s: %s",
                starRecord.get("VSXName"),
                exc,
            )
            return None

        if ticTable is None or len(ticTable) == 0:
            return None

        # TIC query results do not always provide a query-center "distance"
        # column, so compute angular separation explicitly from RA/Dec.
        hasRa = "ra" in ticTable.colnames
        hasDec = "dec" in ticTable.colnames

        ticCandidates = []
        if hasRa and hasDec:
            for row in ticTable:
                try:
                    ticRaDeg = float(row["ra"])
                    ticDecDeg = float(row["dec"])
                    ticTmag = float(row["Tmag"])

                    
                    ticCoord = SkyCoord(
                        ra=float(ticRaDeg) * u.deg,
                        dec=float(ticDecDeg) * u.deg,
                        frame="icrs"
                    )
                    separationArcsec = float(coord.separation(ticCoord).arcsec)
                except Exception:
                    continue

                ticCandidates.append(
                    {
                        "ticId": str(row["ID"]),
                        "ticRaDeg": ticRaDeg,
                        "ticDecDeg": ticDecDeg,
                        "ticTmag": ticTmag,
                        "ticDistanceArcmin": separationArcsec / 60.0,
                    }
                )

        if not ticCandidates:
            return None

        maxMatches = max(1, int(maxMatches))
        ticCandidates.sort(key=lambda candidate: candidate["ticDistanceArcmin"])
        topCandidates = ticCandidates[:maxMatches]

        matchedRecord = dict(starRecord)
        matchedRecord["ticMatches"] = topCandidates
        matchedRecord['VSXId'] = starRecord.get('VSXId')
        matchedRecord['VSXName'] = starRecord.get('VSXName')
        matchedRecord['VSXType'] = starRecord.get('VSXType')
        matchedRecord['family'] = starRecord.get('family')
        matchedRecord['period'] = starRecord.get('period')

        return matchedRecord

    def crossmatchCategoryDictionaryToTic(self, categoryDictionary, radiusArcsec=5.0):
        """
        Crossmatch all stars in a family->stars dictionary to TIC.
        """
        result = {}

        for family, stars in categoryDictionary.items():
            self.logger.info("Crossmatching family '%s' with %d stars to TIC", family, len(stars))
            result[family] = [
                matchedRecord
                for star in stars
                if (matchedRecord := self.crossmatchToTic(star, radiusArcsec=radiusArcsec)) is not None
            ]

        return result
    

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    converter = VSX2TESSConverter()
    # Example usage:

    #check RR Lyr
    starRecord = {"raDeg": 291.36629, "decDeg":  42.78436, "VSXName": ""}
    tessRecord = converter.crossmatchToTic(starRecord)
    if tessRecord['ticId'] != '159717514':
        converter.logger.error("RR Lyrae crossmatch failed: expected TIC ID 159717514, got %s", tessRecord['ticId'])
    else:
        converter.logger.info("RR Lyr crossmatch succeeded: TIC ID %s", tessRecord['ticId'])

    #check TV Boo 
    starRecord = {"raDeg": 214.15242, "decDeg": 42.35992, "VSXName": ""}
    tessRecord = converter.crossmatchToTic(starRecord)
    if tessRecord['ticId'] != '168709463':
        logger.error("TV Boo crossmatch failed: expected TIC ID 168709463, got %s", tessRecord['ticId'])
    else:
        logger.info("TV Boo crossmatch succeeded: TIC ID %s", tessRecord['ticId'])

    #check V1334 Cyg
    starRecord = {"raDeg": 319.84242, "decDeg": 38.23747, "VSXName": ""}
    tessRecord = converter.crossmatchToTic(starRecord)
    if tessRecord['ticId'] != '373202340':
        logger.error("V1334 Cyg crossmatch failed: expected TIC ID 373202340, got %s", tessRecord['ticId'])
    else:
        logger.info("V1334 Cyg crossmatch succeeded: TIC ID %s", tessRecord['ticId'])

    #check TT Lyn
    starRecord = {"raDeg": 135.78246, "decDeg":  44.58558, "VSXName": ""}
    tessRecord = converter.crossmatchToTic(starRecord)
    if tessRecord['ticId'] != '29172806':
        logger.error("TT Lyn crossmatch failed: expected TIC ID 29172806, got %s", tessRecord['ticId'])
    else:
        logger.info("TT Lyn crossmatch succeeded: TIC ID %s", tessRecord['ticId'])

    #check AU Peg
    starRecord = {"raDeg": 321.00100, "decDeg":  18.27883, "VSXName": ""}
    tessRecord = converter.crossmatchToTic(starRecord)
    if tessRecord['ticId'] != '279587090':
        logger.error("AU Peg crossmatch failed: expected TIC ID 279587090, got %s", tessRecord['ticId'])
    else:
        logger.info("AU Peg crossmatch succeeded: TIC ID %s", tessRecord['ticId'])