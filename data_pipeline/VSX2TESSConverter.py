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

    def crossmatchToTic(self, starRecord, radiusArcsec=5.0):
        """
        Crossmatch AAVSO VSX variable-star record to the TESS Input Catalog using RA/Dec.

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
        raColumnName = "ra" if "ra" in ticTable.colnames else None
        decColumnName = "dec" if "dec" in ticTable.colnames else None

        bestMatch = None
        bestSeparationArcsec = None

        if raColumnName is not None and decColumnName is not None:
            for row in ticTable:
                try:
                    ticCoord = SkyCoord(
                        ra=float(row[raColumnName]) * u.deg,
                        dec=float(row[decColumnName]) * u.deg,
                        frame="icrs",
                    )
                    separationArcsec = coord.separation(ticCoord).arcsec
                except Exception:
                    continue

                if bestSeparationArcsec is None or separationArcsec < bestSeparationArcsec:
                    bestSeparationArcsec = separationArcsec
                    bestMatch = row

        if bestMatch is None:
            bestMatch = ticTable[0]

        matchedRecord = dict(starRecord)
        matchedRecord["ticId"] = bestMatch["ID"]
        matchedRecord["ticRaDeg"] = bestMatch["ra"]
        matchedRecord["ticDecDeg"] = bestMatch["dec"]
        matchedRecord["ticTmag"] = bestMatch["Tmag"]
        matchedRecord["ticDistanceArcmin"] = (
            bestSeparationArcsec / 60.0 if bestSeparationArcsec is not None else None
        )
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