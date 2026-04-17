import pandas as pd
import lightkurve as lk
import logging
import os
import sys
import time
import shutil
import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u
from concurrent.futures import ThreadPoolExecutor, as_completed

class TessDataDownloader:
    """
    A class to download TESS light curve data organized by stellar categories.
    """
    
    def __init__(self, tessCacheFolder='TESSCache'):
        """Initialize the TessDataDownloader."""
        self.logger = logging.getLogger("TessDataDownloader")
        self.tessCacheFolder = tessCacheFolder
        if not os.path.exists(self.tessCacheFolder):
            os.makedirs(self.tessCacheFolder)

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
        timeBinSize=0.01,
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
            self.logger.warning("preprocessLightCurve failed: %s", exc)
            return None

        return processed

    def _resolveMetadataPath(self, tessMetadataParquet):
        if os.path.isabs(tessMetadataParquet):
            return tessMetadataParquet

        cachePath = os.path.join(self.tessCacheFolder, tessMetadataParquet)
        if os.path.exists(cachePath):
            return cachePath

        return tessMetadataParquet

    def _normalizeTicId(self, value):
        if value is None:
            return None

        try:
            if pd.isna(value):
                return None
        except TypeError:
            pass

        digits = "".join(ch for ch in str(value) if ch.isdigit())
        if not digits:
            return None

        return str(int(digits))

    def _sortedTicCandidates(self, ticCandidates):
        if ticCandidates is None:
            return []

        if isinstance(ticCandidates, float) and pd.isna(ticCandidates):
            return []

        if hasattr(ticCandidates, "tolist"):
            ticCandidates = ticCandidates.tolist()

        if isinstance(ticCandidates, dict):
            ticCandidates = [ticCandidates]

        normalizedCandidates = []
        for candidate in ticCandidates or []:
            if candidate is None:
                continue
            normalizedCandidates.append(dict(candidate))

        normalizedCandidates.sort(
            key=lambda candidate: float(candidate.get("ticDistanceArcmin", float("inf")))
        )
        return normalizedCandidates

    def _lightCurveFromSearchResult(self, searchResult):
        if searchResult is None or len(searchResult) == 0:
            return None

        try:
            downloaded = searchResult.download_all(download_dir=self.tessCacheFolder)
        except Exception:
            downloaded = searchResult.download(download_dir=self.tessCacheFolder)

        if downloaded is None:
            return None

        if isinstance(downloaded, lk.LightCurveCollection):
            if len(downloaded) == 0:
                return None
            try:
                return downloaded.stitch()
            except Exception as exc:
                self.logger.warning("Failed to stitch light-curve collection: %s", exc)
                return downloaded[0]

        return downloaded

    def _downloadCatalogLightCurve(self, ticId, author):
        try:
            searchResult = lk.search_lightcurve(
                f"TIC {ticId}",
                mission="TESS",
                author=author,
            )
        except Exception as exc:
            self.logger.warning(
                "search_lightcurve failed for TIC %s author %s: %s",
                ticId,
                author,
                exc,
            )
            return None

        if searchResult is None or len(searchResult) == 0:
            return None

        lightCurve = self._lightCurveFromSearchResult(searchResult)
        if lightCurve is None:
            return None

        outputFile = os.path.join(self.tessCacheFolder, f"TIC_{ticId}_{author}.fits")
        lightCurve.to_fits(path=outputFile, overwrite=True)

        sectors = []
        if hasattr(searchResult, "table") and "sequence_number" in searchResult.table.colnames:
            sectors = [
                int(sector)
                for sector in searchResult.table["sequence_number"]
                if sector is not None and str(sector) != "--"
            ]

        return {
            "bestMatch": {"ticId": ticId, "author": author},
            "provenance": author,
            "lightCurvePath": outputFile,
            "lightCurveAvailable": True,
            "extractionMetadata": {
                "downloadMethod": "search_lightcurve",
                "author": author,
                "productCount": int(len(searchResult)),
                "sectors": sectors,
            },
        }

    def _extractTessCutLightCurve(self, starRecord, ticCandidates, cutoutSize):
        raDeg = starRecord.get("raDeg")
        decDeg = starRecord.get("decDeg")
        if raDeg is None or decDeg is None:
            return None

        try:
            coord = SkyCoord(ra=float(raDeg) * u.deg, dec=float(decDeg) * u.deg, frame="icrs")
        except Exception as exc:
            self.logger.warning(
                "Invalid sky coordinates for %s: %s",
                starRecord.get("VSXName", starRecord.get("VSXId", "unknown")),
                exc,
            )
            return None

        try:
            searchResult = lk.search_tesscut(coord)
        except Exception as exc:
            self.logger.warning(
                "search_tesscut failed for %s: %s",
                starRecord.get("VSXName", starRecord.get("VSXId", "unknown")),
                exc,
            )
            return None

        if searchResult is None or len(searchResult) == 0:
            return None

        try:
            tpfCollection = searchResult.download_all(
                cutout_size=cutoutSize,
                download_dir=self.tessCacheFolder,
            )
        except Exception as exc:
            self.logger.warning(
                "TESSCut download failed for %s: %s",
                starRecord.get("VSXName", starRecord.get("VSXId", "unknown")),
                exc,
            )
            return None

        if tpfCollection is None or len(tpfCollection) == 0:
            return None

        extractedCurves = []
        aperturePixelCounts = []
        sectors = []

        for tpf in tpfCollection:
            try:
                apertureMask = tpf.create_threshold_mask(threshold=3, reference_pixel="center")
                if apertureMask is None or not np.any(apertureMask):
                    apertureMask = np.zeros(tpf.flux[0].shape, dtype=bool)
                    apertureMask[apertureMask.shape[0] // 2, apertureMask.shape[1] // 2] = True

                lightCurve = tpf.to_lightcurve(aperture_mask=apertureMask).remove_nans()
                if len(lightCurve) == 0:
                    continue

                extractedCurves.append(lightCurve)
                aperturePixelCounts.append(int(np.sum(apertureMask)))

                sector = getattr(tpf, "sector", None)
                if sector is not None:
                    sectors.append(int(sector))
            except Exception as exc:
                self.logger.warning(
                    "Aperture photometry failed for %s in one TESSCut sector: %s",
                    starRecord.get("VSXName", starRecord.get("VSXId", "unknown")),
                    exc,
                )

        if not extractedCurves:
            return None

        if len(extractedCurves) == 1:
            stitched = extractedCurves[0]
        else:
            try:
                stitched = lk.LightCurveCollection(extractedCurves).stitch()
            except Exception as exc:
                self.logger.warning("Failed to stitch TESSCut light curves: %s", exc)
                stitched = extractedCurves[0]

        stitched = stitched.remove_nans()
        if len(stitched) == 0:
            return None

        preferredCandidate = ticCandidates[0] if ticCandidates else {}
        chosenTicId = self._normalizeTicId(preferredCandidate.get("ticId")) or "NA"

        outputFile = os.path.join(self.tessCacheFolder, f"TIC_{chosenTicId}_TESSCut.fits")
        stitched.to_fits(path=outputFile, overwrite=True)

        return {
            "bestMatch": {"ticId": chosenTicId, "author": "TESSCut"},
            "provenance": "TESSCut",
            "lightCurvePath": outputFile,
            "lightCurveAvailable": True,
            "extractionMetadata": {
                "downloadMethod": "search_tesscut",
                "sourceRaDeg": float(raDeg),
                "sourceDecDeg": float(decDeg),
                "cutoutSize": list(cutoutSize),
                "sectorCount": len(extractedCurves),
                "sectors": sectors,
                "aperturePixelCounts": aperturePixelCounts,
                "selectedCandidateTicId": chosenTicId,
                "selectedCandidateDistanceArcmin": preferredCandidate.get("ticDistanceArcmin"),
                "extractionMethod": "threshold_mask_photometry",
            },
        }

    def downloadTessLightCurves(
        self,
        tessMetadataParquet,
        augmentedMetadataFile="TESSAugmented.parquet",
        cutoutSize=(15, 15),
    ):
        """
        Download or extract TESS light curves for each star in cached metadata.

        The method first tries TIC candidates in angular-separation order using
        SPOC then QLP products. If neither exists, it falls back to TESSCut using
        the original VSX coordinates and writes augmented metadata to parquet.
        """
        metadataPath = self._resolveMetadataPath(tessMetadataParquet)
        df = pd.read_parquet(metadataPath)

        requiredColumns = {"family", "raDeg", "decDeg", "ticCandidates"}
        missingColumns = requiredColumns - set(df.columns)
        if missingColumns:
            raise ValueError(
                f"Missing required columns in {metadataPath}: {sorted(missingColumns)}"
            )

        for column in [
            "bestMatch",
            "provenance",
            "lightCurvePath",
            "extractionMetadata",
            "lightCurveAvailable",
            "noLightCurveReason",
        ]:
            if column not in df.columns:
                df[column] = [None] * len(df)

        orderedIndices = df.sort_values(["family", "VSXName"], na_position="last").index.tolist()
        currentFamily = None

        for count, idx in enumerate(orderedIndices, start=1):
            row = df.loc[idx]
            family = row.get("family")
            if family != currentFamily:
                currentFamily = family
                self.logger.info("Processing family %s", family)

            starName = row.get("VSXName", row.get("VSXId", f"row-{idx}"))
            self.logger.info("Processing star %d/%d: %s", count, len(orderedIndices), starName)

            df.at[idx, "bestMatch"] = None
            df.at[idx, "provenance"] = None
            df.at[idx, "lightCurvePath"] = None
            df.at[idx, "extractionMetadata"] = None
            df.at[idx, "lightCurveAvailable"] = False
            df.at[idx, "noLightCurveReason"] = None

            ticCandidates = self._sortedTicCandidates(row.get("ticCandidates"))
            matchedResult = None

            for candidate in ticCandidates:
                ticId = self._normalizeTicId(candidate.get("ticId"))
                if ticId is None:
                    continue

                for author in ("SPOC", "QLP"):
                    matchedResult = self._downloadCatalogLightCurve(ticId, author)
                    if matchedResult is None:
                        continue

                    matchedResult["bestMatch"]["ticDistanceArcmin"] = candidate.get("ticDistanceArcmin")
                    matchedResult["bestMatch"]["ticRaDeg"] = candidate.get("ticRaDeg")
                    matchedResult["bestMatch"]["ticDecDeg"] = candidate.get("ticDecDeg")
                    matchedResult["bestMatch"]["ticTmag"] = candidate.get("ticTmag")
                    matchedResult["extractionMetadata"]["selectedCandidate"] = dict(candidate)
                    break

                if matchedResult is not None:
                    break

            if matchedResult is None:
                matchedResult = self._extractTessCutLightCurve(row.to_dict(), ticCandidates, cutoutSize)

            if matchedResult is None:
                df.at[idx, "noLightCurveReason"] = (
                    "No SPOC/QLP light curve and no usable TESSCut extraction"
                )
                continue

            df.at[idx, "bestMatch"] = matchedResult["bestMatch"]
            df.at[idx, "provenance"] = matchedResult["provenance"]
            df.at[idx, "lightCurvePath"] = matchedResult["lightCurvePath"]
            df.at[idx, "extractionMetadata"] = matchedResult["extractionMetadata"]
            df.at[idx, "lightCurveAvailable"] = matchedResult["lightCurveAvailable"]

        augmentedPath = os.path.join(self.tessCacheFolder, augmentedMetadataFile)
        df.to_parquet(augmentedPath, index=False)
        self.logger.info("Saved augmented metadata to %s", augmentedPath)
        return df

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    downloader = TessDataDownloader()
    data = downloader.load('sample_stars.csv')
    print(data)