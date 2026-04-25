import pandas as pd
import lightkurve as lk
import logging
import os
import glob
import re
import sys
import time
import shutil
import io
import numpy as np
import threading
import warnings
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.io import fits
from contextlib import redirect_stderr, redirect_stdout
from concurrent.futures import ThreadPoolExecutor, as_completed


class _QualityMaskLogFilter(logging.Filter):
    """
    Drops quality-mask log records emitted by Lightkurve below a given
    ignored-cadence percentage threshold, and updates downloader quality.
    """
    _PATTERN = re.compile(
        r"([0-9]+(?:\.[0-9]+)?)%.*cadences will be ignored due to the quality mask",
        re.IGNORECASE,
    )

    def __init__(self, thresholdPercent=20.0, downloader=None):
        super().__init__()
        self.thresholdPercent = thresholdPercent
        self.downloader = downloader

    def filter(self, record):
        message = record.getMessage()
        match = self._PATTERN.search(message)
        if match is not None:
            percentage = float(match.group(1))
            if self.downloader is not None:
                self.downloader._updateStarQuality(percentage)
            return percentage > self.thresholdPercent
        return True


_qualityMaskLogFilter = _QualityMaskLogFilter(thresholdPercent=20.0, downloader=None)
for _lkLoggerName in (
    "lightkurve",
    "lightkurve.io",
    "lightkurve.lightcurve",
    "lightkurve.utils",
    "lightkurve.search",
    "lightkurve.targetpixelfile",
):
    logging.getLogger(_lkLoggerName).addFilter(_qualityMaskLogFilter)

class TessDataDownloader:
    """
    A class to download TESS light curve data organized by stellar categories.
    """
    
    def __init__(self, tessCacheFolder='TESSCache'):
        """Initialize the TessDataDownloader."""
        self.logger = logging.getLogger("TessDataDownloader")
        self.tessCacheFolder = tessCacheFolder
        self.normalizationStdFloor = 1e-8
        self.lowSnrThreshold = 3.0
        if not os.path.exists(self.tessCacheFolder):
            os.makedirs(self.tessCacheFolder)

        self.ignoredCadenceWarningThresholdPercent = 20.0
        self._threadState = threading.local()
        self._workerDownloadsRoot = os.path.join(self.tessCacheFolder, "_worker_downloads")
        
        global _qualityMaskLogFilter
        _qualityMaskLogFilter.downloader = self

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
        normalizationMetadata = None

        try:
            if doRemoveNans:
                processed = processed.remove_nans()

            if doNormalize:
                processed, normalizationMetadata = self._standardizeLightCurve(processed)
                if processed is None:
                    return None

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

    def _cleanupTransientDownloads(self, downloadRoot=None):
        rootPath = downloadRoot or self.tessCacheFolder
        transientPaths = [
            os.path.join(rootPath, "mastDownload"),
            os.path.join(rootPath, "tesscut"),
        ]

        for transientPath in transientPaths:
            if not os.path.exists(transientPath):
                continue

            try:
                shutil.rmtree(transientPath)
            except Exception as exc:
                self.logger.warning(
                    "Failed to remove transient download directory %s: %s",
                    transientPath,
                    exc,
                )

    def _resetStarQuality(self):
        self._threadState.currentStarQuality = "missing"

    def _getCurrentStarQuality(self):
        return getattr(self._threadState, "currentStarQuality", "missing")

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

    def _sanitizeFilenameComponent(self, value):
        """Sanitize a string for use in filenames by replacing spaces and special chars with underscores."""
        if value is None:
            return None
        # Replace spaces and other problematic characters with underscores
        sanitized = str(value).replace(" ", "_").replace(",", "_").replace(":", "_")
        return sanitized

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

    def _logDuplicateTicMappings(self, df):
        """Log stars that map to the same top TIC candidate."""
        starsByTopTic = {}
        duplicateWarningCount = 0

        for _, row in df.iterrows():
            ticCandidates = self._sortedTicCandidates(row.get("ticCandidates"))
            topTic = self._normalizeTicId((ticCandidates[0] if ticCandidates else {}).get("ticId"))
            if topTic is None:
                continue

            candidateTics = [
                self._normalizeTicId(candidate.get("ticId"))
                for candidate in ticCandidates
            ]
            candidateTics = [tic for tic in candidateTics if tic is not None]

            starsByTopTic.setdefault(topTic, []).append(
                {
                    "family": row.get("family"),
                    "VSXType": row.get("VSXType"),
                    "VSXId": row.get("VSXId"),
                    "tics": candidateTics,
                }
            )

        for topTic, stars in starsByTopTic.items():
            if len(stars) < 2:
                continue
            for star in stars:
                self.logger.error(
                    "Duplicate TIC mapping detected: TIC=%s family=%s vsxtype=%s vsxid=%s tics=%s",
                    topTic,
                    star.get("family"),
                    star.get("VSXType"),
                    star.get("VSXId"),
                    star.get("tics"),
                )
                duplicateWarningCount += 1

        return duplicateWarningCount

    def _standardizeLightCurve(self, lightCurve):
        if lightCurve is None:
            return None, None

        cleanedLightCurve = lightCurve.remove_nans()
        if len(cleanedLightCurve) == 0:
            return None, {
                "normalizationApplied": False,
                "fluxMedian": None,
                "fluxStd": None,
                "fluxSnr": None,
                "medianUnstable": True,
                "lowSNR": True,
                "lowQualityLightCurve": True,
                "lowQualityReason": "No valid cadences after removing NaNs",
                "validCadenceCount": 0,
            }

        fluxArray = np.asarray(getattr(cleanedLightCurve.flux, "value", cleanedLightCurve.flux), dtype=float)
        validMask = np.isfinite(fluxArray)

        fluxMask = getattr(cleanedLightCurve.flux, "mask", None)
        if fluxMask is not None:
            fluxMaskArray = np.asarray(fluxMask, dtype=bool)
            if fluxMaskArray.shape == ():
                validMask &= (not bool(fluxMaskArray))
            else:
                validMask &= np.logical_not(fluxMaskArray)

        if hasattr(cleanedLightCurve.time, "value"):
            timeValues = np.asarray(cleanedLightCurve.time.value, dtype=float)
            validMask &= np.isfinite(timeValues)

        validFlux = fluxArray[validMask]
        if validFlux.size == 0:
            return None, {
                "normalizationApplied": False,
                "fluxMedian": None,
                "fluxStd": None,
                "fluxSnr": None,
                "medianUnstable": True,
                "lowSNR": True,
                "lowQualityLightCurve": True,
                "lowQualityReason": "No valid cadences after masking invalid flux values",
                "validCadenceCount": 0,
            }

        medianFlux = float(np.nanmedian(validFlux))
        stdFlux = float(np.nanstd(validFlux))
        medianUnstable = (not np.isfinite(medianFlux)) or abs(medianFlux) < self.normalizationStdFloor
        lowStd = (not np.isfinite(stdFlux)) or stdFlux < self.normalizationStdFloor
        fluxSnr = None if lowStd else float(abs(medianFlux) / stdFlux)
        lowSNR = fluxSnr is None or fluxSnr < self.lowSnrThreshold

        normalizationMetadata = {
            "normalizationApplied": False,
            "fluxMedian": medianFlux,
            "fluxStd": stdFlux,
            "fluxSnr": fluxSnr,
            "medianUnstable": bool(medianUnstable),
            "lowSNR": bool(lowSNR),
            "lowQualityLightCurve": bool(medianUnstable or lowSNR or lowStd),
            "lowQualityReason": None,
            "validCadenceCount": int(validFlux.size),
        }

        if lowStd:
            normalizationMetadata["lowQualityReason"] = "Flux standard deviation is too small for stable normalization"
            return cleanedLightCurve, normalizationMetadata

        standardizedFlux = np.full_like(fluxArray, np.nan, dtype=float)
        standardizedFlux[validMask] = (fluxArray[validMask] - medianFlux) / stdFlux

        standardizedLightCurve = cleanedLightCurve.copy()
        standardizedLightCurve.flux = standardizedFlux * u.dimensionless_unscaled

        if hasattr(standardizedLightCurve, "flux_err") and standardizedLightCurve.flux_err is not None:
            fluxErrArray = np.asarray(
                getattr(standardizedLightCurve.flux_err, "value", standardizedLightCurve.flux_err),
                dtype=float,
            )
            standardizedFluxErr = np.full_like(fluxErrArray, np.nan, dtype=float)
            finiteFluxErr = np.isfinite(fluxErrArray)
            standardizedFluxErr[finiteFluxErr] = fluxErrArray[finiteFluxErr] / stdFlux
            standardizedLightCurve.flux_err = standardizedFluxErr * u.dimensionless_unscaled

        normalizationMetadata["normalizationApplied"] = True
        return standardizedLightCurve.remove_nans(), normalizationMetadata

    def _storeLightCurve(self, lightCurve, outputFile, ticId, authorLabel):
        try:
            lightCurve.to_fits(path=outputFile, overwrite=True)
        except (AttributeError, ValueError) as exc:
            self.logger.warning(
                "to_fits failed for TIC %s author %s, using fallback: %s",
                ticId,
                authorLabel,
                exc,
            )
            try:
                timeCol = fits.Column(name="TIME", format="D", array=lightCurve.time.jd)
                fluxCol = fits.Column(name="FLUX", format="E", array=np.asarray(lightCurve.flux.value, dtype=float))
                cols = fits.ColDefs([timeCol, fluxCol])
                hdu = fits.BinTableHDU.from_columns(cols)
                hdu.writeto(outputFile, overwrite=True)
            except Exception as fallbackExc:
                self.logger.error(
                    "Fallback FITS write also failed for TIC %s author %s: %s",
                    ticId,
                    authorLabel,
                    fallbackExc,
                )
                return False

        return True

    def _categorizeQualityMaskPercentage(self, percentage):
        """Categorize data quality based on quality_bitmask cadence percentage."""
        if percentage < 10:
            return "clean"
        elif percentage < 25:
            return "acceptable"
        elif percentage < 40:
            return "caution"
        else:
            return "poor"

    def _updateStarQuality(self, percentage):
        """Update current star quality to worst quality seen so far."""
        newQuality = self._categorizeQualityMaskPercentage(percentage)
        qualityOrder = {"clean": 0, "acceptable": 1, "caution": 2, "poor": 3}
        currentQuality = self._getCurrentStarQuality()
        if qualityOrder.get(newQuality, 0) > qualityOrder.get(currentQuality, 0):
            self._threadState.currentStarQuality = newQuality

    def _finalizeStarQuality(self, lightCurveAvailable):
        """Return final quality after considering whether any light curve was available."""
        if not lightCurveAvailable:
            return "missing"

        if self._getCurrentStarQuality() == "missing":
            return "clean"

        return self._getCurrentStarQuality()

    def _runWithFilteredWarnings(self, func, *args, ticId=None, warningContext=None, **kwargs):
        qualityWarningPattern = re.compile(
            r"([0-9]+(?:\.[0-9]+)?)%\s*\([^)]*\)\s*of the cadences will be ignored due to the quality mask",
            re.IGNORECASE,
        )
        negativeMedianPattern = re.compile(
            r"negative median flux",
            re.IGNORECASE,
        )
        zeroCenteredPattern = re.compile(
            r"zero-centered.*normalize\(\)",
            re.IGNORECASE,
        )
        boolInversionDeprecationPattern = re.compile(
            r"bitwise inversion\s+['`~]+\s*on bool\s+is deprecated",
            re.IGNORECASE,
        )

        with warnings.catch_warnings(record=True) as caughtWarnings:
            warnings.simplefilter("always")
            result = func(*args, **kwargs)

        for caughtWarning in caughtWarnings:
            warningMessage = str(caughtWarning.message)
            qualityMatch = qualityWarningPattern.search(warningMessage)

            if qualityMatch is not None:
                ignoredPercent = float(qualityMatch.group(1))
                self._updateStarQuality(ignoredPercent)
                if ignoredPercent > self.ignoredCadenceWarningThresholdPercent:
                    self.logger.warning(
                        "High quality-mask rejection for TIC %s%s: %s",
                        ticId if ticId is not None else "unknown",
                        f" during {warningContext}" if warningContext else "",
                        warningMessage,
                    )
                continue

            if negativeMedianPattern.search(warningMessage) is not None:
                continue

            if zeroCenteredPattern.search(warningMessage) is not None:
                continue

            if boolInversionDeprecationPattern.search(warningMessage) is not None:
                continue

            self.logger.warning(
                "Warning for TIC %s%s: %s",
                ticId if ticId is not None else "unknown",
                f" during {warningContext}" if warningContext else "",
                warningMessage,
            )

        return result

    def _lightCurveFromSearchResult(self, searchResult, ticId=None, downloadDir=None):
        if searchResult is None or len(searchResult) == 0:
            return None

        resolvedDownloadDir = downloadDir or self.tessCacheFolder

        try:
            downloaded = self._runWithFilteredWarnings(
                searchResult.download_all,
                download_dir=resolvedDownloadDir,
                ticId=ticId,
                warningContext="search_lightcurve download_all",
            )
        except Exception:
            downloaded = self._runWithFilteredWarnings(
                searchResult.download,
                download_dir=resolvedDownloadDir,
                ticId=ticId,
                warningContext="search_lightcurve download",
            )

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

    def _runQuietLightkurveSearch(self, searchFn, *args, **kwargs):
        loggerNames = [
            "lightkurve.search",
            "astroquery",
            "astroquery.mast",
        ]
        loggerState = []

        for loggerName in loggerNames:
            packageLogger = logging.getLogger(loggerName)
            loggerState.append((packageLogger, packageLogger.level, packageLogger.disabled))
            packageLogger.setLevel(logging.CRITICAL + 1)

        try:
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                return searchFn(*args, **kwargs)
        finally:
            for packageLogger, level, disabled in loggerState:
                packageLogger.setLevel(level)
                packageLogger.disabled = disabled

    def _downloadCatalogLightCurve(self, ticId, author, downloadDir=None, vsxId=None):
        try:
            searchResult = self._runQuietLightkurveSearch(
                lk.search_lightcurve,
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

        lightCurve = self._lightCurveFromSearchResult(
            searchResult,
            ticId=ticId,
            downloadDir=downloadDir,
        )
        if lightCurve is None:
            return None

        # Save raw light curve
        sanitizedVsxId = self._sanitizeFilenameComponent(vsxId)
        vsxIdStr = f"VSX_{sanitizedVsxId}_" if sanitizedVsxId else ""
        rawOutputFile = os.path.join(self.tessCacheFolder, f"{vsxIdStr}TIC_{ticId}_{author}_raw.fits")
        if not self._storeLightCurve(lightCurve, rawOutputFile, ticId, f"{author}_raw"):
            return None

        # Standardize light curve
        standardizedLightCurve, normalizationMetadata = self._standardizeLightCurve(lightCurve)
        if standardizedLightCurve is None:
            return None

        # Save standardized light curve
        standardizedOutputFile = os.path.join(self.tessCacheFolder, f"{vsxIdStr}TIC_{ticId}_{author}_standardized.fits")
        if not self._storeLightCurve(standardizedLightCurve, standardizedOutputFile, ticId, f"{author}_standardized"):
            return None

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
            "lightCurvePath": standardizedOutputFile,  # Standardized by default
            "rawLightCurvePath": rawOutputFile,
            "lightCurveAvailable": True,
            "extractionMetadata": {
                "downloadMethod": "search_lightcurve",
                "author": author,
                "productCount": int(len(searchResult)),
                "sectors": sectors,
                "fluxNormalization": normalizationMetadata,
            },
            "normalizationApplied": normalizationMetadata["normalizationApplied"],
            "fluxMedian": normalizationMetadata["fluxMedian"],
            "fluxStd": normalizationMetadata["fluxStd"],
            "fluxSnr": normalizationMetadata["fluxSnr"],
            "medianUnstable": normalizationMetadata["medianUnstable"],
            "lowSNR": normalizationMetadata["lowSNR"],
            "lowQualityLightCurve": normalizationMetadata["lowQualityLightCurve"],
            "lowQualityReason": normalizationMetadata["lowQualityReason"],
        }

    def _extractTessCutLightCurve(self, starRecord, ticCandidates, cutoutSize, downloadDir=None):
        selectedCandidate = {}
        selectedCandidateRank = None
        sourceRaDeg = None
        sourceDecDeg = None

        sortedCandidates = self._sortedTicCandidates(ticCandidates)

        # Use the first TIC candidate that has both coordinates populated.
        for rank, candidate in enumerate(sortedCandidates, start=1):
            candidateRaDeg = candidate.get("ticRaDeg")
            candidateDecDeg = candidate.get("ticDecDeg")
            if candidateRaDeg is None or candidateDecDeg is None:
                continue
            selectedCandidate = candidate
            selectedCandidateRank = rank
            sourceRaDeg = candidateRaDeg
            sourceDecDeg = candidateDecDeg
            break

        # Fall back to original VSX coordinates only when no TIC candidate has coordinates.
        if sourceRaDeg is None or sourceDecDeg is None:
            sourceRaDeg = starRecord.get("raDeg")
            sourceDecDeg = starRecord.get("decDeg")

        # Preserve prior behavior for TIC-based metadata if no candidate had usable coordinates.
        if not selectedCandidate:
            selectedCandidate = (sortedCandidates[0] if sortedCandidates else {})

        if selectedCandidateRank is not None:
            self.logger.info(
                "TESSCut fallback will use TIC candidate rank %d TIC=%s distance=%s arcmin for VSX=%s",
                selectedCandidateRank,
                selectedCandidate.get("ticId"),
                selectedCandidate.get("ticDistanceArcmin"),
                starRecord.get("VSXId"),
            )
        else:
            self.logger.info(
                "TESSCut fallback has no TIC candidate with valid coordinates; using original VSX coords for VSX=%s",
                starRecord.get("VSXId"),
            )

        if sourceRaDeg is None or sourceDecDeg is None:
            return None

        resolvedDownloadDir = downloadDir or self.tessCacheFolder

        try:
            coord = SkyCoord(ra=float(sourceRaDeg) * u.deg, dec=float(sourceDecDeg) * u.deg, frame="icrs")
        except Exception as exc:
            self.logger.warning(
                "Invalid sky coordinates for %s: %s",
                starRecord.get("VSXName", starRecord.get("VSXId", "unknown")),
                exc,
            )
            return None

        try:
            searchResult = self._runQuietLightkurveSearch(lk.search_tesscut, coord)
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
            tpfCollection = self._runWithFilteredWarnings(
                searchResult.download_all,
                cutout_size=cutoutSize,
                download_dir=resolvedDownloadDir,
                ticId=self._normalizeTicId(selectedCandidate.get("ticId")),
                warningContext="search_tesscut download_all",
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

                lightCurve = self._runWithFilteredWarnings(
                    tpf.to_lightcurve,
                    aperture_mask=apertureMask,
                    ticId=self._normalizeTicId(selectedCandidate.get("ticId")),
                    warningContext="TESSCut to_lightcurve",
                ).remove_nans()
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
            finally:
                try:
                    if hasattr(tpf, "close") and callable(tpf.close):
                        tpf.close()
                    elif hasattr(tpf, "hdu") and hasattr(tpf.hdu, "close"):
                        tpf.hdu.close()
                except Exception as closeExc:
                    self.logger.debug("Failed to close TESSCut handle cleanly: %s", closeExc)

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

        # Save raw light curve
        chosenTicId = self._normalizeTicId(selectedCandidate.get("ticId")) or "NA"
        vsxId = starRecord.get("VSXId", "NA")
        sanitizedVsxId = self._sanitizeFilenameComponent(vsxId)
        rawOutputFile = os.path.join(self.tessCacheFolder, f"VSX_{sanitizedVsxId}_TIC_{chosenTicId}_TESSCut_raw.fits")
        if not self._storeLightCurve(stitched, rawOutputFile, chosenTicId, "TESSCut_raw"):
            return None

        # Standardize light curve
        standardizedLightCurve, normalizationMetadata = self._standardizeLightCurve(stitched)
        if standardizedLightCurve is None:
            return None

        # Save standardized light curve
        standardizedOutputFile = os.path.join(self.tessCacheFolder, f"VSX_{sanitizedVsxId}_TIC_{chosenTicId}_TESSCut_standardized.fits")
        if not self._storeLightCurve(standardizedLightCurve, standardizedOutputFile, chosenTicId, "TESSCut_standardized"):
            return None

        return {
            "bestMatch": {"ticId": chosenTicId, "author": "TESSCut"},
            "provenance": "TESSCut",
            "lightCurvePath": standardizedOutputFile,  # Standardized by default
            "rawLightCurvePath": rawOutputFile,
            "lightCurveAvailable": True,
            "extractionMetadata": {
                "downloadMethod": "search_tesscut",
                "sourceRaDeg": float(sourceRaDeg),
                "sourceDecDeg": float(sourceDecDeg),
                "cutoutSize": list(cutoutSize),
                "sectorCount": len(extractedCurves),
                "sectors": sectors,
                "aperturePixelCounts": aperturePixelCounts,
                "selectedCandidateTicId": chosenTicId,
                "selectedCandidateDistanceArcmin": selectedCandidate.get("ticDistanceArcmin"),
                "extractionMethod": "threshold_mask_photometry",
                "fluxNormalization": normalizationMetadata,
            },
            "normalizationApplied": normalizationMetadata["normalizationApplied"],
            "fluxMedian": normalizationMetadata["fluxMedian"],
            "fluxStd": normalizationMetadata["fluxStd"],
            "fluxSnr": normalizationMetadata["fluxSnr"],
            "medianUnstable": normalizationMetadata["medianUnstable"],
            "lowSNR": normalizationMetadata["lowSNR"],
            "lowQualityLightCurve": normalizationMetadata["lowQualityLightCurve"],
            "lowQualityReason": normalizationMetadata["lowQualityReason"],
        }

    def _processSingleStar(self, idx, rowData, count, totalCount, cutoutSize, cleanupDownloads):
        starName = rowData.get("VSXName", rowData.get("VSXId", f"row-{idx}"))
        family = rowData.get("family")
        self.logger.info(
            "Processing star %d/%d: %s (family=%s)",
            count,
            totalCount,
            starName,
            family,
        )

        self._resetStarQuality()
        os.makedirs(self._workerDownloadsRoot, exist_ok=True)
        taskDownloadDir = os.path.join(
            self._workerDownloadsRoot,
            f"star_{idx}_{threading.get_ident()}_{int(time.time() * 1000)}",
        )
        os.makedirs(taskDownloadDir, exist_ok=True)

        updates = {
            "bestMatch": None,
            "provenance": None,
            "lightCurvePath": None,
            "rawLightCurvePath": None,
            "extractionMetadata": None,
            "lightCurveAvailable": False,
            "noLightCurveReason": None,
            "normalizationApplied": None,
            "fluxMedian": None,
            "fluxStd": None,
            "fluxSnr": None,
            "medianUnstable": None,
            "lowSNR": None,
            "lowQualityLightCurve": None,
            "lowQualityReason": None,
            "quality": "missing",
        }

        # Check if files already exist
        vsxId = rowData.get("VSXId")
        if vsxId:
            sanitized = self._sanitizeFilenameComponent(vsxId)
            std_pattern = os.path.join(self.tessCacheFolder, f"VSX_{sanitized}_*standardized.fits")
            std_files = glob.glob(std_pattern)
            if std_files:
                std_path = std_files[0]
                raw_path = std_path.replace('_standardized.fits', '_raw.fits')
                if os.path.exists(raw_path):
                    updates["lightCurvePath"] = std_path
                    updates["rawLightCurvePath"] = raw_path
                    updates["lightCurveAvailable"] = True
                    updates["provenance"] = "existing"
                    updates["quality"] = "existing"
                    self.logger.info("Skipping %s: files already exist", starName)
                    return idx, updates

        ticCandidates = self._sortedTicCandidates(rowData.get("ticCandidates"))
        preferredTicId = self._normalizeTicId(
            (ticCandidates[0] if ticCandidates else {}).get("ticId")
        )
        matchedResult = None

        try:
            def _starTask():
                nonlocal matchedResult

                for candidate in ticCandidates:
                    ticId = self._normalizeTicId(candidate.get("ticId"))
                    if ticId is None:
                        continue

                    for author in ("SPOC", "QLP"):
                        matchedResult = self._downloadCatalogLightCurve(
                            ticId,
                            author,
                            downloadDir=taskDownloadDir,
                            vsxId=rowData.get("VSXId"),
                        )
                        if matchedResult is None:
                            continue

                        matchedResult["bestMatch"]["ticDistanceArcmin"] = candidate.get("ticDistanceArcmin")
                        matchedResult["bestMatch"]["ticRaDeg"] = candidate.get("ticRaDeg")
                        matchedResult["bestMatch"]["ticDecDeg"] = candidate.get("ticDecDeg")
                        matchedResult["bestMatch"]["ticTmag"] = candidate.get("ticTmag")
                        matchedResult["extractionMetadata"]["selectedCandidate"] = dict(candidate)
                        return

                matchedResult = self._extractTessCutLightCurve(
                    rowData,
                    ticCandidates,
                    cutoutSize,
                    downloadDir=taskDownloadDir,
                )

            self._runWithFilteredWarnings(
                _starTask,
                ticId=preferredTicId,
                warningContext="star processing",
            )

            if matchedResult is None:
                self.logger.error(
                    "No usable TESS light curve found for %s after SPOC, QLP, and TESSCut attempts",
                    starName,
                )
                updates["noLightCurveReason"] = (
                    "No SPOC/QLP light curve and no usable TESSCut extraction"
                )
                updates["quality"] = self._finalizeStarQuality(False)
                return idx, updates

            updates.update(
                {
                    "bestMatch": matchedResult["bestMatch"],
                    "provenance": matchedResult["provenance"],
                    "lightCurvePath": matchedResult["lightCurvePath"],
                    "rawLightCurvePath": matchedResult.get("rawLightCurvePath"),
                    "extractionMetadata": matchedResult["extractionMetadata"],
                    "lightCurveAvailable": matchedResult["lightCurveAvailable"],
                    "normalizationApplied": matchedResult.get("normalizationApplied"),
                    "fluxMedian": matchedResult.get("fluxMedian"),
                    "fluxStd": matchedResult.get("fluxStd"),
                    "fluxSnr": matchedResult.get("fluxSnr"),
                    "medianUnstable": matchedResult.get("medianUnstable"),
                    "lowSNR": matchedResult.get("lowSNR"),
                    "lowQualityLightCurve": matchedResult.get("lowQualityLightCurve"),
                    "lowQualityReason": matchedResult.get("lowQualityReason"),
                    "quality": self._finalizeStarQuality(
                        matchedResult.get("lightCurveAvailable", False)
                    ),
                }
            )
            return idx, updates
        finally:
            if cleanupDownloads:
                self._cleanupTransientDownloads(taskDownloadDir)

            try:
                shutil.rmtree(taskDownloadDir)
            except FileNotFoundError:
                pass
            except Exception as exc:
                self.logger.warning(
                    "Failed to remove worker download directory %s: %s",
                    taskDownloadDir,
                    exc,
                )

    def downloadTessLightCurves(
        self,
        tessMetadataParquet,
        augmentedMetadataFile="TESSAugmented.parquet",
        cutoutSize=(15, 15),
        cleanupDownloads=True,
        lowSnrThreshold=None,
        maxWorkers=None,
    ):
        """
        Download or extract TESS light curves for each star in cached metadata.

        The method first tries TIC candidates in angular-separation order using
        SPOC then QLP products. If neither exists, it falls back to TESSCut using
        the original VSX coordinates and writes augmented metadata to parquet.
        """
        originalLowSnrThreshold = self.lowSnrThreshold
        if lowSnrThreshold is not None:
            self.lowSnrThreshold = float(lowSnrThreshold)

        metadataPath = self._resolveMetadataPath(tessMetadataParquet)
        df = pd.read_parquet(metadataPath)

        requiredColumns = {"family", "raDeg", "decDeg", "ticCandidates"}
        missingColumns = requiredColumns - set(df.columns)
        if missingColumns:
            raise ValueError(
                f"Missing required columns in {metadataPath}: {sorted(missingColumns)}"
            )

        # Check if augmented metadata exists to resume from previous run
        augmentedPath = os.path.join(self.tessCacheFolder, augmentedMetadataFile)
        if os.path.exists(augmentedPath):
            self.logger.info("Loading existing augmented metadata from %s to resume processing", augmentedPath)
            df = pd.read_parquet(augmentedPath)
        else:
            # Initialize columns if starting fresh
            for column in [
                "bestMatch",
                "provenance",
                "lightCurvePath",
                "rawLightCurvePath",
                "extractionMetadata",
                "lightCurveAvailable",
                "noLightCurveReason",
                "normalizationApplied",
                "fluxMedian",
                "fluxStd",
                "fluxSnr",
                "medianUnstable",
                "lowSNR",
                "lowQualityLightCurve",
                "lowQualityReason",
                "quality",
            ]:
                if column not in df.columns:
                    df[column] = [None] * len(df)

        duplicateWarningCount = self._logDuplicateTicMappings(df)

        # Only process stars that don't have both raw and standardized files
        orderedIndices = df.sort_values(["family", "VSXName"], na_position="last").index.tolist()
        def has_both_files(row):
            lc_path = row.get("lightCurvePath")
            raw_path = row.get("rawLightCurvePath")
            return lc_path and raw_path and os.path.exists(lc_path) and os.path.exists(raw_path)
        orderedIndices = [idx for idx in orderedIndices if not has_both_files(df.loc[idx])]

        if not orderedIndices:
            self.logger.info("All stars already have light curves available. Skipping download.")
            successCount = int(df["lightCurveAvailable"].eq(True).sum())
            totalCount = int(len(df))
            failedCount = totalCount - successCount
            self.logger.info(
                "Download summary: total=%d succeeded=%d failed=%d duplicateTicWarnings=%d",
                totalCount,
                successCount,
                failedCount,
                duplicateWarningCount,
            )
            return df

        workerCount = maxWorkers
        if workerCount is None:
            workerCount = min(4, max(1, len(orderedIndices)))

        try:
            if workerCount == 1:
                for count, idx in enumerate(orderedIndices, start=1):
                    try:
                        _, updates = self._processSingleStar(
                            idx,
                            df.loc[idx].to_dict(),
                            count,
                            len(orderedIndices),
                            cutoutSize,
                            cleanupDownloads,
                        )
                    except Exception as exc:
                        row = df.loc[idx]
                        starName = row.get("VSXName", row.get("VSXId", f"row-{idx}"))
                        self.logger.exception("Star task failed for %s: %s", starName, exc)
                        updates = {
                            "bestMatch": None,
                            "provenance": None,
                            "lightCurvePath": None,
                            "rawLightCurvePath": None,
                            "extractionMetadata": None,
                            "lightCurveAvailable": False,
                            "noLightCurveReason": f"Worker task failed: {exc}",
                            "normalizationApplied": None,
                            "fluxMedian": None,
                            "fluxStd": None,
                            "fluxSnr": None,
                            "medianUnstable": None,
                            "lowSNR": None,
                            "lowQualityLightCurve": None,
                            "lowQualityReason": None,
                            "quality": "missing",
                        }
                    for column, value in updates.items():
                        df.at[idx, column] = value
                    # Save progress after each star
                    df.to_parquet(augmentedPath, index=False)
            else:
                with ThreadPoolExecutor(max_workers=workerCount) as executor:
                    futureToIdx = {
                        executor.submit(
                            self._processSingleStar,
                            idx,
                            df.loc[idx].to_dict(),
                            count,
                            len(orderedIndices),
                            cutoutSize,
                            cleanupDownloads,
                        ): idx
                        for count, idx in enumerate(orderedIndices, start=1)
                    }

                    for future in as_completed(futureToIdx):
                        idx = futureToIdx[future]
                        try:
                            _, updates = future.result()
                        except Exception as exc:
                            row = df.loc[idx]
                            starName = row.get("VSXName", row.get("VSXId", f"row-{idx}"))
                            self.logger.exception("Star task failed for %s: %s", starName, exc)
                            updates = {
                                "bestMatch": None,
                                "provenance": None,
                                "lightCurvePath": None,
                                "rawLightCurvePath": None,
                                "extractionMetadata": None,
                                "lightCurveAvailable": False,
                                "noLightCurveReason": f"Worker task failed: {exc}",
                                "normalizationApplied": None,
                                "fluxMedian": None,
                                "fluxStd": None,
                                "fluxSnr": None,
                                "medianUnstable": None,
                                "lowSNR": None,
                                "lowQualityLightCurve": None,
                                "lowQualityReason": None,
                                "quality": "missing",
                            }

                        for column, value in updates.items():
                            df.at[idx, column] = value
                        # Save progress after each star
                        df.to_parquet(augmentedPath, index=False)
        finally:
            self.lowSnrThreshold = originalLowSnrThreshold
            if cleanupDownloads:
                self._cleanupTransientDownloads()
            try:
                shutil.rmtree(self._workerDownloadsRoot)
            except FileNotFoundError:
                pass
            except Exception as exc:
                self.logger.warning(
                    "Failed to remove worker download root %s: %s",
                    self._workerDownloadsRoot,
                    exc,
                )

        augmentedPath = os.path.join(self.tessCacheFolder, augmentedMetadataFile)
        df.to_parquet(augmentedPath, index=False)
        successCount = int(df["lightCurveAvailable"].eq(True).sum())
        totalCount = int(len(df))
        failedCount = totalCount - successCount
        self.logger.info(
            "Download summary: total=%d succeeded=%d failed=%d duplicateTicWarnings=%d",
            totalCount,
            successCount,
            failedCount,
            duplicateWarningCount,
        )
        self.logger.info("Saved augmented metadata to %s", augmentedPath)
        return df
