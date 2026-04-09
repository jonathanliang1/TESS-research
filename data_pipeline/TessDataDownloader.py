import pandas as pd
import lightkurve as lk
import logging
import os
import sys
import time
import shutil
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

    def scanAvailability(
        self,
        tessMetadata,
        maxWorkers=6,
        minQueryIntervalSec=0.25,
        maxRetries=1,
        retryBackoffBaseSec=1.0,
        batchSize=200,
    ):
        from astroquery.mast import Observations

        def _is_rate_limited_error(exc):
            statusCode = getattr(getattr(exc, "response", None), "status_code", None)
            if statusCode == 429:
                return True
            return "429" in str(exc) or "Too Many Requests" in str(exc)

        def _normalize_tic(value):
            if value is None:
                return None
            digits = "".join(ch for ch in str(value) if ch.isdigit())
            if not digits:
                return None
            return str(int(digits))

        def _chunked(items, chunkSize):
            for idx in range(0, len(items), chunkSize):
                yield items[idx : idx + chunkSize]

        def _query_target_names(targetNames, logLabel):
            targetCount = len(targetNames)
            if targetCount == 0:
                return []

            for attempt in range(maxRetries + 1):
                try:
                    observations = Observations.query_criteria(
                        project=["TESS"],
                        provenance_name=["SPOC", "QLP"],
                        dataproduct_type=["cube", "timeseries"],
                        target_name=targetNames,
                    )
                    if observations is None:
                        return []
                    return [row for row in observations]
                except Exception as exc:
                    if attempt < maxRetries and _is_rate_limited_error(exc):
                        sleepSec = retryBackoffBaseSec * (2 ** attempt)
                        self.logger.warning(
                            "429 during %s (%d TICs). Retrying in %.1fs (attempt %d/%d)",
                            logLabel,
                            targetCount,
                            sleepSec,
                            attempt + 1,
                            maxRetries,
                        )
                        time.sleep(sleepSec)
                        continue
                    self.logger.error("%s failed (%d TICs): %s", logLabel, targetCount, exc)
                    return None

        def _query_chunk(chunkIndex, targetNames):
            # Stagger requests from worker threads to reduce burst traffic.
            staggerDelaySec = minQueryIntervalSec * (chunkIndex % max(1, int(maxWorkers)))
            if staggerDelaySec > 0:
                time.sleep(staggerDelaySec)

            return _query_target_names(targetNames, f"bulk chunk {chunkIndex}")

        starRefs = []
        familyCounts = {family: 0 for family in tessMetadata}
        familyQueriedCounts = {family: 0 for family in tessMetadata}
        availCount = 0
        totalCount = 0
        maxWorkers = max(1, int(maxWorkers))

        for family, stars in tessMetadata.items():
            totalCount += len(stars)
            for star in stars:
                star["author"] = None
                cleanTicId = _normalize_tic(star.get("ticId"))
                if cleanTicId is None:
                    continue
                familyQueriedCounts[family] += 1
                starRefs.append((family, star, cleanTicId))

        if not starRefs:
            self.logger.info("No valid TIC IDs to scan.")
            return

        uniqueTicIds = sorted({cleanTicId for _, _, cleanTicId in starRefs})
        chunks = list(_chunked(uniqueTicIds, max(1, int(batchSize))))
        authorPriority = {"SPOC": 2, "QLP": 1}
        bestAuthorByTic = {}

        with ThreadPoolExecutor(max_workers=maxWorkers) as executor:
            futures = {
                executor.submit(
                    _query_chunk,
                    chunkIndex,
                    [ticId.zfill(9) for ticId in ticChunk],
                ): chunkIndex
                for chunkIndex, ticChunk in enumerate(chunks)
            }

            for future in as_completed(futures):
                rows = future.result()
                if rows is None or len(rows) == 0:
                    continue

                for row in rows:
                    rowAuthor = str(row["provenance_name"]).strip().upper()
                    if rowAuthor not in authorPriority:
                        continue
                    rowTic = _normalize_tic(row["target_name"])
                    if rowTic is None:
                        continue
                    prevAuthor = bestAuthorByTic.get(rowTic)
                    if prevAuthor is None or authorPriority[rowAuthor] > authorPriority.get(prevAuthor, 0):
                        bestAuthorByTic[rowTic] = rowAuthor

        for family, star, cleanTicId in starRefs:
            author = bestAuthorByTic.get(cleanTicId)
            if author is not None:
                star["author"] = author
                familyCounts[family] += 1
                availCount += 1

        for family, lcCount in familyCounts.items():
            queriedCount = familyQueriedCounts.get(family, 0)
            self.logger.info(
                f"Found {lcCount} light curves for family {family} out of {queriedCount} stars queried"
            )
        self.logger.info(
            f"Completed scanning availability of light curves. Total {availCount} light curves found out of {totalCount} stars."
        )
        
    def dlSample(self, tessMetadata, product = 0, flux="pdcsap_flux", author="SPOC", refresh=False):
        """
        Download light curves for TIC IDs with specified product and flux column from the TESS mission.
        
        Args:
            tessMetadata (dict): Dictionary mapping stellar categories to lists of TIC IDs to download.
            product (int, optional): TESS data product to download (default is 0).
            flux (str, optional): Name of the flux column to retrieve from the light curve (default is "pdcsap_flux").
            
        Returns:
            dict: Nested dictionary with structure {category: {ticId: lightCurve}}
                where category is the variable star family, ticId is the TIC ID as string,
                and lightCurve is the downloaded lightkurve object
        """
        try:
            resultDict = {}
            for family, stars in tessMetadata.items():
                self.logger.info(f"Processing {family}")
                resultDict[family] = {}
                for star in stars:
                    # Clean the TIC ID string (remove 'TIC ' prefix if present)
                    ticId = star.get("ticId")
                    cleanTicId = str(ticId).replace('TIC ', '').strip()                    
                    cacheFileSuffix = f"{flux}_{product}"
                    
                    try:
                        cacheFilePath = os.path.join(self.tessCacheFolder, f"TIC_{cleanTicId}_{cacheFileSuffix}.FITS")        
                        lightCurve = None
                        if refresh or not os.path.exists(cacheFilePath):
                            self.logger.info(f"Downloading light curve for TIC {cleanTicId}")
                            
                            # Search for and download the light curve
                            searchResult = None
                            searchResult = lk.search_lightcurve(f"TIC {cleanTicId}", 
                                                                mission='TESS',
                                                                author=author)                            
                            if len(searchResult) > 0:
                                # Download the first available light curve
                                lightCurve = searchResult[product].download().select_flux(flux)# Cache the downloaded light curve in npz format                                  
                                self.logger.info(f"Successfully downloaded light curve for TIC {cleanTicId}")
                                try:
                                    shutil.move(lightCurve.filename, cacheFilePath)
                                    self.logger.info(f"Cached light curve for TIC {cleanTicId} at {cacheFilePath}")
                                except Exception as cacheError:
                                    self.logger.error(f"Error caching light curve for TIC {cleanTicId}: {cacheError}")
                            else:
                                self.logger.warning(f"No light curve data found for TIC {cleanTicId}")
                        else:
                            # load previously cached light curve from TESSCache, in FITS format
                            try:
                                lightCurve = lk.read(cacheFilePath).select_flux(flux)
                                self.logger.info(f"Loaded cached light curve for TIC {cleanTicId}")
                            except Exception as cacheError:
                                self.logger.error(f"Error loading cached light curve for TIC {cleanTicId}: {cacheError}")
                                lightCurve = None
                        resultDict[family][cleanTicId] = lightCurve
                            
                    except Exception as downloadError:
                        self.logger.error(f"Error downloading light curve for TIC {cleanTicId}: {downloadError}")
                        resultDict[family][cleanTicId] = None
            
            return resultDict
            
        except Exception as fileError:
            self.logger.error(f"Error reading parquet file {tessMetadata}: {fileError}")
            raise

    
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
        stitch=True,
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
                sector=sector,
            )
        except Exception as exc:
            self.logger.warning("search_lightcurve failed for %s: %s", targetName, exc)
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
            self.logger.warning("download failed for %s: %s", targetName, exc)
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

    def attachLightCurvesToCategoryDictionary(
        self,
        categoryDictionary,
        author="SPOC",
        exptime=120,
        sector=None,
        fluxColumn="pdcsap_flux",
        stitch=True,
        preprocess=True,
        preprocessConfig=None,
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
                "timeBinSize": 0.01,
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
                    stitch=stitch,
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

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    downloader = TessDataDownloader()
    data = downloader.load('sample_stars.csv')
    print(data)