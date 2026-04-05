import pandas as pd
import lightkurve as lk
import logging
import os
import sys
import time

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
        
    def load(self, tessDataReq):
        """
        Load TIC IDs by category and download corresponding light curve objects.
        
        Args:
            tessDataReq: lists of TIC IDs to download
            
        Returns:
            dict: Nested dictionary with structure {category: {ticId: lightCurve}}
                where category is the column name, ticId is the TIC ID as string,
                and lightCurve is the downloaded lightkurve object
        """
        try:
            # Read the parquet file
            tessMetadata = pd.read_parquet(tessDataReq)
            
            # Initialize the result dictionary
            resultDict = {}
            
            # Process each category (column) in the dataFrame
            for categoryName in tessMetadata.columns:
                self.logger.info(f"Processing category: {categoryName}")
                resultDict[categoryName] = {}
                
                # Get TIC IDs for this category, dropping NaN values
                ticIds = tessMetadata[categoryName].dropna()
                
                # Download light curves for each TIC ID in this category
                for ticId in ticIds:
                    # Clean the TIC ID string (remove 'TIC ' prefix if present)
                    cleanTicId = str(ticId).replace('TIC ', '').strip()
                    
                    try:
                        self.logger.info(f"Downloading light curve for TIC {cleanTicId}")
                        
                        # Search for and download the light curve
                        searchResult = lk.search_lightcurve(f"TIC {cleanTicId}", mission='TESS')
                        
                        if len(searchResult) > 0:
                            # Download the first available light curve
                            lightCurve = searchResult[0].download()
                            resultDict[categoryName][cleanTicId] = lightCurve
                            self.logger.info(f"Successfully downloaded light curve for TIC {cleanTicId}")
                        else:
                            self.logger.warning(f"No light curve data found for TIC {cleanTicId}")
                            resultDict[categoryName][cleanTicId] = None
                            
                    except Exception as downloadError:
                        self.logger.error(f"Error downloading light curve for TIC {cleanTicId}: {downloadError}")
                        resultDict[categoryName][cleanTicId] = None
            
            return resultDict
            
        except Exception as fileError:
            self.logger.error(f"Error reading parquet file {tessDataReq}: {fileError}")
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
            logger.warning("preprocessLightCurve failed: %s", exc)
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