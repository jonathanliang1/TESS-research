from VSXCategoryLoader import VSXCategoryLoader
from VSX2TESSConverter import VSX2TESSConverter
from TessDataDownloader import TessDataDownloader
from collections import defaultdict
import pandas as pd
import logging
import sys
import os

class ResearchPipeline:
    """
    ResearchPipeline orchestrates the loading of VSX variable-star categories,
    crossmatching them to the TESS Input Catalog, and writing the resulting
    TIC IDs to a CSV file for downstream analysis.
    """
    def __init__(self, vsxLoader, tessConverter, nFamilies = None, nInstances=1000, vsxCacheFolder='VSXCache', tessCacheFolder='TESSCache'):
        self.loader = vsxLoader
        self.converter = tessConverter
        self.vsxCacheFolder = vsxCacheFolder
        self.tessCacheFolder = tessCacheFolder
        if nFamilies is not None:
            self.varFamilies = self.loader.getSupportedFamilies()[:nFamilies]
        else:
            self.varFamilies = self.loader.getSupportedFamilies()
        self.nInstances = nInstances
        self.ticFile = None
        self.tessMatchedDict = None
        self.logger = logging.getLogger()

    def combineVSXTESS(self):
        #This is a time consuming operation, as it involves crossmatching potentially
        #thousands of VSX variable stars against the TESS Input Catalog
        #This function is supposed to called only once to generate the CSV of TIC IDs
        #for downstream analysis, since the crossmatching step is expensive

        varReq = {fam: self.nInstances for fam in self.varFamilies}

        self.logger.info("Loading categories from VSX")
        categoryDictionary = self.loader.loadCategories(varReq)

        self.logger.info("Crossmatching categories to TIC")
        self.tessMatchedDict = self.converter.crossmatchCategoryDictionaryToTic(categoryDictionary)
        self.tessMatchedDict = {
            family: stars for family, stars in self.tessMatchedDict.items() if stars
        }
        return self.tessMatchedDict
    
    def cacheMetadata(self, tessMarchedDict, metadataFile='VSXMetadata.parquet'):
        df = []
        for family, stars in tessMarchedDict.items():
            for star in stars:
                record = dict(star)
                df.append(record)
        df = pd.DataFrame(df)
        df.to_parquet(self.tessCacheFolder + os.path.sep + metadataFile, index=False)

    def loadTicFile(self, ticFile="tess_matched_stars.csv"):
        """
        Loads a previously generated CSV of TIC IDs into memory for downstream analysis.
        """
        self.ticFile = self.tessCacheFolder + os.path.sep + ticFile
        self.tessMatchedDict = {}
        with open(self.ticFile, "r") as f:
            for line in f:
                family, *ticIds = line.strip().split(",")
                self.tessMatchedDict[family] = [{"ticId": ticId} for ticId in ticIds]
        
        return self.tessMatchedDict
    
    def loadCachedMetadata(self, metadataFile="VSXMetadata.parquet"):
        self.metadataFile = self.tessCacheFolder + os.path.sep + metadataFile
        if os.path.exists(self.metadataFile):
            df = pd.read_parquet(self.metadataFile)
            self.tessMatchedDict = defaultdict(list)
            for _, row in df.iterrows():
                family = row.get("family")
                if family is not None:
                    starRecord = row.to_dict()
                    self.tessMatchedDict[family].append(starRecord)
            return self.tessMatchedDict
        else:
            self.logger.error(f"Metadata file {self.metadataFile} does not exist.")
            return RuntimeError(f"Metadata file {self.metadataFile} does not exist.")
    
    def loadCandidates(self, fromCache=True):
        """
        if fromCache is True, attempts to load cached metadata and TIC IDs.
        If fromCache is False or cached data is unavailable, runs the full pipeline
        from scratch: loads VSX categories, crossmatches them to the TESS Input Catalog,
        and caches both the TIC IDs and metadata for downstream analysis.
        """
        if fromCache:
            tessData = self.loadCachedMetadata()
            if tessData is not None:
                return tessData
            else:
                self.logger.error("Cached TESS metadata not found or failed to load. Running full pipeline.")
                raise RuntimeError("Cached TESS metadata not found or failed to load. Running full pipeline.")

        tessData = self.combineVSXTESS()
        self.cacheMetadata(tessData, metadataFile="VSXMetadata.parquet")
        return tessData

    def countBestMatchesByFamily(self, tessMetadataParquet):
        """
        Count stars with a valid bestMatch for each family from cached parquet metadata.

        Returns:
            tuple[dict[str, int], int]:
                - per_family_count: number of stars with bestMatch found in each family
                - total_best_matches: total number of stars with bestMatch found
        """
        df = pd.read_parquet(tessMetadataParquet)

        required_columns = {"family", "bestMatch"}
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise ValueError(
                f"Missing required columns in {tessMetadataParquet}: {sorted(missing_columns)}"
            )

        # Treat null/NaN and literal string 'None' as no match.
        has_match = df["bestMatch"].notna() & (df["bestMatch"].astype(str) != "None")

        per_family_count = (
            df.loc[has_match]
            .groupby("family")
            .size()
            .astype(int)
            .to_dict()
        )
        total_best_matches = int(has_match.sum())

        return per_family_count, total_best_matches


if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(levelname)s:%(filename)s:%(lineno)d:%(message)s",
        force=True,
    )
    logger = logging.getLogger()
    reload = False

    vsxLoader = VSXCategoryLoader(refreshCache=reload)
    tessConverter = VSX2TESSConverter()

    pipeline = ResearchPipeline(vsxLoader, tessConverter, nFamilies = 5, nInstances=5)
    tessMetadata = pipeline.loadCandidates()
    print("TESS Metadata loaded for %d variable star candidates across %d families" % (sum(len(stars) for stars in tessMetadata.values()), len(tessMetadata)))
    stat, total = pipeline.countBestMatchesByFamily(pipeline.tessCacheFolder + os.path.sep + "VSXMetadata.parquet")
    for family, count in stat.items():
        print(f"Family {family}: {count} stars with bestMatch, out of {len(tessMetadata.get(family, []))} total stars in family")
    print(f"Total stars with bestMatch: {total} out of {sum(len(stars) for stars in tessMetadata.values())} candidates")
    tessDataloader = TessDataDownloader()
    augmented = tessDataloader.downloadTessLightCurves("VSXMetadata.parquet")
    # lcurves = tessDataloader.dlSample(tmpticDict, refresh=True)
    # lcurves
    # tessDataloader = TessDataDownloader()
    # lcurves = tessDataloader.scanAvailability(ticDict)
