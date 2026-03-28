import pandas as pd
import lightkurve as lk
import logging


class TessDataDownloader:
    """
    A class to download TESS light curve data organized by stellar categories.
    """
    
    def __init__(self):
        """Initialize the TessDataDownloader."""
        self.logger = logging.getLogger(__name__)
        
    def load(self, symbolFile):
        """
        Load TIC IDs by category and download corresponding light curve objects.
        
        Args:
            symbolFile (str): Path to CSV file containing TIC IDs organized by categories
            
        Returns:
            dict: Nested dictionary with structure {category: {ticId: lightCurve}}
                where category is the column name, ticId is the TIC ID as string,
                and lightCurve is the downloaded lightkurve object
        """
        try:
            # Read the CSV file
            dataFrame = pd.read_csv(symbolFile)
            
            # Initialize the result dictionary
            resultDict = {}
            
            # Process each category (column) in the dataFrame
            for categoryName in dataFrame.columns:
                self.logger.info(f"Processing category: {categoryName}")
                resultDict[categoryName] = {}
                
                # Get TIC IDs for this category, dropping NaN values
                ticIds = dataFrame[categoryName].dropna()
                
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
            self.logger.error(f"Error reading symbol file {symbolFile}: {fileError}")
            raise

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    downloader = TessDataDownloader()
    data = downloader.load('sample_stars.csv')
    print(data)