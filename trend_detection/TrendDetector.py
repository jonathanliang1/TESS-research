"""
TrendDetector.py

Detect long-term trends in TESS light curves using Lomb-Scargle periodogram
analysis on low frequencies.
"""

import logging
import glob
from pathlib import Path
from typing import Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.timeseries import LombScargle


class TrendDetector:
    """
    Detect long-term trends in light curves using Lomb-Scargle periodogram.
    """

    def __init__(self):
        self.logger = logging.getLogger("TrendDetector")

    def detectLongTermTrendWithLS(
        self,
        rawFitsPath: str,
        minFinitePoints: int = 100,
        lowFreqPowerRatioThreshold: float = 5.0,
        lowFreqMaxPowerThreshold: float = 0.25,
    ) -> Dict:
        """
        Detect long-term trends in a light curve using Lomb-Scargle analysis.

        Args:
            rawFitsPath: Path to the raw FITS light curve file
            minFinitePoints: Minimum number of finite time/flux points required
            lowFreqPowerRatioThreshold: Ratio threshold for trend detection
            lowFreqMaxPowerThreshold: Absolute power threshold for trend detection

        Returns:
            Dictionary with keys:
                - trendDetected (bool): True if trend is detected
                - trendStatus (str): "trend", "no_trend", "insufficient_data", or "failed"
                - trendScore (float): Trend significance score
                - lowFreqMaxPower (float or NaN): Max low-frequency power
                - lowFreqPowerRatio (float or NaN): Ratio of low-freq to ref power
                - trendBaselineDays (float or NaN): Time span of light curve in days
                - trendNumFinitePoints (int): Number of finite time/flux points
                - trendMethod (str): Method identifier
        """
        try:
            # Load the raw FITS file
            self.logger.info(f"Loading raw FITS file: {rawFitsPath}")
            with fits.open(rawFitsPath, memmap=False) as hdul:
                if len(hdul) <= 1 or hdul[1].data is None:
                    raise ValueError("Missing table data in HDU[1]")

                table = hdul[1].data
                names = set(getattr(table, "names", []) or [])
                if "TIME" not in names or "FLUX" not in names:
                    raise ValueError("HDU[1] must contain TIME and FLUX columns")

                time = np.asarray(table["TIME"], dtype=float)
                flux = np.asarray(table["FLUX"], dtype=float)

            self.logger.debug(f"Initial time/flux size: {len(time)}")

            # Remove NaN, inf, and non-finite rows
            finite_mask = np.isfinite(time) & np.isfinite(flux)
            time_finite = time[finite_mask]
            flux_finite = flux[finite_mask]

            num_finite = len(time_finite)
            self.logger.info(f"Number of finite time/flux points: {num_finite}")

            # Check if we have enough data
            if num_finite < minFinitePoints:
                self.logger.warning(
                    f"Insufficient data for trend detection: {num_finite} < {minFinitePoints}"
                )
                return {
                    "trendDetected": False,
                    "trendStatus": "insufficient_data",
                    "trendScore": np.nan,
                    "lowFreqMaxPower": np.nan,
                    "lowFreqPowerRatio": np.nan,
                    "trendBaselineDays": np.nan,
                    "trendNumFinitePoints": num_finite,
                    "trendMethod": "ls_low_frequency_power_v1",
                }

            # Normalize time to start at 0
            time_normalized = time_finite - time_finite[0]

            # Calculate baseline in days
            baseline_days = time_normalized[-1] - time_normalized[0]
            if baseline_days <= 0:
                self.logger.warning("Non-positive baseline duration")
                return {
                    "trendDetected": False,
                    "trendStatus": "insufficient_data",
                    "trendScore": np.nan,
                    "lowFreqMaxPower": np.nan,
                    "lowFreqPowerRatio": np.nan,
                    "trendBaselineDays": baseline_days,
                    "trendNumFinitePoints": num_finite,
                    "trendMethod": "ls_low_frequency_power_v1",
                }

            self.logger.info(f"Baseline duration: {baseline_days:.2f} days")

            # Define frequency ranges
            # Low-frequency range: periods from 0.5*baseline to 2*baseline
            low_freq_min = 1.0 / (2.0 * baseline_days)  # cycles/day
            low_freq_max = 1.0 / (0.5 * baseline_days)  # cycles/day

            # Reference frequency range: broader range excluding very low frequencies
            ref_freq_min = 1.0 / baseline_days  # cycles/day
            ref_freq_max = 24.0  # cycles/day (1 cycle per hour or faster)

            self.logger.debug(
                f"Low-freq range: {low_freq_min:.6f} to {low_freq_max:.6f} cycles/day"
            )
            self.logger.debug(
                f"Ref-freq range: {ref_freq_min:.6f} to {ref_freq_max:.6f} cycles/day"
            )

            # Compute Lomb-Scargle periodogram over low-frequency range
            low_freq_range = np.linspace(low_freq_min, low_freq_max, 500)
            ls_low = LombScargle(time_normalized, flux_finite, normalization="psd")
            low_freq_power = ls_low.power(low_freq_range)

            lowFreqMaxPower = float(np.max(low_freq_power))
            self.logger.info(f"Low-frequency max power: {lowFreqMaxPower:.6f}")

            # Compute Lomb-Scargle periodogram over reference frequency range
            ref_freq_range = np.linspace(ref_freq_min, ref_freq_max, 2000)
            ls_ref = LombScargle(time_normalized, flux_finite, normalization="psd")
            ref_freq_power = ls_ref.power(ref_freq_range)

            refMedianPower = float(np.median(ref_freq_power))
            self.logger.info(f"Reference median power: {refMedianPower:.6f}")

            # Calculate power ratio
            if refMedianPower > 0:
                lowFreqPowerRatio = lowFreqMaxPower / refMedianPower
            else:
                lowFreqPowerRatio = np.nan

            self.logger.info(f"Low-frequency power ratio: {lowFreqPowerRatio:.6f}")

            # Determine if trend is detected
            trendDetected = (
                lowFreqPowerRatio >= lowFreqPowerRatioThreshold
                or lowFreqMaxPower >= lowFreqMaxPowerThreshold
            )

            if trendDetected:
                trendStatus = "trend"
                trendScore = max(lowFreqPowerRatio, lowFreqMaxPower)
                self.logger.warning(f"TREND DETECTED: score={trendScore:.6f}")
            else:
                trendStatus = "no_trend"
                trendScore = max(lowFreqPowerRatio, lowFreqMaxPower)
                self.logger.info(f"No trend detected: score={trendScore:.6f}")

            return {
                "trendDetected": trendDetected,
                "trendStatus": trendStatus,
                "trendScore": float(trendScore),
                "lowFreqMaxPower": lowFreqMaxPower,
                "lowFreqPowerRatio": lowFreqPowerRatio,
                "trendBaselineDays": float(baseline_days),
                "trendNumFinitePoints": num_finite,
                "trendMethod": "ls_low_frequency_power_v1",
            }

        except Exception as e:
            self.logger.exception(f"Trend detection failed for {rawFitsPath}: {e}")
            return {
                "trendDetected": False,
                "trendStatus": "failed",
                "trendScore": np.nan,
                "lowFreqMaxPower": np.nan,
                "lowFreqPowerRatio": np.nan,
                "trendBaselineDays": np.nan,
                "trendNumFinitePoints": 0,
                "trendMethod": "ls_low_frequency_power_v1",
            }

    def _process_single_row(self, idx: int, row: pd.Series, fits_folder: Path) -> Tuple[int, dict]:
        """
        Process a single row for trend detection.

        Args:
            idx: Row index
            row: Row data
            fits_folder: Path to FITS folder

        Returns:
            Tuple of (idx, results_dict)
        """
        result = {
            "lsTrendDetected": False,
            "lsTrendStatus": "failed",
            "lsTrendScore": np.nan,
            "lsTrendMaxPower": np.nan,
            "lsTrendPowerRatio": np.nan,
            "lsTrendBaselineDays": np.nan,
            "lsTrendFinitePoints": 0,
            "lsTrendMethod": "ls_low_frequency_power_v1",
        }

        # Extract VSX ID to find raw FITS file
        vsx_id = row.get("VSXId")
        if not vsx_id:
            self.logger.debug(f"Row {idx}: No VSXId, skipping")
            return idx, result

        # Find raw FITS file matching this VSX ID
        sanitized_vsx = str(vsx_id).replace("/", "_").replace(" ", "_")
        raw_fits_files = list(fits_folder.glob(f"VSX_{sanitized_vsx}_*_raw.fits"))

        if not raw_fits_files:
            self.logger.debug(f"Row {idx} ({vsx_id}): No raw FITS file found")
            return idx, result

        raw_fits_path = str(raw_fits_files[0])
        self.logger.debug(f"Row {idx} ({vsx_id}): Found raw FITS file: {raw_fits_path}")

        # Run trend detection
        trend_result = self.detectLongTermTrendWithLS(raw_fits_path)

        # Return mapped results
        return idx, {
            "lsTrendDetected": trend_result["trendDetected"],
            "lsTrendStatus": trend_result["trendStatus"],
            "lsTrendScore": trend_result["trendScore"],
            "lsTrendMaxPower": trend_result["lowFreqMaxPower"],
            "lsTrendPowerRatio": trend_result["lowFreqPowerRatio"],
            "lsTrendBaselineDays": trend_result["trendBaselineDays"],
            "lsTrendFinitePoints": trend_result["trendNumFinitePoints"],
            "lsTrendMethod": trend_result["trendMethod"],
        }

    def run(self, metaParquetFile: str, fitsFolder: str, maxWorkers: int = 16, outputPath: Optional[str] = None) -> str:
        """
        Process a metadata parquet file and run trend detection on associated FITS files using multithreading.

        Args:
            metaParquetFile: Path to the metadata parquet file
            fitsFolder: Folder containing raw FITS light curve files
            maxWorkers: Maximum number of worker threads (default: 16)
            outputPath: Output directory path (optional; default: same folder as input)

        Returns:
            Path to the output parquet file with trend detection results
        """
        # Load metadata parquet
        self.logger.info(f"Loading metadata from {metaParquetFile}")
        df = pd.read_parquet(metaParquetFile)
        self.logger.info(f"Loaded {len(df)} rows from metadata")

        # Add columns for trend detection results
        trend_columns = [
            "lsTrendDetected",
            "lsTrendStatus",
            "lsTrendScore",
            "lsTrendMaxPower",
            "lsTrendPowerRatio",
            "lsTrendBaselineDays",
            "lsTrendFinitePoints",
            "lsTrendMethod",
        ]
        for col in trend_columns:
            if col not in df.columns:
                df[col] = None

        fits_folder = Path(fitsFolder)
        if not fits_folder.exists() or not fits_folder.is_dir():
            self.logger.error(f"FITS folder not found or not a directory: {fitsFolder}")
            raise ValueError(f"Invalid FITS folder: {fitsFolder}")

        # Process each star using thread pool
        processed_count = 0
        trend_detected_count = 0
        no_trend_count = 0
        insufficient_count = 0
        failed_count = 0

        self.logger.info(f"Starting trend detection with {maxWorkers} worker threads")

        with ThreadPoolExecutor(max_workers=maxWorkers) as executor:
            # Submit all tasks
            futures = {
                executor.submit(self._process_single_row, idx, df.loc[idx], fits_folder): idx
                for idx in df.index
            }

            # Process results as they complete
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    result_idx, result_data = future.result()

                    # Update dataframe with results
                    for col, val in result_data.items():
                        df.at[result_idx, col] = val

                    # Update counters
                    processed_count += 1
                    status = result_data["lsTrendStatus"]
                    if status == "trend":
                        trend_detected_count += 1
                    elif status == "no_trend":
                        no_trend_count += 1
                    elif status == "insufficient_data":
                        insufficient_count += 1
                    elif status == "failed":
                        failed_count += 1

                    if processed_count % 100 == 0:
                        self.logger.info(f"Processed {processed_count} rows")

                except Exception as e:
                    self.logger.exception(f"Error processing row {idx}: {e}")

        # Save results
        output_file_name = f"{Path(metaParquetFile).stem}_trend.parquet"
        if outputPath:
            output_dir = Path(outputPath)
        else:
            output_dir = Path(metaParquetFile).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / output_file_name
        df.to_parquet(output_path, index=False)
        self.logger.info(f"Results saved to {output_path}")

        # Log summary
        self.logger.info(f"\nTrend Detection Summary:")
        self.logger.info(f"  Processed: {processed_count}")
        self.logger.info(f"  Trend detected: {trend_detected_count}")
        self.logger.info(f"  No trend: {no_trend_count}")
        self.logger.info(f"  Insufficient data: {insufficient_count}")
        self.logger.info(f"  Failed: {failed_count}")

        return str(output_path)
    
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run trend detection on TESS light curves")
    parser.add_argument("--metaParquetFile", type=str, required=True, help="Path to metadata parquet file")
    parser.add_argument("--fitsFolder", type=str, required=True, help="Folder containing raw FITS files")
    parser.add_argument("--maxWorkers", type=int, default=16, help="Maximum number of worker threads")
    parser.add_argument("--outputPath", type=str, default=None, help="Output directory for trend parquet file (optional)")
    args = parser.parse_args()

    trend_detector = TrendDetector()
    trend_detector.run(args.metaParquetFile, args.fitsFolder, args.maxWorkers, args.outputPath)
