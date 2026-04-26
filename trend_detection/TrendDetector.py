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

# ---------------------------------------------------------------------------
# Configurable defaults – change here rather than hunting through method bodies
# ---------------------------------------------------------------------------
DEFAULT_LOW_FREQ_POWER_RATIO_THRESHOLD: float = 8.0
DEFAULT_MIN_FINITE_POINTS: int = 100
DEFAULT_LOW_FREQ_N_SAMPLES: int = 500
DEFAULT_REF_FREQ_N_SAMPLES: int = 2000
DEFAULT_REF_FREQ_MAX: float = 24.0          # cycles/day  (1 cycle per hour)
DEFAULT_LOW_FREQ_PERIOD_MIN_FACTOR: float = 0.5   # period = factor * baseline
DEFAULT_LOW_FREQ_PERIOD_MAX_FACTOR: float = 2.0   # period = factor * baseline
DEFAULT_REF_FREQ_MIN_FACTOR: float = 1.5          # refFreqMin = factor * lowFreqMax
TREND_METHOD: str = "ls_low_frequency_power_v2"
# ---------------------------------------------------------------------------


class TrendDetector:
    """
    Detect long-term trends in light curves using Lomb-Scargle periodogram.
    """

    def __init__(self):
        self.logger = logging.getLogger("TrendDetector")

    def detectLongTermTrendWithLS(
        self,
        rawFitsPath: str,
        minFinitePoints: int = DEFAULT_MIN_FINITE_POINTS,
        lowFreqPowerRatioThreshold: float = DEFAULT_LOW_FREQ_POWER_RATIO_THRESHOLD,
    ) -> Dict:
        """
        Detect long-term low-frequency power excess in a light curve using Lomb-Scargle.

        Decision is ratio-based only: lsLowFreqPowerRatio >= lowFreqPowerRatioThreshold.
        lowFreqMaxPower is stored as a diagnostic value only.

        Frequency bands (cycles/day):
            low-frequency : [1/(2*baseline), 1/(0.5*baseline)]
            reference     : [1.5*lowFreqMax, 24.0]

        Args:
            rawFitsPath: Path to the pipeline-generated raw FITS file
            minFinitePoints: Minimum number of finite TIME/FLUX points required
            lowFreqPowerRatioThreshold: Ratio threshold for low-frequency flag

        Returns:
            Dict with keys:
                lsLowFrequencyPowerDetected (bool)
                lsTrendStatus (str): "low_frequency_power" | "no_low_frequency_power"
                                     | "insufficient_data" | "failed"
                lsLowFreqPowerRatio (float)
                lsLowFreqMaxPower (float)   – diagnostic only
                lsRefMedianPower (float)    – diagnostic only
                lsTrendBaselineDays (float)
                lsTrendNumFinitePoints (int)
                lsTrendMethod (str)
                lsLowFreqPowerRatioThreshold (float)
                lsRefFreqMin (float)
                lsRefFreqMax (float)
                lsLowFreqMin (float)
                lsLowFreqMax (float)
        """
        _insufficient: Dict = {
            "lsLowFrequencyPowerDetected": False,
            "lsTrendStatus": "insufficient_data",
            "lsLowFreqPowerRatio": np.nan,
            "lsLowFreqMaxPower": np.nan,
            "lsRefMedianPower": np.nan,
            "lsTrendBaselineDays": np.nan,
            "lsTrendNumFinitePoints": 0,
            "lsTrendMethod": TREND_METHOD,
            "lsLowFreqPowerRatioThreshold": lowFreqPowerRatioThreshold,
            "lsRefFreqMin": np.nan,
            "lsRefFreqMax": np.nan,
            "lsLowFreqMin": np.nan,
            "lsLowFreqMax": np.nan,
        }

        try:
            # ------------------------------------------------------------------
            # 1. Load pipeline-generated raw FITS table from HDU[1]
            # ------------------------------------------------------------------
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

            # ------------------------------------------------------------------
            # 2. Finite-point filter
            # ------------------------------------------------------------------
            finite_mask = np.isfinite(time) & np.isfinite(flux)
            time_finite = time[finite_mask]
            flux_finite = flux[finite_mask]
            num_finite = len(time_finite)
            self.logger.info(f"Finite time/flux points: {num_finite}")

            if num_finite < minFinitePoints:
                self.logger.warning(
                    f"Insufficient data: {num_finite} < {minFinitePoints} required"
                )
                result = dict(_insufficient)
                result["lsTrendNumFinitePoints"] = num_finite
                return result

            # ------------------------------------------------------------------
            # 3. Baseline check
            # ------------------------------------------------------------------
            time_normalized = time_finite - time_finite[0]
            baseline_days = float(time_normalized[-1] - time_normalized[0])
            if baseline_days <= 0:
                self.logger.warning(f"Non-positive baseline: {baseline_days}")
                result = dict(_insufficient)
                result["lsTrendNumFinitePoints"] = num_finite
                result["lsTrendBaselineDays"] = baseline_days
                return result

            self.logger.info(f"Baseline duration: {baseline_days:.2f} days")

            # ------------------------------------------------------------------
            # 4. Frequency bands
            # ------------------------------------------------------------------
            low_freq_min = 1.0 / (DEFAULT_LOW_FREQ_PERIOD_MAX_FACTOR * baseline_days)
            low_freq_max = 1.0 / (DEFAULT_LOW_FREQ_PERIOD_MIN_FACTOR * baseline_days)
            ref_freq_min = DEFAULT_REF_FREQ_MIN_FACTOR * low_freq_max
            ref_freq_max = DEFAULT_REF_FREQ_MAX

            self.logger.debug(
                f"Low-freq band: {low_freq_min:.6f} – {low_freq_max:.6f} cycles/day"
            )
            self.logger.debug(
                f"Ref-freq band: {ref_freq_min:.6f} – {ref_freq_max:.6f} cycles/day"
            )

            if ref_freq_min >= ref_freq_max:
                self.logger.warning(
                    f"Reference band degenerate: refFreqMin={ref_freq_min:.6f} >= refFreqMax={ref_freq_max:.6f}"
                )
                result = dict(_insufficient)
                result.update({
                    "lsTrendNumFinitePoints": num_finite,
                    "lsTrendBaselineDays": baseline_days,
                    "lsLowFreqMin": low_freq_min,
                    "lsLowFreqMax": low_freq_max,
                    "lsRefFreqMin": ref_freq_min,
                    "lsRefFreqMax": ref_freq_max,
                })
                return result

            # ------------------------------------------------------------------
            # 5. Lomb-Scargle over low-frequency band
            # ------------------------------------------------------------------
            low_freq_range = np.linspace(low_freq_min, low_freq_max, DEFAULT_LOW_FREQ_N_SAMPLES)
            ls_low = LombScargle(time_normalized, flux_finite, normalization="psd")
            low_freq_power = ls_low.power(low_freq_range)
            low_freq_max_power = float(np.max(low_freq_power))
            self.logger.info(f"Low-freq max power: {low_freq_max_power:.6f}")

            # ------------------------------------------------------------------
            # 6. Lomb-Scargle over reference band
            # ------------------------------------------------------------------
            ref_freq_range = np.linspace(ref_freq_min, ref_freq_max, DEFAULT_REF_FREQ_N_SAMPLES)
            ls_ref = LombScargle(time_normalized, flux_finite, normalization="psd")
            ref_freq_power = ls_ref.power(ref_freq_range)
            ref_median_power = float(np.median(ref_freq_power))
            self.logger.info(f"Ref median power: {ref_median_power:.6f}")

            # ------------------------------------------------------------------
            # 7. Safety: reference median must be finite and positive
            # ------------------------------------------------------------------
            if not np.isfinite(ref_median_power) or ref_median_power <= 0:
                self.logger.warning(
                    f"Reference median power invalid: {ref_median_power}"
                )
                result = dict(_insufficient)
                result.update({
                    "lsTrendNumFinitePoints": num_finite,
                    "lsTrendBaselineDays": baseline_days,
                    "lsLowFreqMaxPower": low_freq_max_power,
                    "lsRefMedianPower": ref_median_power,
                    "lsLowFreqMin": low_freq_min,
                    "lsLowFreqMax": low_freq_max,
                    "lsRefFreqMin": ref_freq_min,
                    "lsRefFreqMax": ref_freq_max,
                })
                return result

            # ------------------------------------------------------------------
            # 8. Ratio-based decision only
            # ------------------------------------------------------------------
            low_freq_power_ratio = low_freq_max_power / ref_median_power
            self.logger.info(f"Low-freq power ratio: {low_freq_power_ratio:.4f} (threshold={lowFreqPowerRatioThreshold})")

            low_freq_detected = low_freq_power_ratio >= lowFreqPowerRatioThreshold
            trend_status = "low_frequency_power" if low_freq_detected else "no_low_frequency_power"

            if low_freq_detected:
                self.logger.warning(
                    f"LOW-FREQUENCY POWER DETECTED: ratio={low_freq_power_ratio:.4f}"
                )
            else:
                self.logger.info(
                    f"No low-frequency power: ratio={low_freq_power_ratio:.4f}"
                )

            return {
                "lsLowFrequencyPowerDetected": low_freq_detected,
                "lsTrendStatus": trend_status,
                "lsLowFreqPowerRatio": float(low_freq_power_ratio),
                "lsLowFreqMaxPower": low_freq_max_power,
                "lsRefMedianPower": ref_median_power,
                "lsTrendBaselineDays": baseline_days,
                "lsTrendNumFinitePoints": num_finite,
                "lsTrendMethod": TREND_METHOD,
                "lsLowFreqPowerRatioThreshold": lowFreqPowerRatioThreshold,
                "lsRefFreqMin": float(ref_freq_min),
                "lsRefFreqMax": float(ref_freq_max),
                "lsLowFreqMin": float(low_freq_min),
                "lsLowFreqMax": float(low_freq_max),
            }

        except Exception as e:
            self.logger.exception(f"Trend detection failed for {rawFitsPath}: {e}")
            return {
                "lsLowFrequencyPowerDetected": False,
                "lsTrendStatus": "failed",
                "lsLowFreqPowerRatio": np.nan,
                "lsLowFreqMaxPower": np.nan,
                "lsRefMedianPower": np.nan,
                "lsTrendBaselineDays": np.nan,
                "lsTrendNumFinitePoints": 0,
                "lsTrendMethod": TREND_METHOD,
                "lsLowFreqPowerRatioThreshold": lowFreqPowerRatioThreshold,
                "lsRefFreqMin": np.nan,
                "lsRefFreqMax": np.nan,
                "lsLowFreqMin": np.nan,
                "lsLowFreqMax": np.nan,
            }

    def _process_single_row(self, idx: int, row: pd.Series, fits_folder: Path) -> Tuple[int, dict]:
        """
        Process a single row for trend detection.

        Args:
            idx: Row index
            row: Row data
            fits_folder: Path to FITS folder

        Returns:
            Tuple of (idx, results_dict) using the v2 field names
        """
        _failed: dict = {
            "lsLowFrequencyPowerDetected": False,
            "lsTrendStatus": "failed",
            "lsLowFreqPowerRatio": np.nan,
            "lsLowFreqMaxPower": np.nan,
            "lsRefMedianPower": np.nan,
            "lsTrendBaselineDays": np.nan,
            "lsTrendNumFinitePoints": 0,
            "lsTrendMethod": TREND_METHOD,
            "lsLowFreqPowerRatioThreshold": DEFAULT_LOW_FREQ_POWER_RATIO_THRESHOLD,
            "lsRefFreqMin": np.nan,
            "lsRefFreqMax": np.nan,
            "lsLowFreqMin": np.nan,
            "lsLowFreqMax": np.nan,
        }

        # Extract VSX ID to find raw FITS file
        vsx_id = row.get("VSXId")
        if not vsx_id:
            self.logger.debug(f"Row {idx}: No VSXId, skipping")
            return idx, _failed

        # Find raw FITS file matching this VSX ID
        sanitized_vsx = str(vsx_id).replace("/", "_").replace(" ", "_")
        raw_fits_files = list(fits_folder.glob(f"VSX_{sanitized_vsx}_*_raw.fits"))

        if not raw_fits_files:
            self.logger.debug(f"Row {idx} ({vsx_id}): No raw FITS file found")
            return idx, _failed

        raw_fits_path = str(raw_fits_files[0])
        self.logger.debug(f"Row {idx} ({vsx_id}): Found raw FITS file: {raw_fits_path}")

        # Run trend detection
        trend_result = self.detectLongTermTrendWithLS(raw_fits_path)
        return idx, trend_result

    def run(self, metaParquetFile: str, fitsFolder: str, maxWorkers: int = 16, outputPath: Optional[str] = None) -> str:
        """
        Process a metadata parquet file and run low-frequency trend detection on
        associated raw FITS files using a thread pool.

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

        # Ensure all result columns exist
        trend_columns = [
            "lsLowFrequencyPowerDetected",
            "lsTrendStatus",
            "lsLowFreqPowerRatio",
            "lsLowFreqMaxPower",
            "lsRefMedianPower",
            "lsTrendBaselineDays",
            "lsTrendNumFinitePoints",
            "lsTrendMethod",
            "lsLowFreqPowerRatioThreshold",
            "lsRefFreqMin",
            "lsRefFreqMax",
            "lsLowFreqMin",
            "lsLowFreqMax",
        ]
        for col in trend_columns:
            if col not in df.columns:
                df[col] = None

        fits_folder = Path(fitsFolder)
        if not fits_folder.exists() or not fits_folder.is_dir():
            self.logger.error(f"FITS folder not found or not a directory: {fitsFolder}")
            raise ValueError(f"Invalid FITS folder: {fitsFolder}")

        # Counters
        processed_count = 0
        low_freq_count = 0
        no_low_freq_count = 0
        insufficient_count = 0
        failed_count = 0
        total = len(df)

        self.logger.info(
            f"Starting low-frequency trend detection with {maxWorkers} worker threads "
            f"(ratio threshold={DEFAULT_LOW_FREQ_POWER_RATIO_THRESHOLD})"
        )

        with ThreadPoolExecutor(max_workers=maxWorkers) as executor:
            futures = {
                executor.submit(self._process_single_row, idx, df.loc[idx], fits_folder): idx
                for idx in df.index
            }

            for future in as_completed(futures):
                idx = futures[future]
                try:
                    result_idx, result_data = future.result()

                    for col, val in result_data.items():
                        df.at[result_idx, col] = val

                    processed_count += 1
                    status = result_data["lsTrendStatus"]
                    if status == "low_frequency_power":
                        low_freq_count += 1
                    elif status == "no_low_frequency_power":
                        no_low_freq_count += 1
                    elif status == "insufficient_data":
                        insufficient_count += 1
                    else:
                        failed_count += 1

                    if processed_count % 100 == 0:
                        self.logger.info(
                            f"Progress: {processed_count}/{total} processed "
                            f"| low_freq={low_freq_count} no_low_freq={no_low_freq_count} "
                            f"| insufficient={insufficient_count} failed={failed_count}"
                        )

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

        # Summary logging
        pct = lambda n: f"{100 * n / processed_count:.1f}%" if processed_count else "N/A"
        self.logger.info("=" * 60)
        self.logger.info("Low-Frequency Trend Detection Summary")
        self.logger.info(f"  Total rows in parquet : {total}")
        self.logger.info(f"  Processed             : {processed_count}")
        self.logger.info(f"  Low-freq detected     : {low_freq_count}  ({pct(low_freq_count)})")
        self.logger.info(f"  No low-freq           : {no_low_freq_count}  ({pct(no_low_freq_count)})")
        self.logger.info(f"  Insufficient data     : {insufficient_count}  ({pct(insufficient_count)})")
        self.logger.info(f"  Failed                : {failed_count}  ({pct(failed_count)})")
        self.logger.info("=" * 60)

        return str(output_path)

    def summarize_by_family(self, trendParquetFile: str, outputDir: Optional[str] = None) -> str:
        """
        Read a trend-result parquet file and compute per-family aggregation statistics.

        Saves a CSV named trend_detection_family_summary.csv alongside the parquet
        (or in outputDir if provided) and logs the table.

        Args:
            trendParquetFile: Path to the parquet file produced by run()
            outputDir: Directory for the CSV (default: same as trendParquetFile)

        Returns:
            Path to the saved CSV file
        """
        self.logger.info(f"Loading trend parquet for family summary: {trendParquetFile}")
        df = pd.read_parquet(trendParquetFile)

        if "Family" not in df.columns:
            self.logger.warning("Column 'Family' not found – using 'unknown' for all rows")
            df["Family"] = "unknown"

        total_rows = len(df)
        rows = []
        for family, grp in df.groupby("Family", sort=True):
            n_total = len(grp)
            detected = grp["lsLowFrequencyPowerDetected"].fillna(False).astype(bool)
            n_low_freq = int(detected.sum())

            ratios = pd.to_numeric(grp["lsLowFreqPowerRatio"], errors="coerce")
            status_col = grp["lsTrendStatus"] if "lsTrendStatus" in grp.columns else pd.Series(dtype=str)
            n_insufficient = int((status_col == "insufficient_data").sum())
            n_failed = int((status_col == "failed").sum())

            rows.append({
                "family": family,
                "total_count": n_total,
                "low_freq_count": n_low_freq,
                "low_freq_percent": round(100.0 * n_low_freq / n_total, 2) if n_total else 0.0,
                "median_low_freq_power_ratio": round(float(ratios.median()), 4) if ratios.notna().any() else np.nan,
                "p75_low_freq_power_ratio": round(float(ratios.quantile(0.75)), 4) if ratios.notna().any() else np.nan,
                "p90_low_freq_power_ratio": round(float(ratios.quantile(0.90)), 4) if ratios.notna().any() else np.nan,
                "insufficient_data_count": n_insufficient,
                "failed_count": n_failed,
            })

        summary_df = pd.DataFrame(rows)

        # Determine output path
        out_dir = Path(outputDir) if outputDir else Path(trendParquetFile).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "trend_detection_family_summary.csv"
        summary_df.to_csv(csv_path, index=False)
        self.logger.info(f"Family summary saved to {csv_path}")

        # Log summary table
        self.logger.info("=" * 80)
        self.logger.info(f"Family-Level Low-Frequency Trend Detection Summary  (total rows: {total_rows})")
        self.logger.info(f"{'Family':<30} {'Total':>7} {'LF':>7} {'LF%':>7} {'MedRatio':>10} {'P75':>10} {'P90':>10} {'Insuf':>6} {'Fail':>5}")
        self.logger.info("-" * 80)
        for r in rows:
            self.logger.info(
                f"{str(r['family']):<30} {r['total_count']:>7} {r['low_freq_count']:>7} "
                f"{r['low_freq_percent']:>6.1f}% {r['median_low_freq_power_ratio']:>10.4f} "
                f"{r['p75_low_freq_power_ratio']:>10.4f} {r['p90_low_freq_power_ratio']:>10.4f} "
                f"{r['insufficient_data_count']:>6} {r['failed_count']:>5}"
            )
        self.logger.info("=" * 80)

        return str(csv_path)
    
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run low-frequency trend detection on TESS light curves")
    parser.add_argument("--metaParquetFile", type=str, required=True, help="Path to metadata parquet file")
    parser.add_argument("--fitsFolder", type=str, required=True, help="Folder containing raw FITS files")
    parser.add_argument("--maxWorkers", type=int, default=16, help="Maximum number of worker threads")
    parser.add_argument("--outputPath", type=str, default=None, help="Output directory for trend parquet file (optional)")
    parser.add_argument(
        "--summarize",
        action="store_true",
        default=False,
        help="After processing, also generate family-level summary CSV",
    )
    parser.add_argument("--logFile", type=str, default=None, help="Path to log file (optional; logs always go to stdout)")
    args = parser.parse_args()

    _log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    if args.logFile:
        log_file_path = Path(args.logFile)
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        _handlers: list = [logging.FileHandler(log_file_path, mode="a", encoding="utf-8")]
    else:
        _handlers = [logging.StreamHandler()]
    logging.basicConfig(level=logging.INFO, format=_log_format, handlers=_handlers)

    if args.logFile:
        logging.getLogger("TrendDetector").info(f"Logging to file: {args.logFile}")

    trend_detector = TrendDetector()
    output_parquet = trend_detector.run(args.metaParquetFile, args.fitsFolder, args.maxWorkers, args.outputPath)

    if args.summarize:
        trend_detector.summarize_by_family(output_parquet, args.outputPath)
