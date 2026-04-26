"""
TrendDetector.py

Detect long-term trends in TESS light curves using Lomb-Scargle periodogram
analysis on low frequencies.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
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
DEFAULT_GAP_THRESHOLD_DAYS: float = 1.0
DEFAULT_MIN_SEGMENT_POINTS: int = 100
DEFAULT_MIN_SEGMENT_DURATION_DAYS: float = 5.0
DEFAULT_SEGMENT_FRACTION_THRESHOLD: float = 0.5
DEFAULT_LOW_FREQ_N_SAMPLES: int = 500
DEFAULT_REF_FREQ_N_SAMPLES: int = 2000
DEFAULT_REF_FREQ_MAX: float = 24.0          # cycles/day  (1 cycle per hour)
DEFAULT_LOW_FREQ_PERIOD_MIN_FACTOR: float = 0.5   # period = factor * baseline
DEFAULT_LOW_FREQ_PERIOD_MAX_FACTOR: float = 2.0   # period = factor * baseline
DEFAULT_REF_FREQ_MIN_FACTOR: float = 1.5          # refFreqMin = factor * lowFreqMax
DEFAULT_DEBUG_SAMPLE_COUNT: int = 5
TREND_METHOD: str = "ls_segmented_low_frequency_power_v1"
# ---------------------------------------------------------------------------


class TrendDetector:
    """
    Detect long-term trends in light curves using Lomb-Scargle periodogram.
    """

    def __init__(self):
        self.logger = logging.getLogger("TrendDetector")

    def _empty_segmented_result(
        self,
        status: str,
        lowFreqPowerRatioThreshold: float,
        numFinitePoints: int = 0,
        trendBaselineDays: float = np.nan,
    ) -> Dict:
        return {
            "lsLowFrequencyPowerDetected": False,
            "lsTrendStatus": status,
            "lsValidSegmentCount": 0,
            "lsSegmentTrendCount": 0,
            "lsFractionSegmentsWithLowFreq": np.nan,
            "lsMaxSegmentLowFreqPowerRatio": np.nan,
            "lsMedianSegmentLowFreqPowerRatio": np.nan,
            "lsTrendMethod": TREND_METHOD,
            "lsTrendNumFinitePoints": numFinitePoints,
            "lsTrendBaselineDays": trendBaselineDays,
            "lsLowFreqPowerRatioThreshold": lowFreqPowerRatioThreshold,
            # Optional compatibility diagnostics derived from the strongest segment.
            "lsLowFreqPowerRatio": np.nan,
            "lsLowFreqMaxPower": np.nan,
            "lsRefMedianPower": np.nan,
            "lsLowFreqMin": np.nan,
            "lsLowFreqMax": np.nan,
            "lsRefFreqMin": np.nan,
            "lsRefFreqMax": np.nan,
        }

    def _split_into_segments(
        self,
        time: np.ndarray,
        flux: np.ndarray,
        gapThresholdDays: float,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        if len(time) == 0:
            return []

        dt = np.diff(time)
        segment_breaks = np.where(dt > gapThresholdDays)[0] + 1
        segment_starts = np.concatenate(([0], segment_breaks))
        segment_ends = np.concatenate((segment_breaks, [len(time)]))

        return [
            (time[start:end], flux[start:end])
            for start, end in zip(segment_starts, segment_ends)
            if end > start
        ]

    def _analyze_segment(
        self,
        segmentTime: np.ndarray,
        segmentFlux: np.ndarray,
        lowFreqPowerRatioThreshold: float,
        minSegmentPoints: int,
        minSegmentDuration: float,
    ) -> Optional[Dict]:
        num_points = len(segmentTime)
        if num_points < minSegmentPoints:
            return None

        segment_duration = float(segmentTime[-1] - segmentTime[0])
        if segment_duration < minSegmentDuration:
            return None

        low_freq_min = 1.0 / (DEFAULT_LOW_FREQ_PERIOD_MAX_FACTOR * segment_duration)
        low_freq_max = 1.0 / (DEFAULT_LOW_FREQ_PERIOD_MIN_FACTOR * segment_duration)
        ref_freq_min = DEFAULT_REF_FREQ_MIN_FACTOR * low_freq_max
        ref_freq_max = DEFAULT_REF_FREQ_MAX

        if not np.isfinite(segment_duration) or segment_duration <= 0:
            return None
        if not np.isfinite(ref_freq_min) or ref_freq_min >= ref_freq_max:
            return None

        try:
            time_zeroed = segmentTime - segmentTime[0]

            low_freq_range = np.linspace(low_freq_min, low_freq_max, DEFAULT_LOW_FREQ_N_SAMPLES)
            low_ls = LombScargle(time_zeroed, segmentFlux, normalization="psd")
            low_power = low_ls.power(low_freq_range)
            low_freq_max_power = float(np.nanmax(low_power))

            ref_freq_range = np.linspace(ref_freq_min, ref_freq_max, DEFAULT_REF_FREQ_N_SAMPLES)
            ref_ls = LombScargle(time_zeroed, segmentFlux, normalization="psd")
            ref_power = ref_ls.power(ref_freq_range)
            ref_median_power = float(np.nanmedian(ref_power))
        except Exception:
            return None

        if not np.isfinite(low_freq_max_power):
            return None
        if not np.isfinite(ref_median_power) or ref_median_power <= 0:
            return None

        low_freq_power_ratio = low_freq_max_power / ref_median_power
        if not np.isfinite(low_freq_power_ratio):
            return None

        return {
            "segmentDuration": segment_duration,
            "segmentNumPoints": num_points,
            "segmentLowFreqMin": float(low_freq_min),
            "segmentLowFreqMax": float(low_freq_max),
            "segmentRefFreqMin": float(ref_freq_min),
            "segmentRefFreqMax": float(ref_freq_max),
            "segmentLowFreqMaxPower": low_freq_max_power,
            "segmentRefMedianPower": ref_median_power,
            "segmentLowFreqPowerRatio": float(low_freq_power_ratio),
            "segmentHasLowFreqPower": low_freq_power_ratio >= lowFreqPowerRatioThreshold,
        }

    def _load_time_flux_from_raw_fits(self, rawFitsPath: str) -> Tuple[np.ndarray, np.ndarray]:
        with fits.open(rawFitsPath, memmap=False) as hdul:
            if len(hdul) <= 1 or hdul[1].data is None:
                raise ValueError("Missing table data in HDU[1]")

            table = hdul[1].data
            names = set(getattr(table, "names", []) or [])
            if "TIME" not in names or "FLUX" not in names:
                raise ValueError("HDU[1] must contain TIME and FLUX columns")

            time = np.asarray(table["TIME"], dtype=float)
            flux = np.asarray(table["FLUX"], dtype=float)

        return time, flux

    def detectLongTermTrendWithLS(
        self,
        rawFitsPath: str,
        minFinitePoints: int = DEFAULT_MIN_FINITE_POINTS,
        lowFreqPowerRatioThreshold: float = DEFAULT_LOW_FREQ_POWER_RATIO_THRESHOLD,
        gapThresholdDays: float = DEFAULT_GAP_THRESHOLD_DAYS,
        minSegmentPoints: int = DEFAULT_MIN_SEGMENT_POINTS,
        minSegmentDuration: float = DEFAULT_MIN_SEGMENT_DURATION_DAYS,
    ) -> Dict:
        """
        Detect low-frequency power excess within contiguous observing segments.

        Args:
            rawFitsPath: Path to the pipeline-generated raw FITS file
            minFinitePoints: Minimum number of finite TIME/FLUX points required
            lowFreqPowerRatioThreshold: Ratio threshold for low-frequency flag
            gapThresholdDays: Start a new segment when time gaps exceed this value
            minSegmentPoints: Minimum points required for a valid segment
            minSegmentDuration: Minimum duration in days required for a valid segment

        Returns:
            Segment-aware trend metadata for one stitched light curve.
        """
        try:
            self.logger.info(f"Loading raw FITS file: {rawFitsPath}")
            time, flux = self._load_time_flux_from_raw_fits(rawFitsPath)

            self.logger.debug(f"Initial time/flux size: {len(time)}")
            finite_mask = np.isfinite(time) & np.isfinite(flux)
            time_finite = time[finite_mask]
            flux_finite = flux[finite_mask]

            if len(time_finite) == 0:
                return self._empty_segmented_result(
                    status="insufficient_data",
                    lowFreqPowerRatioThreshold=lowFreqPowerRatioThreshold,
                    numFinitePoints=0,
                )

            sort_idx = np.argsort(time_finite)
            time_finite = time_finite[sort_idx]
            flux_finite = flux_finite[sort_idx]
            num_finite = len(time_finite)

            full_baseline_days = float(time_finite[-1] - time_finite[0]) if num_finite > 1 else np.nan
            self.logger.info(f"Finite time/flux points: {num_finite}")

            if num_finite < minFinitePoints:
                return self._empty_segmented_result(
                    status="insufficient_data",
                    lowFreqPowerRatioThreshold=lowFreqPowerRatioThreshold,
                    numFinitePoints=num_finite,
                    trendBaselineDays=full_baseline_days,
                )

            segments = self._split_into_segments(time_finite, flux_finite, gapThresholdDays)
            self.logger.info(
                f"Split into {len(segments)} contiguous segments using gapThresholdDays={gapThresholdDays}"
            )

            valid_segments: List[Dict] = []
            for segment_time, segment_flux in segments:
                segment_result = self._analyze_segment(
                    segmentTime=segment_time,
                    segmentFlux=segment_flux,
                    lowFreqPowerRatioThreshold=lowFreqPowerRatioThreshold,
                    minSegmentPoints=minSegmentPoints,
                    minSegmentDuration=minSegmentDuration,
                )
                if segment_result is not None:
                    valid_segments.append(segment_result)

            valid_segment_count = len(valid_segments)
            self.logger.info(f"Valid segments retained: {valid_segment_count}")

            if valid_segment_count == 0:
                return self._empty_segmented_result(
                    status="insufficient_data",
                    lowFreqPowerRatioThreshold=lowFreqPowerRatioThreshold,
                    numFinitePoints=num_finite,
                    trendBaselineDays=full_baseline_days,
                )

            segment_trend_count = int(sum(segment["segmentHasLowFreqPower"] for segment in valid_segments))
            segment_ratios = [segment["segmentLowFreqPowerRatio"] for segment in valid_segments]
            max_segment_ratio = float(np.max(segment_ratios))
            median_segment_ratio = float(np.median(segment_ratios))
            fraction_segments_with_low_freq = segment_trend_count / valid_segment_count

            strongest_segment = max(valid_segments, key=lambda segment: segment["segmentLowFreqPowerRatio"])
            low_freq_detected = (
                max_segment_ratio >= lowFreqPowerRatioThreshold
                or fraction_segments_with_low_freq >= DEFAULT_SEGMENT_FRACTION_THRESHOLD
            )
            trend_status = "low_frequency_power" if low_freq_detected else "no_low_frequency_power"

            self.logger.info(
                "Segment summary: valid=%d trend=%d fraction=%.3f max_ratio=%.3f median_ratio=%.3f",
                valid_segment_count,
                segment_trend_count,
                fraction_segments_with_low_freq,
                max_segment_ratio,
                median_segment_ratio,
            )

            return {
                "lsLowFrequencyPowerDetected": low_freq_detected,
                "lsTrendStatus": trend_status,
                "lsValidSegmentCount": valid_segment_count,
                "lsSegmentTrendCount": segment_trend_count,
                "lsFractionSegmentsWithLowFreq": float(fraction_segments_with_low_freq),
                "lsMaxSegmentLowFreqPowerRatio": max_segment_ratio,
                "lsMedianSegmentLowFreqPowerRatio": median_segment_ratio,
                "lsTrendMethod": TREND_METHOD,
                "lsTrendNumFinitePoints": num_finite,
                "lsTrendBaselineDays": float(strongest_segment["segmentDuration"]),
                "lsLowFreqPowerRatioThreshold": lowFreqPowerRatioThreshold,
                # Optional compatibility diagnostics from the strongest segment.
                "lsLowFreqPowerRatio": float(strongest_segment["segmentLowFreqPowerRatio"]),
                "lsLowFreqMaxPower": float(strongest_segment["segmentLowFreqMaxPower"]),
                "lsRefMedianPower": float(strongest_segment["segmentRefMedianPower"]),
                "lsLowFreqMin": float(strongest_segment["segmentLowFreqMin"]),
                "lsLowFreqMax": float(strongest_segment["segmentLowFreqMax"]),
                "lsRefFreqMin": float(strongest_segment["segmentRefFreqMin"]),
                "lsRefFreqMax": float(strongest_segment["segmentRefFreqMax"]),
            }

        except Exception as e:
            self.logger.exception(f"Trend detection failed for {rawFitsPath}: {e}")
            return self._empty_segmented_result(
                status="failed",
                lowFreqPowerRatioThreshold=lowFreqPowerRatioThreshold,
            )

    def _process_single_row(
        self,
        idx: int,
        row: pd.Series,
        fits_folder: Path,
        gapThresholdDays: float,
        minSegmentPoints: int,
        minSegmentDuration: float,
        lowFreqPowerRatioThreshold: float,
    ) -> Tuple[int, dict]:
        """
        Process a single row for trend detection.

        Args:
            idx: Row index
            row: Row data
            fits_folder: Path to FITS folder

        Returns:
            Tuple of (idx, results_dict) using the v2 field names
        """
        _failed: dict = self._empty_segmented_result(
            status="failed",
            lowFreqPowerRatioThreshold=lowFreqPowerRatioThreshold,
        )

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
        trend_result = self.detectLongTermTrendWithLS(
            rawFitsPath=raw_fits_path,
            lowFreqPowerRatioThreshold=lowFreqPowerRatioThreshold,
            gapThresholdDays=gapThresholdDays,
            minSegmentPoints=minSegmentPoints,
            minSegmentDuration=minSegmentDuration,
        )
        return idx, trend_result

    def run(
        self,
        metaParquetFile: str,
        fitsFolder: str,
        maxWorkers: int = 16,
        outputPath: Optional[str] = None,
        gapThresholdDays: float = DEFAULT_GAP_THRESHOLD_DAYS,
        minSegmentPoints: int = DEFAULT_MIN_SEGMENT_POINTS,
        minSegmentDuration: float = DEFAULT_MIN_SEGMENT_DURATION_DAYS,
        lowFreqPowerRatioThreshold: float = DEFAULT_LOW_FREQ_POWER_RATIO_THRESHOLD,
    ) -> str:
        """
        Process a metadata parquet file and run low-frequency trend detection on
        associated raw FITS files using a thread pool.

        Args:
            metaParquetFile: Path to the metadata parquet file
            fitsFolder: Folder containing raw FITS light curve files
            maxWorkers: Maximum number of worker threads (default: 16)
            outputPath: Output directory path (optional; default: same folder as input)
            gapThresholdDays: Split stitched light curves into segments on gaps > this many days
            minSegmentPoints: Minimum points required in a valid segment
            minSegmentDuration: Minimum duration in days required in a valid segment
            lowFreqPowerRatioThreshold: Ratio threshold applied to each segment

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
            "lsValidSegmentCount",
            "lsSegmentTrendCount",
            "lsFractionSegmentsWithLowFreq",
            "lsMaxSegmentLowFreqPowerRatio",
            "lsMedianSegmentLowFreqPowerRatio",
            # Optional compatibility diagnostics
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
        stars_with_valid_segments = 0
        segment_count_distribution: Dict[int, int] = {}
        total = len(df)

        self.logger.info(
            f"Starting low-frequency trend detection with {maxWorkers} worker threads "
            f"(ratio threshold={lowFreqPowerRatioThreshold}, gap={gapThresholdDays}d, "
            f"minSegmentPoints={minSegmentPoints}, minSegmentDuration={minSegmentDuration}d)"
        )

        with ThreadPoolExecutor(max_workers=maxWorkers) as executor:
            futures = {
                executor.submit(
                    self._process_single_row,
                    idx,
                    df.loc[idx],
                    fits_folder,
                    gapThresholdDays,
                    minSegmentPoints,
                    minSegmentDuration,
                    lowFreqPowerRatioThreshold,
                ): idx
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
                    valid_segment_count = int(result_data.get("lsValidSegmentCount", 0) or 0)
                    if valid_segment_count > 0:
                        stars_with_valid_segments += 1
                    segment_count_distribution[valid_segment_count] = (
                        segment_count_distribution.get(valid_segment_count, 0) + 1
                    )

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
                            f"| valid_segments>0={stars_with_valid_segments} "
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
        self.logger.info(f"  >=1 valid segment     : {stars_with_valid_segments}  ({pct(stars_with_valid_segments)})")
        self.logger.info(f"  Low-freq detected     : {low_freq_count}  ({pct(low_freq_count)})")
        self.logger.info(f"  No low-freq           : {no_low_freq_count}  ({pct(no_low_freq_count)})")
        self.logger.info(f"  Insufficient data     : {insufficient_count}  ({pct(insufficient_count)})")
        self.logger.info(f"  Failed                : {failed_count}  ({pct(failed_count)})")
        self.logger.info("  Segment count distribution:")
        for segment_count in sorted(segment_count_distribution):
            self.logger.info(
                f"    segments={segment_count}: {segment_count_distribution[segment_count]} stars"
            )
        self.logger.info("=" * 60)

        self.summarize_outputs(output_path, output_dir)

        return str(output_path)

    def _build_summary_df(self, df: pd.DataFrame, groupColumn: str) -> pd.DataFrame:
        working_df = df.copy()
        if groupColumn not in working_df.columns:
            self.logger.warning("Column '%s' not found – using 'unknown'", groupColumn)
            working_df[groupColumn] = "unknown"

        working_df[groupColumn] = working_df[groupColumn].fillna("unknown")
        rows = []
        for group_value, grp in working_df.groupby(groupColumn, sort=True):
            total_count = len(grp)
            detected = grp["lsLowFrequencyPowerDetected"].fillna(False).astype(bool)
            low_freq_count = int(detected.sum())
            rows.append(
                {
                    groupColumn: group_value,
                    "total": total_count,
                    "low_freq_count": low_freq_count,
                    "percent": round(100.0 * low_freq_count / total_count, 2) if total_count else 0.0,
                }
            )

        return pd.DataFrame(rows)

    def summarize_outputs(self, trendParquetFile: str, outputDir: Optional[str] = None) -> Dict[str, str]:
        """
        Read a trend-result parquet file and compute per-family and per-source summaries.

        Saves CSV files alongside the parquet (or in outputDir if provided).

        Args:
            trendParquetFile: Path to the parquet file produced by run()
            outputDir: Directory for the CSV files (default: same as trendParquetFile)

        Returns:
            Dictionary of output CSV paths
        """
        self.logger.info(f"Loading trend parquet for summaries: {trendParquetFile}")
        df = pd.read_parquet(trendParquetFile)
        out_dir = Path(outputDir) if outputDir else Path(trendParquetFile).parent
        out_dir.mkdir(parents=True, exist_ok=True)

        family_summary_df = self._build_summary_df(df, "family")
        source_summary_df = self._build_summary_df(df, "provenance")

        family_csv_path = out_dir / "trend_detection_family_summary.csv"
        source_csv_path = out_dir / "trend_detection_source_summary.csv"
        family_summary_df.to_csv(family_csv_path, index=False)
        source_summary_df.to_csv(source_csv_path, index=False)

        self.logger.info(f"Family summary saved to {family_csv_path}")
        self.logger.info(f"Source summary saved to {source_csv_path}")

        self.logger.info("=" * 80)
        self.logger.info("Family-Level Low-Frequency Trend Detection Summary")
        self.logger.info(f"{'Family':<30} {'Total':>7} {'LF':>7} {'LF%':>7}")
        self.logger.info("-" * 80)
        for _, row in family_summary_df.iterrows():
            self.logger.info(
                f"{str(row['family']):<30} {int(row['total']):>7} {int(row['low_freq_count']):>7} "
                f"{float(row['percent']):>6.1f}%"
            )
        self.logger.info("-" * 80)
        self.logger.info("Source-Level Low-Frequency Trend Detection Summary")
        self.logger.info(f"{'Source':<30} {'Total':>7} {'LF':>7} {'LF%':>7}")
        self.logger.info("-" * 80)
        for _, row in source_summary_df.iterrows():
            self.logger.info(
                f"{str(row['provenance']):<30} {int(row['total']):>7} {int(row['low_freq_count']):>7} "
                f"{float(row['percent']):>6.1f}%"
            )
        self.logger.info("=" * 80)

        return {
            "family": str(family_csv_path),
            "source": str(source_csv_path),
        }

    def summarize_by_family(self, trendParquetFile: str, outputDir: Optional[str] = None) -> str:
        return self.summarize_outputs(trendParquetFile, outputDir)["family"]

    def debug_sample_segments(
        self,
        fitsFolder: str,
        sampleCount: int = DEFAULT_DEBUG_SAMPLE_COUNT,
        gapThresholdDays: float = DEFAULT_GAP_THRESHOLD_DAYS,
        minSegmentPoints: int = DEFAULT_MIN_SEGMENT_POINTS,
        minSegmentDuration: float = DEFAULT_MIN_SEGMENT_DURATION_DAYS,
        lowFreqPowerRatioThreshold: float = DEFAULT_LOW_FREQ_POWER_RATIO_THRESHOLD,
    ) -> None:
        """
        Run segmented LS on a small sample of raw FITS files and log segment counts.
        """
        fits_folder = Path(fitsFolder)
        sample_files = sorted(fits_folder.glob("*_raw.fits"))[:sampleCount]
        self.logger.info(
            f"Debugging segmented trend detection on {len(sample_files)} sample FITS files"
        )

        for sample_file in sample_files:
            try:
                time, flux = self._load_time_flux_from_raw_fits(str(sample_file))
                finite_mask = np.isfinite(time) & np.isfinite(flux)
                time = np.asarray(time[finite_mask], dtype=float)
                flux = np.asarray(flux[finite_mask], dtype=float)
                if len(time) == 0:
                    self.logger.info(f"{sample_file.name}: no finite rows")
                    continue

                sort_idx = np.argsort(time)
                time = time[sort_idx]
                flux = flux[sort_idx]
                segments = self._split_into_segments(time, flux, gapThresholdDays)
                segment_ratios = []
                valid_segment_count = 0
                for segment_time, segment_flux in segments:
                    segment_result = self._analyze_segment(
                        segmentTime=segment_time,
                        segmentFlux=segment_flux,
                        lowFreqPowerRatioThreshold=lowFreqPowerRatioThreshold,
                        minSegmentPoints=minSegmentPoints,
                        minSegmentDuration=minSegmentDuration,
                    )
                    if segment_result is not None:
                        valid_segment_count += 1
                        segment_ratios.append(segment_result["segmentLowFreqPowerRatio"])

                self.logger.info(
                    "%s: total_segments=%d valid_segments=%d ratios=%s",
                    sample_file.name,
                    len(segments),
                    valid_segment_count,
                    [round(ratio, 3) for ratio in segment_ratios],
                )
            except Exception as e:
                self.logger.exception(f"Debug sample failed for {sample_file}: {e}")
    
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run low-frequency trend detection on TESS light curves")
    parser.add_argument("--metaParquetFile", type=str, required=True, help="Path to metadata parquet file")
    parser.add_argument("--fitsFolder", type=str, required=True, help="Folder containing raw FITS files")
    parser.add_argument("--maxWorkers", type=int, default=16, help="Maximum number of worker threads")
    parser.add_argument("--outputPath", type=str, default=None, help="Output directory for trend parquet file (optional)")
    parser.add_argument("--gapThresholdDays", type=float, default=DEFAULT_GAP_THRESHOLD_DAYS, help="Split segments when time gaps exceed this many days")
    parser.add_argument("--minSegmentPoints", type=int, default=DEFAULT_MIN_SEGMENT_POINTS, help="Minimum points required in a valid segment")
    parser.add_argument("--minSegmentDuration", type=float, default=DEFAULT_MIN_SEGMENT_DURATION_DAYS, help="Minimum segment duration in days")
    parser.add_argument("--lowFreqPowerRatioThreshold", type=float, default=DEFAULT_LOW_FREQ_POWER_RATIO_THRESHOLD, help="Low-frequency power ratio threshold")
    parser.add_argument(
        "--summarize",
        action="store_true",
        default=False,
        help="Retained for compatibility; summaries are generated automatically",
    )
    parser.add_argument("--debugSampleCount", type=int, default=0, help="Run segmented-LS debug sampling on the first N raw FITS files")
    parser.add_argument("--logFile", type=str, default=None, help="Path to log file (optional; when provided logs are written to file only)")
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
    if args.debugSampleCount > 0:
        trend_detector.debug_sample_segments(
            fitsFolder=args.fitsFolder,
            sampleCount=args.debugSampleCount,
            gapThresholdDays=args.gapThresholdDays,
            minSegmentPoints=args.minSegmentPoints,
            minSegmentDuration=args.minSegmentDuration,
            lowFreqPowerRatioThreshold=args.lowFreqPowerRatioThreshold,
        )

    output_parquet = trend_detector.run(
        args.metaParquetFile,
        args.fitsFolder,
        args.maxWorkers,
        args.outputPath,
        args.gapThresholdDays,
        args.minSegmentPoints,
        args.minSegmentDuration,
        args.lowFreqPowerRatioThreshold,
    )

    if args.summarize:
        logging.getLogger("TrendDetector").info(
            "Summaries are generated automatically; --summarize is kept for compatibility."
        )
