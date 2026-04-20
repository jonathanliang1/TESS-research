#!/usr/bin/env python3
"""
fits_sanity_scan.py

Recursively scan a folder for FITS files and flag suspicious files using checks
similar to the TESS/TESSCut notebook logic.

Usage:
    python fits_sanity_scan.py /path/to/folder
    python fits_sanity_scan.py /path/to/folder --csv report.csv
    python fits_sanity_scan.py /path/to/folder --min-size-kb 100 --min-finite-rows 100
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from astropy.io import fits


TIME_COL_CANDIDATES = ["TIME", "BJD", "TIMECORR"]
FLUX_COL_CANDIDATES = ["FLUX", "SAP_FLUX", "PDCSAP_FLUX", "KSPSAP_FLUX", "RAW_FLUX"]
FITS_SUFFIXES = {".fits", ".fit", ".fts", ".fits.gz", ".fit.gz", ".fts.gz"}


@dataclass
class ScanResult:
    path: str
    size_bytes: int
    verdict: str
    suspicious: bool
    has_time_flux_table: bool
    has_image_hdu: bool
    hdu_count: int
    table_hdu_index: Optional[int]
    time_col: Optional[str]
    flux_col: Optional[str]
    total_rows: Optional[int]
    finite_rows: Optional[int]
    finite_fraction: Optional[float]
    time_span: Optional[float]
    flux_std: Optional[float]
    issues: str


def looks_like_fits(path: Path) -> bool:
    name = path.name.lower()
    return any(name.endswith(sfx) for sfx in FITS_SUFFIXES)


def find_time_flux_pair(hdul) -> Tuple[Optional[int], Optional[str], Optional[str]]:
    for i, hdu in enumerate(hdul):
        data = hdu.data
        names = getattr(data, "names", None)
        if names is None:
            continue
        names_upper = {n.upper(): n for n in names}
        time_col = next((names_upper[t] for t in TIME_COL_CANDIDATES if t in names_upper), None)
        flux_col = next((names_upper[f] for f in FLUX_COL_CANDIDATES if f in names_upper), None)
        if time_col and flux_col:
            return i, time_col, flux_col
    return None, None, None


def has_image_like_hdu(hdul) -> bool:
    for hdu in hdul:
        data = hdu.data
        if isinstance(data, np.ndarray) and data.ndim >= 2 and data.size > 0:
            return True
    return False


def safe_float(value) -> Optional[float]:
    try:
        val = float(value)
        if np.isnan(val):
            return None
        return val
    except Exception:
        return None


def analyze_table(hdul, hdu_idx: int, time_col: str, flux_col: str):
    table = hdul[hdu_idx].data
    total_rows = len(table)

    try:
        time = np.asarray(table[time_col], dtype=float)
        flux = np.asarray(table[flux_col], dtype=float)
    except Exception:
        return total_rows, None, None, None

    finite_mask = np.isfinite(time) & np.isfinite(flux)
    finite_rows = int(finite_mask.sum())
    finite_fraction = finite_rows / total_rows if total_rows else None

    if finite_rows == 0:
        return total_rows, finite_rows, finite_fraction, None, None

    time_f = time[finite_mask]
    flux_f = flux[finite_mask]

    time_span = safe_float(np.nanmax(time_f) - np.nanmin(time_f))
    flux_std = safe_float(np.nanstd(flux_f))
    return total_rows, finite_rows, finite_fraction, time_span, flux_std


def classify_and_flag(
    path: Path,
    min_size_kb: int,
    min_finite_rows: int,
    min_finite_fraction: float,
) -> ScanResult:
    issues: List[str] = []
    size_bytes = path.stat().st_size

    if size_bytes < min_size_kb * 1024:
        issues.append(f"small_file<{min_size_kb}KB")

    try:
        with fits.open(path) as hdul:
            hdu_count = len(hdul)
            image_hdu = has_image_like_hdu(hdul)
            hdu_idx, time_col, flux_col = find_time_flux_pair(hdul)

            has_time_flux = hdu_idx is not None

            if has_time_flux:
                total_rows, finite_rows, finite_fraction, time_span, flux_std = analyze_table(
                    hdul, hdu_idx, time_col, flux_col
                )
            else:
                total_rows = finite_rows = None
                finite_fraction = time_span = flux_std = None

            if has_time_flux:
                verdict = "LIKELY_FINAL_LIGHT_CURVE_FITS"
            elif image_hdu:
                verdict = "LIKELY_IMAGE_OR_TESSCUT_PRODUCT"
            else:
                verdict = "UNCLEAR_OR_METADATA_ONLY"

            if not has_time_flux:
                issues.append("no_recognizable_time_flux_table")

            if has_time_flux and total_rows is not None and total_rows == 0:
                issues.append("empty_time_flux_table")

            if has_time_flux and finite_rows is not None and finite_rows < min_finite_rows:
                issues.append(f"too_few_finite_rows<{min_finite_rows}")

            if has_time_flux and finite_fraction is not None and finite_fraction < min_finite_fraction:
                issues.append(f"low_finite_fraction<{min_finite_fraction:.2f}")

            if has_time_flux and time_span is not None and time_span <= 0:
                issues.append("non_positive_time_span")

            if has_time_flux and flux_std is not None and flux_std == 0:
                issues.append("zero_flux_scatter")

            suspicious = len(issues) > 0

            return ScanResult(
                path=str(path),
                size_bytes=size_bytes,
                verdict=verdict,
                suspicious=suspicious,
                has_time_flux_table=has_time_flux,
                has_image_hdu=image_hdu,
                hdu_count=hdu_count,
                table_hdu_index=hdu_idx,
                time_col=time_col,
                flux_col=flux_col,
                total_rows=total_rows,
                finite_rows=finite_rows,
                finite_fraction=finite_fraction,
                time_span=time_span,
                flux_std=flux_std,
                issues=";".join(issues),
            )

    except Exception as e:
        return ScanResult(
            path=str(path),
            size_bytes=size_bytes,
            verdict="FAILED_TO_OPEN",
            suspicious=True,
            has_time_flux_table=False,
            has_image_hdu=False,
            hdu_count=0,
            table_hdu_index=None,
            time_col=None,
            flux_col=None,
            total_rows=None,
            finite_rows=None,
            finite_fraction=None,
            time_span=None,
            flux_std=None,
            issues=f"open_error:{type(e).__name__}:{e}",
        )


def scan_folder(folder: Path, min_size_kb: int, min_finite_rows: int, min_finite_fraction: float) -> List[ScanResult]:
    results: List[ScanResult] = []
    for path in folder.rglob("*"):
        if path.is_file() and looks_like_fits(path):
            results.append(classify_and_flag(path, min_size_kb, min_finite_rows, min_finite_fraction))
    return results


def write_csv(results: List[ScanResult], csv_path: Path) -> None:
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()) if results else list(ScanResult.__annotations__.keys()))
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))


def print_summary(results: List[ScanResult]) -> None:
    total = len(results)
    suspicious = sum(r.suspicious for r in results)
    failed = sum(r.verdict == "FAILED_TO_OPEN" for r in results)
    lightcurve = sum(r.verdict == "LIKELY_FINAL_LIGHT_CURVE_FITS" for r in results)
    image_like = sum(r.verdict == "LIKELY_IMAGE_OR_TESSCUT_PRODUCT" for r in results)
    unclear = sum(r.verdict == "UNCLEAR_OR_METADATA_ONLY" for r in results)

    print(f"Total FITS files scanned: {total}")
    print(f"Suspicious files: {suspicious}")
    print(f"Failed to open: {failed}")
    print(f"Likely final light curve FITS: {lightcurve}")
    print(f"Likely image/TESSCut products: {image_like}")
    print(f"Unclear/metadata only: {unclear}")

    if suspicious:
        print("\nSuspicious files:")
        for r in results:
            if r.suspicious:
                print(f"- {r.path}")
                print(f"  verdict={r.verdict}, size={r.size_bytes} bytes, issues={r.issues}")


def main():
    parser = argparse.ArgumentParser(description="Scan a folder recursively and flag suspicious FITS files.")
    parser.add_argument("folder", help="Folder to scan recursively")
    parser.add_argument("--csv", dest="csv_path", default=None, help="Optional CSV output path")
    parser.add_argument("--min-size-kb", type=int, default=50, help="Flag files smaller than this size in KB")
    parser.add_argument("--min-finite-rows", type=int, default=100, help="Flag files with fewer finite time/flux rows than this")
    parser.add_argument("--min-finite-fraction", type=float, default=0.5, help="Flag files with lower finite-row fraction than this")
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.exists() or not folder.is_dir():
        raise SystemExit(f"Folder does not exist or is not a directory: {folder}")

    results = scan_folder(
        folder=folder,
        min_size_kb=args.min_size_kb,
        min_finite_rows=args.min_finite_rows,
        min_finite_fraction=args.min_finite_fraction,
    )
    print_summary(results)

    if args.csv_path:
        csv_path = Path(args.csv_path)
        write_csv(results, csv_path)
        print(f"\nCSV report written to: {csv_path}")


if __name__ == "__main__":
    main()
