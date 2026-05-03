#!/usr/bin/env python3
"""
FeatureExtractor.py

Class-based feature extraction for VSX -> TESS variable-star classification.

Designed for the repository layout used by:
    data_pipeline/TessDataDownloader.py

Key design choices
------------------
1. Uses `lightCurvePath` as the canonical standardized FITS input.
2. Does not perform dataset-level scaling.
3. Extracts only per-star features, so it is safe to run before train/test split.
4. Supports parallel processing with configurable thread pool size.
5. Adds percentile-based tail-asymmetry features.

Example
-------
python FeatureExtractor.py \
    --input TESSCache/TESSAugmented_QC_trend.parquet \
    --output TESSCache/TESS_features.parquet \
    --worker-count 16
"""

from __future__ import annotations

import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

import lightkurve as lk
from astropy.io import fits
from astropy.timeseries import LombScargle
from scipy.stats import skew, kurtosis


class FeatureExtractor:
    """Extracts ML-ready features from standardized TESS light curve FITS files."""

    QualityScoreMap = {
        "clean": 0,
        "acceptable": 1,
        "caution": 2,
        "poor": 3,
        "missing": 4,
    }

    ProvenanceScoreMap = {
        "SPOC": 0,
        "QLP": 1,
        "TESSCut": 2,
    }

    def __init__(
        self,
        MinPeriodDays: float = 0.05,
        MaxPeriodDays: float = 100.0,
        SamplesPerPeak: int = 10,
        MinCadences: int = 100,
        PhaseBinCount: int = 20,
        WorkerCount: int = 16,
    ):
        self.MinPeriodDays = MinPeriodDays
        self.MaxPeriodDays = MaxPeriodDays
        self.SamplesPerPeak = SamplesPerPeak
        self.MinCadences = MinCadences
        self.PhaseBinCount = PhaseBinCount
        self.WorkerCount = WorkerCount
        self.Eps = 1e-12
        self.Logger = logging.getLogger("FeatureExtractor")

    def ExtractFeatures(self, MetadataDf: pd.DataFrame, MetadataPath: Path | str) -> pd.DataFrame:
        MetadataPath = Path(MetadataPath)

        if self.WorkerCount <= 1:
            return self._ExtractFeaturesSerial(MetadataDf, MetadataPath)

        return self._ExtractFeaturesParallel(MetadataDf, MetadataPath)

    def ExtractFeaturesFromParquet(self, InputPath: Path | str, OutputPath: Path | str) -> pd.DataFrame:
        InputPath = Path(InputPath)
        OutputPath = Path(OutputPath)

        MetadataDf = pd.read_parquet(InputPath)
        FeatureDf = self.ExtractFeatures(MetadataDf, InputPath)

        OutputPath.parent.mkdir(parents=True, exist_ok=True)
        FeatureDf.to_parquet(OutputPath, index=False)

        self.Logger.info(
            "Saved %s rows and %s columns to %s",
            len(FeatureDf),
            len(FeatureDf.columns),
            OutputPath,
        )

        if "FeatureStatus" in FeatureDf.columns:
            self.Logger.info("FeatureStatus counts:\n%s", FeatureDf["FeatureStatus"].value_counts(dropna=False))

        return FeatureDf

    def _ExtractFeaturesSerial(self, MetadataDf: pd.DataFrame, MetadataPath: Path) -> pd.DataFrame:
        ResultRows = []

        for Position, (_, Row) in enumerate(MetadataDf.iterrows()):
            if Position % 100 == 0:
                self.Logger.info("Processing %s/%s", Position, len(MetadataDf))

            FeatureRow = self._ExtractFeaturesForRow(Row, MetadataPath)
            FeatureRow["_InputOrder"] = Position
            ResultRows.append(FeatureRow)

        return self._FinalizeFeatureDf(pd.DataFrame(ResultRows))

    def _ExtractFeaturesParallel(self, MetadataDf: pd.DataFrame, MetadataPath: Path) -> pd.DataFrame:
        ResultRows = []
        TotalRows = len(MetadataDf)

        self.Logger.info("Starting parallel feature extraction with WorkerCount=%s", self.WorkerCount)

        with ThreadPoolExecutor(max_workers=self.WorkerCount) as Executor:
            FutureMap = {}

            for Position, (_, Row) in enumerate(MetadataDf.iterrows()):
                Future = Executor.submit(self._ExtractFeaturesForRow, Row, MetadataPath)
                FutureMap[Future] = Position

            CompletedCount = 0
            for Future in as_completed(FutureMap):
                Position = FutureMap[Future]

                try:
                    FeatureRow = Future.result()
                except Exception as Exc:
                    FeatureRow = {
                        "FeatureStatus": "failed",
                        "FeatureError": repr(Exc),
                    }

                FeatureRow["_InputOrder"] = Position
                ResultRows.append(FeatureRow)

                CompletedCount += 1
                if CompletedCount % 100 == 0 or CompletedCount == TotalRows:
                    self.Logger.info("Completed %s/%s", CompletedCount, TotalRows)

        return self._FinalizeFeatureDf(pd.DataFrame(ResultRows))

    def _FinalizeFeatureDf(self, FeatureDf: pd.DataFrame) -> pd.DataFrame:
        if "_InputOrder" in FeatureDf.columns:
            FeatureDf = FeatureDf.sort_values("_InputOrder").drop(columns=["_InputOrder"]).reset_index(drop=True)
        return FeatureDf

    def _ExtractFeaturesForRow(self, Row: pd.Series, MetadataPath: Path) -> Dict[str, Any]:
        Result: Dict[str, Any] = {
            "FeatureStatus": "unknown",
            "FeatureError": None,
        }

        Result.update(self._IdentifierAndMetadataFeatures(Row))
        LightCurvePath = self._ResolvePath(Row.get("lightCurvePath"), MetadataPath)

        if LightCurvePath is None:
            Result["FeatureStatus"] = "missing_lightcurve_path"
            Result["FeatureError"] = "No lightCurvePath"
            return Result

        if not LightCurvePath.exists():
            Result["FeatureStatus"] = "missing_lightcurve_file"
            Result["FeatureError"] = str(LightCurvePath)
            return Result

        try:
            Time, Flux, FluxErr = self._LoadLightCurve(LightCurvePath)

            if len(Flux) < self.MinCadences:
                Result["FeatureStatus"] = "too_few_cadences"
                Result["FeatureError"] = f"Only {len(Flux)} finite cadences"
                Result["CadenceCount"] = float(len(Flux))
                return Result

            Result.update(self._BasicStatisticalFeatures(Time, Flux))
            Result.update(self._VariabilityFeatures(Time, Flux))

            LombScargleFeatures = self._LombScargleFeatures(Time, Flux, FluxErr)
            Result.update(LombScargleFeatures)

            BestPeriod = LombScargleFeatures.get("LsBestPeriod", np.nan)
            Result.update(self._PhaseFeatures(Time, Flux, BestPeriod))

            Result["FeatureStatus"] = "ok"

        except Exception as Exc:
            Result["FeatureStatus"] = "failed"
            Result["FeatureError"] = repr(Exc)

        return Result

    def _LoadLightCurve(self, LightCurvePath: Path) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        FluxErrRaw: Optional[np.ndarray] = None

        try:
            LightCurve = lk.read(str(LightCurvePath))

            Time = np.asarray(getattr(LightCurve.time, "value", LightCurve.time), dtype=float)
            Flux = np.asarray(getattr(LightCurve.flux, "value", LightCurve.flux), dtype=float)

            if not hasattr(LightCurve, "flux_err") or LightCurve.flux_err is None:
                self.Logger.debug("flux_err ignored: missing")
            else:
                FluxErrRaw = np.asarray(getattr(LightCurve.flux_err, "value", LightCurve.flux_err), dtype=float)

        except Exception as LightkurveExc:
            self.Logger.debug("Using astropy.io.fits fallback for %s after lightkurve.read failed: %s", LightCurvePath, LightkurveExc)

            with fits.open(str(LightCurvePath)) as Hdul:
                TableData = None
                for Hdu in Hdul:
                    if Hdu.data is None:
                        continue
                    if not hasattr(Hdu, "columns"):
                        continue
                    TableData = Hdu.data
                    break

                if TableData is None:
                    raise ValueError(f"FITS fallback failed: no table HDU with data in {LightCurvePath}") from LightkurveExc

                ColumnMap = {Name.upper(): Name for Name in TableData.names}

                def _GetColumn(Candidates: list[str]) -> Optional[np.ndarray]:
                    for Candidate in Candidates:
                        ColumnName = ColumnMap.get(Candidate.upper())
                        if ColumnName is not None:
                            return np.asarray(TableData[ColumnName], dtype=float)
                    return None

                Time = _GetColumn(["TIME", "time"])
                Flux = _GetColumn(["FLUX", "flux", "SAP_FLUX", "PDCSAP_FLUX"])
                FluxErrRaw = _GetColumn(["FLUX_ERR", "flux_err", "SAP_FLUX_ERR", "PDCSAP_FLUX_ERR"])

                if Time is None or Flux is None:
                    raise ValueError(
                        f"FITS fallback failed: required time/flux columns not found in {LightCurvePath}"
                    ) from LightkurveExc

                if FluxErrRaw is None:
                    self.Logger.debug("flux_err ignored: missing")

        OriginalFluxShape = Flux.shape

        # Time/Flux are the required arrays; never gate them on flux_err quality.
        Mask = np.isfinite(Time) & np.isfinite(Flux)

        Time = Time[Mask]
        Flux = Flux[Mask]

        FluxErr: Optional[np.ndarray] = None
        if FluxErrRaw is not None:
            if FluxErrRaw.shape != OriginalFluxShape:
                self.Logger.debug(
                    "flux_err ignored: shape mismatch (flux_err=%s, flux=%s)",
                    FluxErrRaw.shape,
                    OriginalFluxShape,
                )
            else:
                CandidateFluxErr = FluxErrRaw[Mask]
                ValidErrMask = np.isfinite(CandidateFluxErr) & (CandidateFluxErr > 0)

                if not np.any(ValidErrMask):
                    self.Logger.debug("flux_err ignored: no finite positive values")
                elif not np.all(ValidErrMask):
                    self.Logger.debug("flux_err ignored: contains non-finite or non-positive values")
                else:
                    FluxErr = CandidateFluxErr

        Order = np.argsort(Time)
        Time = Time[Order]
        Flux = Flux[Order]

        if FluxErr is not None:
            FluxErr = FluxErr[Order]

        return Time, Flux, FluxErr

    def _BasicStatisticalFeatures(self, Time: np.ndarray, Flux: np.ndarray) -> Dict[str, float]:
        CadenceCount = len(Flux)

        P01, P05, P10, P25, P50, P75, P90, P95, P99 = np.percentile(
            Flux,
            [1, 5, 10, 25, 50, 75, 90, 95, 99],
        )

        FluxStd = float(np.std(Flux))

        TailUpper = float(P95 - P50)
        TailLower = float(P50 - P05)
        TailAsymmetry = float(TailUpper - TailLower)
        TailRatio = self._SafeDivide(TailUpper, TailLower + self.Eps)

        return {
            "CadenceCount": float(CadenceCount),
            "TimeSpanDays": float(np.max(Time) - np.min(Time)) if CadenceCount > 1 else np.nan,
            "MedianCadenceDays": float(np.median(np.diff(Time))) if CadenceCount > 2 else np.nan,
            "FluxMean": float(np.mean(Flux)),
            "FluxStd": FluxStd,
            "FluxVariance": float(np.var(Flux)),
            "FluxMedian": float(P50),
            "FluxMad": float(np.median(np.abs(Flux - P50))),
            "FluxMin": float(np.min(Flux)),
            "FluxMax": float(np.max(Flux)),
            "FluxP01": float(P01),
            "FluxP05": float(P05),
            "FluxP10": float(P10),
            "FluxP25": float(P25),
            "FluxP75": float(P75),
            "FluxP90": float(P90),
            "FluxP95": float(P95),
            "FluxP99": float(P99),
            "FluxIqr": float(P75 - P25),
            "FluxAmplitude": float(np.max(Flux) - np.min(Flux)),
            "FluxPercentAmplitude95To5": float(P95 - P05),
            "FluxPercentAmplitude90To10": float(P90 - P10),
            "FluxSkewness": float(skew(Flux, bias=False)) if CadenceCount >= 3 and FluxStd > self.Eps else np.nan,
            "FluxKurtosis": float(kurtosis(Flux, bias=False)) if CadenceCount >= 4 and FluxStd > self.Eps else np.nan,
            "TailUpper": TailUpper,
            "TailLower": TailLower,
            "TailAsymmetry": TailAsymmetry,
            "TailRatio": TailRatio,
        }

    def _VariabilityFeatures(self, Time: np.ndarray, Flux: np.ndarray) -> Dict[str, float]:
        CadenceCount = len(Flux)

        if CadenceCount < 3:
            return {
                "EtaVonNeumann": np.nan,
                "MaxAbsSlope": np.nan,
                "MedianAbsSuccessiveDiff": np.nan,
                "FractionBeyond1Std": np.nan,
                "FractionBeyond2Std": np.nan,
            }

        FluxMean = float(np.mean(Flux))
        FluxStd = float(np.std(Flux))
        FluxVariance = float(np.var(Flux))

        FluxDiff = np.diff(Flux)
        TimeDiff = np.diff(Time)

        ValidTimeDiff = np.isfinite(TimeDiff) & (np.abs(TimeDiff) > self.Eps)
        Slopes = FluxDiff[ValidTimeDiff] / TimeDiff[ValidTimeDiff] if np.any(ValidTimeDiff) else np.array([])

        Eta = (
            np.sum(FluxDiff ** 2) / ((CadenceCount - 1) * FluxVariance)
            if FluxVariance > self.Eps
            else np.nan
        )

        return {
            "EtaVonNeumann": float(Eta) if np.isfinite(Eta) else np.nan,
            "MaxAbsSlope": float(np.max(np.abs(Slopes))) if Slopes.size else np.nan,
            "MedianAbsSuccessiveDiff": float(np.median(np.abs(FluxDiff))),
            "FractionBeyond1Std": float(np.mean(np.abs(Flux - FluxMean) > FluxStd)) if FluxStd > self.Eps else np.nan,
            "FractionBeyond2Std": float(np.mean(np.abs(Flux - FluxMean) > 2 * FluxStd)) if FluxStd > self.Eps else np.nan,
        }

    def _LombScargleFeatures(
        self,
        Time: np.ndarray,
        Flux: np.ndarray,
        FluxErr: Optional[np.ndarray],
    ) -> Dict[str, float]:
        Result = {
            "LsBestPeriod": np.nan,
            "LsBestFrequency": np.nan,
            "LsMaxPower": np.nan,
            "LsFalseAlarmProbability": np.nan,
            "LsPeriod2": np.nan,
            "LsPeriod3": np.nan,
            "LsPower2": np.nan,
            "LsPower3": np.nan,
            "LsPowerRatio21": np.nan,
            "LsPowerRatio31": np.nan,
            "LsPeriodRatio21": np.nan,
            "LsPeriodRatio31": np.nan,
        }

        if len(Flux) < self.MinCadences:
            return Result

        TimeSpan = float(np.max(Time) - np.min(Time))
        if not np.isfinite(TimeSpan) or TimeSpan <= 0:
            return Result

        MaxPeriod = min(self.MaxPeriodDays, 0.9 * TimeSpan)
        if MaxPeriod <= self.MinPeriodDays:
            return Result

        CenteredFlux = Flux - np.nanmedian(Flux)

        try:
            if FluxErr is not None:
                LombScargleModel = LombScargle(Time, CenteredFlux, dy=FluxErr)
            else:
                LombScargleModel = LombScargle(Time, CenteredFlux)

            Frequency, Power = LombScargleModel.autopower(
                minimum_frequency=1.0 / MaxPeriod,
                maximum_frequency=1.0 / self.MinPeriodDays,
                samples_per_peak=self.SamplesPerPeak,
            )

            FiniteMask = np.isfinite(Frequency) & np.isfinite(Power) & (Frequency > 0)
            Frequency = Frequency[FiniteMask]
            Power = Power[FiniteMask]

            if len(Power) == 0:
                return Result

            Order = np.argsort(Power)[::-1]

            BestFrequency = float(Frequency[Order[0]])
            BestPeriod = float(1.0 / BestFrequency)
            BestPower = float(Power[Order[0]])

            Result.update(
                {
                    "LsBestPeriod": BestPeriod,
                    "LsBestFrequency": BestFrequency,
                    "LsMaxPower": BestPower,
                    "LsFalseAlarmProbability": float(LombScargleModel.false_alarm_probability(BestPower)),
                }
            )

            if len(Order) > 1:
                Period2 = float(1.0 / Frequency[Order[1]])
                Power2 = float(Power[Order[1]])
                Result.update(
                    {
                        "LsPeriod2": Period2,
                        "LsPower2": Power2,
                        "LsPowerRatio21": self._SafeDivide(Power2, BestPower),
                        "LsPeriodRatio21": self._SafeDivide(Period2, BestPeriod),
                    }
                )

            if len(Order) > 2:
                Period3 = float(1.0 / Frequency[Order[2]])
                Power3 = float(Power[Order[2]])
                Result.update(
                    {
                        "LsPeriod3": Period3,
                        "LsPower3": Power3,
                        "LsPowerRatio31": self._SafeDivide(Power3, BestPower),
                        "LsPeriodRatio31": self._SafeDivide(Period3, BestPeriod),
                    }
                )

        except Exception as Exc:
            self.Logger.debug("Lomb-Scargle failed: %s", Exc)

        return Result

    def _PhaseFeatures(self, Time: np.ndarray, Flux: np.ndarray, Period: float) -> Dict[str, float]:
        Result = {
            "PhaseCurveStd": np.nan,
            "PhaseCurveRange": np.nan,
            "PhaseCurveSmoothness": np.nan,
            "PhasePeakPhase": np.nan,
            "PhaseTroughPhase": np.nan,
            "PhasePeakToTroughPhaseDelta": np.nan,
        }

        if not np.isfinite(Period) or Period <= 0 or len(Flux) < self.MinCadences:
            return Result

        try:
            Phase = (Time % Period) / Period
            Order = np.argsort(Phase)
            Phase = Phase[Order]
            Flux = Flux[Order]

            BinEdges = np.linspace(0.0, 1.0, self.PhaseBinCount + 1)
            BinIndex = np.digitize(Phase, BinEdges) - 1

            BinnedPhase = []
            BinnedFlux = []

            for BinNumber in range(self.PhaseBinCount):
                BinMask = BinIndex == BinNumber
                if np.any(BinMask):
                    BinnedPhase.append(float(np.median(Phase[BinMask])))
                    BinnedFlux.append(float(np.median(Flux[BinMask])))

            BinnedPhase = np.asarray(BinnedPhase, dtype=float)
            BinnedFlux = np.asarray(BinnedFlux, dtype=float)

            if len(BinnedFlux) < 5:
                return Result

            PeakIndex = int(np.argmax(BinnedFlux))
            TroughIndex = int(np.argmin(BinnedFlux))

            PeakPhase = float(BinnedPhase[PeakIndex])
            TroughPhase = float(BinnedPhase[TroughIndex])

            RawDelta = abs(PeakPhase - TroughPhase)
            CyclicDelta = min(RawDelta, 1.0 - RawDelta)

            CyclicDiff = np.diff(np.r_[BinnedFlux, BinnedFlux[0]])

            Result.update(
                {
                    "PhaseCurveStd": float(np.std(BinnedFlux)),
                    "PhaseCurveRange": float(np.max(BinnedFlux) - np.min(BinnedFlux)),
                    "PhaseCurveSmoothness": float(np.std(CyclicDiff)),
                    "PhasePeakPhase": PeakPhase,
                    "PhaseTroughPhase": TroughPhase,
                    "PhasePeakToTroughPhaseDelta": float(CyclicDelta),
                }
            )

        except Exception as Exc:
            self.Logger.debug("Phase feature extraction failed: %s", Exc)

        return Result

    def _IdentifierAndMetadataFeatures(self, Row: pd.Series) -> Dict[str, Any]:
        Result: Dict[str, Any] = {}

        ColumnsToPreserve = [
            "family",
            "VSXType",
            "VSXId",
            "Name",
            "ticId",
            "bestTicId",
            "ticDistanceArcmin",
            "lightCurvePath",
            "rawLightCurvePath",
            "trendFlag",
            "trendScore",
            "adfPValue",
        ]

        for ColumnName in ColumnsToPreserve:
            if ColumnName in Row.index:
                Result[ColumnName] = Row.get(ColumnName)

        QualityValue = Row.get("quality", Row.get("fitsQcStatus", Row.get("lightCurveQuality", np.nan)))
        ProvenanceValue = Row.get("provenance", Row.get("author", np.nan))

        QualityLabel = str(QualityValue) if not pd.isna(QualityValue) else "missing"
        ProvenanceLabel = str(ProvenanceValue) if not pd.isna(ProvenanceValue) else "missing"

        Result.update(
            {
                "QualityLabel": QualityLabel,
                "QualityScore": self.QualityScoreMap.get(QualityLabel, np.nan),
                "Provenance": ProvenanceLabel,
                "ProvenanceScore": self.ProvenanceScoreMap.get(ProvenanceLabel, np.nan),
                "OriginalFluxMedian": self._SafeFloat(Row.get("fluxMedian", np.nan)),
                "OriginalFluxStd": self._SafeFloat(Row.get("fluxStd", np.nan)),
                "OriginalFluxSnr": self._SafeFloat(Row.get("fluxSnr", np.nan)),
                "LowSnr": self._SafeBool(Row.get("lowSNR", False)),
                "LowQualityLightCurve": self._SafeBool(Row.get("lowQualityLightCurve", False)),
            }
        )

        return Result

    def _ResolvePath(self, PathValue: Any, MetadataPath: Path) -> Optional[Path]:
        if PathValue is None:
            return None

        if isinstance(PathValue, float) and pd.isna(PathValue):
            return None

        PathObj = Path(str(PathValue))

        if PathObj.is_absolute() and PathObj.exists():
            return PathObj

        if PathObj.exists():
            return PathObj

        CandidatePath = MetadataPath.parent / PathObj
        if CandidatePath.exists():
            return CandidatePath

        return PathObj

    def _SafeFloat(self, Value: Any) -> float:
        try:
            FloatValue = float(Value)
        except Exception:
            return np.nan

        return FloatValue if np.isfinite(FloatValue) else np.nan

    def _SafeBool(self, Value: Any) -> bool:
        if Value is None:
            return False
        if isinstance(Value, float) and pd.isna(Value):
            return False
        return bool(Value)

    def _SafeDivide(self, Numerator: float, Denominator: float) -> float:
        if not np.isfinite(Numerator) or not np.isfinite(Denominator) or abs(Denominator) < self.Eps:
            return np.nan
        return float(Numerator / Denominator)


def ParseArguments() -> argparse.Namespace:
    Parser = argparse.ArgumentParser(description="Extract ML-ready features from standardized TESS FITS files.")

    Parser.add_argument("--input", required=True, help="Input metadata parquet, e.g. TESSAugmented_QC_trend.parquet")
    Parser.add_argument("--output", required=True, help="Output feature parquet, e.g. TESS_features.parquet")
    Parser.add_argument("--min-period-days", type=float, default=0.05)
    Parser.add_argument("--max-period-days", type=float, default=100.0)
    Parser.add_argument("--samples-per-peak", type=int, default=10)
    Parser.add_argument("--min-cadences", type=int, default=100)
    Parser.add_argument("--phase-bin-count", type=int, default=20)
    Parser.add_argument("--worker-count", type=int, default=16)
    Parser.add_argument("--log-level", default="INFO")
    Parser.add_argument("--log-file", default=None, help="Optional path to write log output to a file")

    return Parser.parse_args()


def Main() -> None:
    Args = ParseArguments()

    LogHandlers = [logging.StreamHandler()]
    if Args.log_file:
        LogHandlers.append(logging.FileHandler(Args.log_file))

    logging.basicConfig(
        level=getattr(logging, Args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s:%(name)s:%(message)s",
        handlers=LogHandlers,
    )

    Extractor = FeatureExtractor(
        MinPeriodDays=Args.min_period_days,
        MaxPeriodDays=Args.max_period_days,
        SamplesPerPeak=Args.samples_per_peak,
        MinCadences=Args.min_cadences,
        PhaseBinCount=Args.phase_bin_count,
        WorkerCount=Args.worker_count,
    )

    Extractor.ExtractFeaturesFromParquet(Args.input, Args.output)


if __name__ == "__main__":
    Main()
