# Scalable Construction and Machine Learning Analysis of a VSX–TESS Variable Star Dataset

## Abstract

We present an end-to-end research pipeline for large-scale variable star classification using photometric observations from the NASA TESS mission and labels from the AAVSO VSX catalog. A major challenge in this problem is the incomplete availability of high-quality TESS light curves and the ambiguity in mapping VSX catalog objects to TESS TIC identifiers. To address these issues, we developed a scalable multi-stage pipeline that combines multi-candidate TIC crossmatching, hierarchical light curve recovery using SPOC, QLP, and TESSCut products, and automated feature extraction using time-series analysis techniques including Lomb–Scargle periodograms.

Our pipeline improves usable light curve coverage from approximately 12% under naive nearest-neighbor mapping to approximately 96.5% through the integration of TESSCut-based recovery. We further investigate the role of trend detection and conditional detrending in improving machine learning performance. A segment-wise robust drift detector was developed to identify strong long-term drift within TESSCut light curves. Although conditional detrending successfully removed strong drift signatures, downstream Random Forest classification performance showed little improvement, suggesting that TESSCut limitations are not dominated by simple low-frequency trends alone.

Feature extraction and classification experiments were performed using Logistic Regression and Random Forest models across multiple TESS provenance sources. Results indicate strong provenance-dependent behavior: SPOC-derived light curves achieved the strongest classification performance, while TESSCut data remained substantially more challenging despite successful data recovery. These findings highlight the importance of both scalable data engineering and provenance-aware modeling in modern astronomical time-series analysis.

---

# 1. Introduction

Large-scale time-domain astronomy missions such as the entity["space_mission","Transiting Exoplanet Survey Satellite","NASA space telescope"] (TESS) provide unprecedented opportunities for studying stellar variability. TESS continuously monitors large regions of the sky and produces massive quantities of photometric time-series data suitable for variable star analysis.

At the same time, the entity["organization","American Association of Variable Star Observers","AAVSO"] Variable Star Index (VSX) provides an extensive catalog of variable star classifications. Combining VSX labels with TESS photometric observations enables large-scale supervised machine learning studies of stellar variability.

However, constructing such a dataset presents several challenges:

1. VSX objects do not map uniquely to TESS TIC identifiers.
2. Many variable stars do not have pipeline-generated SPOC light curves.
3. TESS observational coverage is incomplete and provenance-dependent.
4. Extracted TESSCut light curves often contain stronger systematics and lower photometric quality.
5. Variable star families exhibit very different temporal characteristics.

This project investigates whether a scalable and scientifically robust pipeline can be constructed to:

- maximize TESS light curve recovery,
- preserve data provenance and quality information,
- extract meaningful time-series features,
- and perform variable star classification using machine learning methods.

---

# 2. Data Sources

## 2.1 AAVSO VSX Catalog

Variable star labels were obtained from the entity["organization","American Association of Variable Star Observers","AAVSO"] Variable Star Index (VSX). The VSX catalog contains large numbers of classified variable stars across multiple astrophysical families.

Families included in this work include:

- CEPHEID
- CV
- DSCT_SXPHE
- ECLIPSING
- ELLIPSOIDAL_ROT
- LONG_PERIOD
- RRLYR
- XRAY
- YSO

These labels serve as the supervised learning targets for downstream classification.

## 2.2 TESS Light Curve Sources

TESS light curves were obtained through three primary sources:

### SPOC

Pipeline-generated light curves from the TESS Science Processing Operations Center (SPOC).

Advantages:

- highest photometric quality
- systematic corrections already applied
- optimized apertures
- PDCSAP flux available

Limitations:

- limited target coverage
- biased toward pre-selected targets

### QLP

Quick-Look Pipeline (QLP) light curves.

Advantages:

- broader sky coverage
- publicly available pipeline products

Limitations:

- lower quality than SPOC
- reduced systematic correction quality

### TESSCut

Coordinate-based extraction from TESS Full Frame Images (FFIs).

Advantages:

- dramatically increases data coverage
- enables recovery of stars without pipeline products

Limitations:

- higher noise
- aperture contamination
- background estimation issues
- sector-to-sector inconsistencies

---

# 3. Data Pipeline

## 3.1 VSX–TIC Crossmatching

A major challenge in combining VSX and TESS data is the ambiguity in mapping VSX coordinates to TESS TIC identifiers.

A naive nearest-neighbor approach produced very poor recovery rates (~12%), largely because:

- the nearest TIC object often lacked TESS light curves,
- multiple nearby TIC candidates existed,
- and TESS pipeline products were incomplete.

To address this, we implemented a multi-candidate crossmatching strategy.

For each VSX object:

1. Up to five nearby TIC candidates were identified.
2. Batch MAST queries were used to determine SPOC/QLP availability.
3. Candidates were ranked hierarchically:

$$
\text{SPOC} \rightarrow \text{QLP} \rightarrow \text{TESSCut fallback}
$$

This significantly improved light curve recovery.

---

## 3.2 Two-Stage Pipeline Design

The pipeline was divided into two stages.

### Stage 1 — Metadata Construction

The metadata stage:

- performs VSX–TIC matching,
- records provenance information,
- stores candidate TIC lists,
- and determines best available source.

### Stage 2 — Large-Scale Download

The download stage:

- retrieves SPOC/QLP products when available,
- otherwise falls back to TESSCut extraction,
- stores FITS files locally,
- and records quality-control metadata.

This separation greatly improved scalability and reproducibility.

---

## 3.3 TESSCut Recovery

TESSCut recovery proved essential.

Final coverage statistics:

| Source | Count | Percent |
|---|---:|---:|
| TESSCut | 5034 | 68.30% |
| SPOC | 467 | 6.34% |
| QLP | 1612 | 21.87% |
| Missing | 257 | 3.49% |

Total usable light curves:

$$
96.5\%
$$

This represented a major improvement over:

- naive nearest-neighbor mapping (~12%)
- SPOC+QLP only (~22.5%)

The results demonstrated that the majority of recoverable variable stars required TESSCut extraction.

---

## 3.4 Dual Light Curve Representation

Two representations were stored:

### Raw FITS

Preserves the original extracted light curve.

### Standardized FITS

Applies normalization and preprocessing to facilitate downstream feature extraction.

Both representations were retained to avoid irreversible preprocessing decisions and to reduce the risk of data leakage.

---

# 4. Trend Detection and Conditional Detrending

## 4.1 Motivation

Initial machine learning experiments showed that TESSCut-derived light curves produced substantially worse classification performance than SPOC and QLP.

One possible explanation was the presence of:

- long-term instrumental drift,
- background systematics,
- sector-level offsets,
- and low-frequency contamination.

This motivated an investigation into automated trend detection and conditional detrending.

---

## 4.2 Lomb–Scargle Low-Frequency Trend Detection

The initial approach used a low-frequency entity["scientific_concept","Lomb–Scargle periodogram","time series frequency analysis"] detector.

The idea was:

- compute low-frequency LS power,
- compare against reference-band power,
- and classify strong low-frequency dominance as trend.

However, applying LS directly to stitched multi-sector light curves produced unrealistic results:

- trend rates exceeded 80–90%,
- even for short-period variable families,
- indicating that the detector was responding to sector gaps and broad curvature rather than true removable drift.

This led to an important methodological observation:

> Lomb–Scargle analysis is highly effective for periodicity analysis but overly sensitive for trend detection on short, gapped TESS segments.

The LS detector was therefore retained only as a diagnostic metric.

---

## 4.3 Segment-Wise Robust Drift Detection

A second approach was developed using a segment-wise robust drift metric.

Light curves were first split into contiguous observing segments based on large temporal gaps.

For each segment, a linear trend was fitted:

$$
\text{flux} = a + bt
$$

The total fitted drift across the segment was computed:

$$
\text{drift} = |b \times \text{segmentDuration}|
$$

Robust scatter was estimated using the Median Absolute Deviation (MAD):

$$
\text{MAD} = \text{median}(|x_i - \text{median}(x)|)
$$

The final drift strength metric was defined as:

$$
\text{segmentDriftStrength} =
\frac{|\text{slope} \times \text{segmentDuration}|}
{1.4826 \times \text{MAD}}
$$

A segment was classified as drifting when:

$$
\text{segmentDriftStrength} \ge 7.0
$$

A light curve was classified as drifting only if:

$$
\text{fractionSegmentsWithDrift} \ge 0.5
$$

This majority criterion reduced sensitivity to isolated problematic segments.

---

## 4.4 Conditional Detrending

Conditional detrending was applied only to:

- TESSCut provenance light curves
- with detected robust drift.

SPOC and QLP products were left unchanged because they already contain significant systematic correction.

Detrended FITS files were stored separately from raw light curves, and metadata columns were added to record:

- detrended status,
- detrended file path,
- drift statistics,
- and detrending thresholds.

---

## 4.5 Detrending Results

Trend detection and detrending were successfully implemented and verified.

However, downstream machine learning experiments showed:

> Conditional detrending did not materially improve Random Forest classification performance for TESSCut light curves.

This suggests that TESSCut limitations are not dominated by simple removable low-frequency drift.

Instead, remaining limitations likely include:

- photometric noise,
- aperture contamination,
- background estimation uncertainty,
- and reduced signal separability.

This negative result is scientifically meaningful because it demonstrates that simple detrending alone is insufficient to recover SPOC-like performance.

---

# 5. Feature Extraction

## 5.1 Overview

A large-scale feature extraction pipeline was developed to convert light curves into machine-learning-ready feature vectors.

Features were extracted from:

- raw light curves,
- standardized light curves,
- and conditionally detrended TESSCut light curves.

---

## 5.2 Lomb–Scargle Features

The primary periodicity analysis tool used in this work was the entity["scientific_concept","Lomb–Scargle periodogram","time series frequency analysis"].

Extracted LS features included:

- best frequency,
- best period,
- peak power,
- false alarm probability,
- top-N peak frequencies,
- period ratios,
- power ratios.

These features capture periodic structure across variable star families.

---

## 5.3 Statistical Features

Additional statistical features included:

- amplitude,
- mean flux,
- median flux,
- standard deviation,
- MAD,
- skewness,
- kurtosis,
- percentile ranges,
- tail asymmetry.

These features capture:

- variability magnitude,
- distribution shape,
- and asymmetry.

---

## 5.4 Quality and Provenance Features

Metadata and quality-related features were also preserved:

- provenance source,
- drift statistics,
- quality flags,
- segment counts,
- and finite-point statistics.

This enabled provenance-aware downstream analysis.

---

# 6. Machine Learning Analysis

## 6.1 Models

Two supervised learning approaches were explored:

### Logistic Regression

Served as a linear baseline model.

### Random Forest

Used as the primary nonlinear classification model.

Random Forest was selected because:

- it handles heterogeneous features well,
- is robust to nonlinear relationships,
- and provides feature-importance estimates.

---

## 6.2 Overall Results

Random Forest consistently outperformed Logistic Regression.

However, performance varied strongly by provenance source.

### SPOC

Highest classification performance.

### QLP

Moderate performance.

### TESSCut

Substantially lower performance despite successful recovery.

This demonstrated that increasing coverage does not automatically guarantee equivalent scientific quality.

---

## 6.3 Provenance Dependence

The provenance-dependent behavior was one of the most important findings of this work.

SPOC products benefit from:

- optimized apertures,
- sophisticated systematic correction,
- cotrending basis vectors,
- and higher photometric precision.

TESSCut products, while dramatically increasing coverage, remain significantly noisier.

This creates an important tradeoff:

$$
\text{completeness} \leftrightarrow \text{photometric quality}
$$

---

## 6.4 Family-Level Behavior

Different variable star families exhibited substantially different behavior.

Examples:

- RR Lyrae stars showed strong dependence on TESSCut recovery.
- Long-period variables exhibited naturally strong low-frequency variability.
- Some short-period pulsators were easier to classify.
- Irregular systems such as YSO and CV remained difficult.

These observations suggest that future work may benefit from:

- family-specific preprocessing,
- provenance-aware modeling,
- or hierarchical classification.

---

# 7. Discussion

This project demonstrates that scalable astronomical machine learning requires substantial data engineering in addition to modeling.

Several important conclusions emerged:

1. Naive VSX–TESS matching produces severe incompleteness.
2. TESSCut recovery is essential for high coverage.
3. Provenance strongly affects downstream ML performance.
4. Simple detrending alone does not solve TESSCut limitations.
5. Lomb–Scargle methods are highly effective for periodicity analysis but less suitable for generic trend detection.

The project also highlights an important scientific tradeoff:

$$
\text{coverage} \neq \text{quality}
$$

Recovering more data can increase dataset completeness while simultaneously reducing average photometric quality.

---

# 8. Future Work

Several future directions remain promising.

## 8.1 Provenance-Aware Modeling

Instead of treating all sources equally, future models may explicitly incorporate provenance information.

## 8.2 Advanced Cotrending

Simplified versions of SPOC-style cotrending basis vector correction may improve TESSCut quality.

## 8.3 Phase-Folded Deep Learning

Future work may explore:

- phase-folded representations,
- CNN-based approaches,
- or transformer architectures.

## 8.4 Physical Interpretation

Misclassified stars may contain:

- ambiguous variability,
- noisy observations,
- or potentially incorrect labels.

Cross-analysis with astrophysical context may provide additional insight.

---

# 9. Conclusion

We developed a scalable and provenance-aware pipeline for large-scale variable star analysis using VSX labels and TESS photometric data.

The pipeline dramatically improved usable light curve coverage through:

- multi-candidate TIC matching,
- hierarchical source selection,
- and TESSCut recovery.

We further investigated trend detection and conditional detrending using robust segment-wise drift analysis. While conditional detrending successfully removed strong drift behavior from many TESSCut light curves, it did not substantially improve downstream Random Forest classification performance.

This result suggests that the primary limitations of TESSCut photometry extend beyond simple low-frequency drift and likely involve deeper photometric and systematic challenges.

Overall, the project demonstrates the importance of combining:

- scalable data engineering,
- robust time-series analysis,
- and provenance-aware machine learning

for modern astronomical classification problems.

The resulting dataset and methodology provide a strong foundation for future work in variable star classification and large-scale astronomical time-series analysis.
