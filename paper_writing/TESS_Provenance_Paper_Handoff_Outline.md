# TESS Provenance Paper — Handoff Outline

**Working title:** Quantifying the Impact of TESS Light-Curve Provenance on Machine-Learning Variable-Star Classification  
**Format:** AASTeX / AAS LaTeX  
**Purpose:** Canonical handoff for the agreed manuscript outline, scientific narrative, key results, and drafting workflow.

## Central scientific narrative

The paper should not primarily be framed as “building a variable-star classifier.” The Random Forest is the experimental instrument. The main contribution is the controlled demonstration that TESS light-curve provenance changes downstream classification performance and learned feature importance, including when the same astrophysical objects are compared.

Narrative:
1. Construct a VSX-selected variable-star dataset and Random Forest classifier.
2. Observe unexpectedly large performance differences among SPOC, QLP, and TESSCut provenance.
3. Train provenance-specific models and observe differences in performance and feature-importance rankings.
4. Recognize that unmatched provenance subsets contain different stars, sample sizes, and class compositions.
5. Construct matched-star experiments: SPOC vs. TESSCut and QLP vs. TESSCut.
6. Hold stars, labels, train/test assignments, feature definitions, and RF configuration constant.
7. Show that performance and feature-importance differences persist.
8. Conclude that provenance should be treated as an explicit methodological factor in heterogeneous TESS ML analyses.

The manuscript must distinguish between (a) what was measured, (b) what the experiments support, and (c) hypotheses about why the effect occurs. The current work establishes the effect under controlled conditions but does not establish the exact processing mechanism responsible.

# Abstract

Write last. Target roughly 200–300 words, subject to venue requirements.

Include:
- problem: SPOC, QLP, and TESSCut differ in processing/extraction;
- dataset: 7000+ VSX stars, nine classes, common feature extraction, Random Forest;
- initial provenance-associated observation;
- matched-star controlled experiments;
- conclusion: provenance changes classification performance and feature-importance structure.

Give the matched experiments more emphasis than the original classifier.

# 1. Introduction

## 1.1 Variable-Star Classification from Photometric Time Series
Introduce variable stars, photometric time series, large-survey classification, ML, and variability features. Avoid a textbook survey of every variable-star family.

## 1.2 TESS and Heterogeneous Light-Curve Products
Introduce TESS, SPOC/PDCSAP, QLP, and TESSCut-derived photometry. Establish that these products are not automatically interchangeable numerical representations because their extraction and processing differ.

## 1.3 The Provenance Problem
Explain that ML work often emphasizes features, classifiers, hyperparameters, and training populations, while upstream photometric provenance may alter the variability signatures from which features are derived. Explain how the original classifier produced unexpected provenance-stratified performance differences and motivated the new research question.

## 1.4 Research Questions and Contributions
RQ1: Does RF classification performance differ systematically among TESS light-curve provenances?

RQ2: Do models trained from different provenances assign different relative importance to the same extracted features?

RQ3: Do these differences persist when astrophysical objects, labels, train/test assignments, feature definitions, and classifier configuration are held constant?

Contributions: VSX-selected stratified dataset; confidence-first VSX–TESS matching; initial provenance analysis; matched-star experiments; performance comparison; built-in and permutation importance; Spearman rank comparison.

# 2. Background and Related Work

## 2.1 Machine Learning for Variable-Star Classification
Review relevant variable-star ML work, Random Forest, Logistic Regression where useful, common variability features, and TESS classification. Position this paper as a provenance study rather than primarily a new classifier.

## 2.2 TESS Light-Curve Products
Describe SPOC, QLP, and TESSCut separately. Explain that TESSCut provides FFI cutouts and that this study derives photometry from them. Consider a compact comparison table of source, extraction, flux/product, and relevant processing.

## 2.3 Catalog Cross-Matching and Dataset Construction
Discuss VSX, TIC/TESS association, positional matching, relevant work such as Fetherolf et al., and why this study starts from VSX to build a labeled supervised-learning dataset with controlled representation.

## 2.4 Data Provenance in Astronomical Machine Learning
Frame the broader methodological problem: upstream measurement, extraction, calibration, and processing can affect downstream ML. Literature for this subsection should be researched rather than inferred.

# 3. Data and Dataset Construction

## 3.1 VSX Source Catalog and Class Taxonomy
Describe the nine families: CEPHEID, CV, DSCT_SXPHE, ECLIPSING, ELLIPSOIDAL_ROT, LONG_PERIOD, RRLYR, XRAY, YSO. Document subtype mapping, sampling caps, and final usable counts. Include a class table.

## 3.2 Stratified VSX Sampling
Describe sampling across class/family, subtype where applicable, RA, and Dec. Prefer “approximately class-balanced and spatially stratified” rather than claiming exact balance.

## 3.3 Confidence-First VSX–TESS Cross-Matching
Conceptual flow: VSX coordinates → TIC candidate within 1 arcsec → suitable SPOC/QLP product if available → otherwise TESSCut-derived photometry at the VSX position. Verify exact priority/candidate logic from notebooks before final prose. Include a workflow figure.

## 3.4 Cross-Match Validation
### 3.4.1 Positional Separation
Report match success and separation statistics.

### 3.4.2 Duplicate/Ambiguous Matches
Describe duplicate TIC and ambiguity checks.

### 3.4.3 Proper-Motion Validation
Describe coordinate epoch assumptions, SIMBAD proper-motion retrieval, TESS observation epochs, predicted displacement, and implications for the 1-arcsec radius.

## 3.5 Final Dataset
Report final total, class distribution, provenance distribution, and quality-filtered counts. Current poster-level numbers to verify before manuscript lock: total 7,281; SPOC 509; QLP 1,353; TESSCut 5,419.

# 4. Light-Curve Processing and Feature Extraction

## 4.1 Light-Curve Acquisition
Describe SPOC, QLP, and TESSCut acquisition separately, including sectors, multiple-sector handling, quality filtering, and flux products actually used.

## 4.2 TESSCut Photometry
Document cutout retrieval, target location, aperture, background handling, flux extraction, and quality filtering. Include only implemented methods.

## 4.3 Preprocessing
Document normalization, invalid/outlier handling, sector treatment, and detrending decisions. The unsuccessful conditional TESSCut detrending analysis should probably be summarized briefly or moved to an appendix unless it becomes important to interpretation.

## 4.4 Feature Extraction
Organize the 53 features into:
- distribution/statistical;
- time-domain variability;
- periodicity/Lomb–Scargle;
- phase/morphology.

Put the complete feature-definition table in an appendix or supplement.

# 5. Machine-Learning Methodology

## 5.1 Random Forest Classifier
Document library, hyperparameters, random seed, training procedure, and rationale. RF is useful for nonlinear classification and for feature-importance comparisons.

## 5.2 Logistic Regression Baseline
Treat Logistic Regression as a baseline. Detailed results may go to supplementary material if they do not materially affect the provenance conclusion.

## 5.3 Train/Validation/Test Design
Document splitting, stratification, proportions, random state, and how matched experiments use identical object assignments across provenance.

## 5.4 Evaluation Metrics
Define accuracy, balanced accuracy, precision, recall, F1, and confusion matrices as needed. Emphasize balanced accuracy because provenance-specific samples are not perfectly balanced.

## 5.5 Feature Importance
### 5.5.1 RF Impurity-Based Importance
Explain the numerical importance and relevant limitations.

### 5.5.2 Permutation Importance
Explain the permutation procedure and why it is complementary.

### 5.5.3 Spearman Rank Comparison
Explain ranking the same features and comparing rankings across provenance with Spearman rho.

# 6. Initial Evidence of a Provenance Effect

Frame this as the discovery/observational stage, not the definitive experiment.

## 6.1 Mixed-Provenance Classifier
Current held-out provenance-stratified accuracies: SPOC 73.6%, QLP 64.5%, TESSCut 31.7%. Explain that this unexpected disparity motivated the provenance study.

## 6.2 Separate Provenance-Specific Models
Current unmatched results:

| Provenance | Test Accuracy | Balanced Accuracy |
|---|---:|---:|
| SPOC | 74.3% | 52.1% |
| QLP | 71.9% | 47.9% |
| TESSCut | 34.4% | 29.3% |

## 6.3 Feature-Importance Disagreement
Current built-in RF rank correlations: SPOC–QLP rho=0.719; SPOC–TESSCut rho=0.665; QLP–TESSCut rho=0.355.

## 6.4 Why the Initial Comparison Is Insufficient
State explicitly that provenance subsets differ in object identity, sample size, and class composition. These results establish association but cannot isolate provenance. This motivates Section 7.

# 7. Matched-Star Controlled Provenance Experiments

This is the centerpiece of the paper and should receive more emphasis than Section 6.

## 7.1 Experimental Design
A three-way comparison would greatly reduce the sample because relatively few stars have all products. Use two independent experiments: SPOC vs. TESSCut and QLP vs. TESSCut.

Key control statement:

**Same stars; same labels; same train/test assignments; same feature definitions; same RF configuration; different light-curve provenance.**

## 7.2 SPOC vs. TESSCut
N=506; train=354; test=152.

| Metric | SPOC | TESSCut |
|---|---:|---:|
| Test accuracy | 75.0% | 67.1% |
| Balanced accuracy | 53.1% | 43.6% |

Feature-importance rank correlations: built-in RF rho=0.743; permutation rho=-0.107.

Include performance comparison, confusion matrices, and feature-importance/rank visualizations as appropriate.

## 7.3 QLP vs. TESSCut
N=1,355; train=948; test=407.

| Metric | QLP | TESSCut |
|---|---:|---:|
| Test accuracy | 69.5% | 52.6% |
| Balanced accuracy | 42.6% | 26.9% |

Feature-importance rank correlations: built-in RF rho=0.678; permutation rho=0.407.

## 7.4 Synthesis
Both matched experiments show higher RF performance for SPOC/QLP than corresponding TESSCut-derived light curves under matched ML conditions. Feature-importance structure also changes. Distinguish the predictive-performance effect from the learned feature-importance effect.

# 8. Discussion

## 8.1 What the Matched Experiments Establish
Unmatched comparisons establish association; matched comparisons remove major population-level confounds. Use careful language such as “provide strong evidence that light-curve provenance itself contributes to the observed differences.”

## 8.2 Possible Mechanisms
Discuss as hypotheses unless tested:
- aperture definition;
- background subtraction;
- contamination/crowding;
- detrending;
- systematic-error correction;
- calibration;
- noise characteristics;
- preservation/suppression of astrophysical variability;
- other verified product-specific differences.

Do not claim the current study identifies the exact mechanism.

## 8.3 Feature Importance as Evidence of Changed Model Dependence
Discuss the fact that provenance affects not only accuracy but the relative features used by the model. Treat the matched SPOC–TESSCut permutation rho=-0.107 as striking but do not overinterpret a single coefficient. Feature importance is model behavior, not direct astrophysical causality.

## 8.4 Implications for Astronomical ML
Recommend that studies using heterogeneous TESS products:
- record provenance explicitly;
- avoid assuming products are interchangeable;
- test provenance-stratified performance;
- consider provenance during dataset construction;
- evaluate feature stability across processing products.

## 8.5 Limitations
Potential limitations to discuss:
- Random Forest is the primary classifier;
- TESSCut extraction choices are study-specific;
- matched experiments are pairwise rather than three-way;
- remaining class imbalance;
- feature set is engineered rather than exhaustive;
- exact mechanisms underlying provenance effects are not isolated;
- results should not automatically be generalized to all TESS ML tasks or all photometric products.

## 8.6 Future Work
Possible extensions:
- controlled processing experiments to isolate aperture/detrending/background effects;
- additional classifiers;
- other surveys/products;
- additional matched samples;
- feature-level stability analysis;
- potentially Gaia-derived contextual features, while keeping the provenance question conceptually separate.

# 9. Conclusions

Summarize:
1. construction of the VSX-selected TESS variable-star dataset;
2. confidence-first matching and validation;
3. initial provenance-associated performance differences;
4. matched-star controlled experiments;
5. persistent changes in accuracy/balanced accuracy;
6. changes in feature-importance rankings;
7. methodological implication: TESS light-curve provenance should be treated explicitly rather than assuming heterogeneous products are interchangeable.

Keep conclusions concise and avoid introducing new analysis.

# Appendices / Supplementary Material

Potential items:
- complete 53-feature definitions;
- RF and Logistic Regression hyperparameters;
- detailed pipeline implementation;
- additional confusion matrices;
- complete feature-importance tables;
- detrending experiments;
- cross-match validation details;
- supplementary class/provenance counts;
- additional figures.

# Drafting Workflow

Use Overleaf as the canonical manuscript.

Recommended loop:
1. Request one section/subsection in AASTeX-compatible LaTeX.
2. Draft it from verified project materials/results, with citation/figure/table TODOs where needed.
3. Paste into Overleaf and compile.
4. Review scientific correctness and narrative before polishing.
5. Return feedback/corrections.
6. Revise until the section is locked.
7. Move to the next section.
8. Every 2–3 major sections, upload the current full `.tex` for a manuscript-wide consistency review.

Recommended writing order:

**Sections 3–5 (Methods) → Sections 6–7 (Results) → Section 8 (Discussion) → Sections 1–2 (Introduction/Related Work) → Section 9 (Conclusion) → Abstract**

This order ensures the Introduction and Abstract describe what the completed analysis actually demonstrates.

# Manuscript Guardrails

- Verify every final numerical value against the canonical analysis notebooks before publication.
- Do not silently reconcile conflicting historical numbers; identify the final canonical result.
- Use “approximately balanced/stratified” unless exact balance is demonstrated.
- Separate association in unmatched datasets from stronger evidence in matched experiments.
- Do not claim the exact mechanism of the provenance effect unless it is experimentally isolated.
- Do not equate RF feature importance with astrophysical causality.
- Keep Random Forest as the experimental framework; provenance is the scientific focus.
- Use citations for methodological and astronomical claims; avoid citations for the study's own results.
- Keep AASTeX/Overleaf as the manuscript format.
