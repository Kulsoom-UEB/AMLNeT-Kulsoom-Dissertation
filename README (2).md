# Explainable Machine Learning for Suspicious Transaction Risk Detection in Anti-Money Laundering

**Umm E Kulsoom** (B01040340)
MSc Computer Science and Technology — Ulster University, Birmingham Campus
Supervisor: Dr. Anwar Haq

A reproducible, imbalance-aware and explainable machine-learning pipeline for suspicious-transaction risk detection on the AMLNet synthetic dataset. All model outputs are framed as **decision-support evidence for human AML reviewers**, not as automated compliance or legal determinations.

---

## Headline result

Class-weighted XGBoost, selected automatically on validation PR-AUC at an F1-optimal threshold of 0.99, evaluated **once** on the untouched test set:

| Metric | Value |
|---|---|
| Precision | 0.971 |
| Recall | 0.897 |
| F1-score | 0.933 |
| PR-AUC | 0.952 |
| ROC-AUC | 0.999 |

242 alerts raised from 163,526 test transactions (14.8 per 10,000), of which 97.1% were genuine laundering cases: 235 true positives, 7 false positives, 27 false negatives.

---

## Getting the dataset

**The dataset is not included in this repository.** The raw file is approximately 691 MB, which exceeds GitHub's 100 MB per-file limit, and the AMLNet licence (CC BY-NC 4.0) is best respected by pointing to the original source rather than redistributing a copy.

Download it from Zenodo:

- **AMLNet: Synthetic anti-money laundering transaction dataset**
- Author: Huda, S. (2025)
- DOI: [10.5281/zenodo.16736515](https://doi.org/10.5281/zenodo.16736515)
- Licence: CC BY-NC 4.0 (non-commercial, attribution required)
- File: `AMLNet.csv` — 691.3 MB (659.3 MiB)
- MD5: `7668fc7d74c787e07546ce85c6f790b9`

Place the downloaded file at:

```
data/raw/AMLNet.csv
```

`src/step00_setup.py` verifies the file against the MD5 checksum above before any processing begins, so a corrupted or wrong download is caught immediately.

### Dataset summary

| Property | Value |
|---|---|
| Transactions (published) | 1,090,173 |
| Transactions (after cleaning) | 1,090,172 — one record with a missing target removed |
| Laundering-positive | 1,745 (0.16%) |
| Normal | 1,088,427 |
| Imbalance ratio | 623.74 : 1 |
| Columns | 17 original |
| Payment types | 8 |
| Transaction categories | 11 |
| Simulation span | 195 days |

---

## Installation

Requires **Python 3.14.5**.

```bash
git clone https://github.com/Kulsoom-UEB/AMLNeT-Kulsoom-Dissertation.git
cd AMLNeT-Kulsoom-Dissertation

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Then place `AMLNet.csv` in `data/raw/` as described above.

---

## Running the pipeline

Run everything in sequence:

```bash
python run_all.py
```

Or run any stage independently — each writes its own outputs, so stages can be re-run without repeating the whole experiment:

```bash
python src/step00_setup.py                    # environment lock, seeds, dataset checksum
python src/step01_data_understanding.py       # profiling, class distribution, leakage audit
python src/step02_data_preparation.py         # leakage exclusion + cleaning
python src/step03_feature_engineering.py      # 11 engineered features
python src/step04_split.py                    # stratified 70/15/15 split (seed 42)
python src/step05_model_development.py        # 6 candidate pipelines + tuning
python src/step06_threshold_optimization.py   # threshold sweep + model selection
python src/step07_final_testing.py            # single evaluation on the test set
python src/step08_explainability.py           # built-in, permutation and SHAP
```

Launch the reviewer dashboard:

```bash
streamlit run src/app.py
```

---

## Repository structure

```
.
├── data/
│   ├── raw/                     AMLNet.csv  (not committed - download from Zenodo)
│   └── processed/               cleaned_amlnet.parquet, engineered_amlnet.parquet,
│                                X_train/X_val/X_test, y_train/y_val/y_test (.parquet)
├── src/
│   ├── amlnet_common.py         paths, seeds, column constants, metric helpers
│   ├── amlnet_resampling.py     custom SMOTENC pipeline (scale -> resample -> one-hot)
│   └── step00 ... step08        the nine numbered pipeline stages
├── models/
│   ├── final_model_pipeline.joblib   fitted pipeline (preprocessing + estimator)
│   └── final_model_config.json       frozen model, hyperparameters and threshold
├── outputs/
│   ├── tables/                  every result table as CSV
│   ├── figures/                 every figure as PNG
│   └── report_evidence/         consolidated evidence index
├── dashboard/
│   └── app.py                   Streamlit reviewer decision-support prototype
├── run_all.py                   single entry point for the full pipeline
└── requirements.txt             pinned package versions
```

---

## Methodology overview

A CRISP-DM informed experimental lifecycle across six phases.

**Leakage control.** Seven columns are excluded and the exclusion is enforced by a runtime assertion, so it cannot silently regress:

| Column | Reason |
|---|---|
| `laundering_typology` | Typology label directly reveals the target |
| `metadata` | Generator dictionary with embedded risk scores; also ~600 MB of the 691 MB file |
| `fraud_probability` | Generated risk score (mean 0.025 normal vs 0.405 laundering) |
| `isFraud` | Related generated label, 1:1 with the target |
| `nameOrig`, `nameDest` | High-cardinality account identifiers — memorisation risk |
| `step` | Documented as a sequential transaction step/ID, not a temporal feature; redundant given `hour`, `day_of_week`, `day_of_month`, `month` |

**Feature engineering.** Eleven features derived only from `amount`, `oldbalanceOrg` and `newbalanceOrig` — balance movement, amount-to-balance ratios, zero-balance indicators and log transforms — giving 20 model inputs.

**Imbalance handling.** Two paradigms compared under identical tuning: class weighting (`scale_pos_weight` = 623.996) and SMOTENC oversampling (`sampling_strategy` = 0.05, `k` = 5). SMOTENC rather than SMOTE because two features are nominal; the pipeline order is scale → integer-code → resample → one-hot so the majority-vote step sees real categories rather than dummy columns.

**Models.** Logistic Regression, Random Forest and XGBoost, each under both imbalance strategies, tuned with `RandomizedSearchCV` (6 candidates, 3-fold stratified CV, scored on `average_precision`).

**Selection.** Thresholds swept 0.01–0.99 on validation data with the F1-maximising point chosen per candidate; models then ranked by PR-AUC with ties broken by F1 and alert volume. Nothing is chosen by hand — the winner, its hyperparameters and its threshold are written to `models/final_model_config.json` and read by the testing, explainability and dashboard stages, so the deployed artefact is provably the one that was evaluated.

**Explainability.** Three complementary methods — built-in gain importance, permutation importance and SHAP TreeExplainer — cross-checked at feature-group level, plus local explanations for individual flagged transactions.

---

## Reproducibility

- Global seed `42` fixed across `random`, `numpy`, `PYTHONHASHSEED`, scikit-learn and XGBoost
- All package versions pinned in `requirements.txt`
- Dataset verified by MD5 before use
- Preprocessing fitted on training folds only; validation and test are transformed, never fitted
- Test set read exactly once, after model and threshold selection are complete
- Every figure and table regenerated programmatically from saved outputs

---

## Limitations

AMLNet is synthetic. Laundering-positive cases concentrate heavily in specific transaction categories by construction of the generator, so the model may partly be learning generator rules rather than behavioural signal — the explainability analysis exposes this rather than hiding it. The split is stratified random rather than chronological and account-disjoint, so it does not reproduce the deployment case where a model scores future transactions from unseen accounts. Probability calibration was not assessed, so the dashboard risk bands are ranking statements rather than verified likelihoods. The dashboard has not been evaluated with practising AML analysts.

**This is academic research, not a production AML system.** It does not replace human judgement or compliance sign-off.

---

## Citation

Dataset:

> Huda, S. (2025). *AMLNet: Synthetic anti-money laundering transaction dataset.* Zenodo. https://doi.org/10.5281/zenodo.16736515

---

## Licence

Code in this repository is provided for academic use. The AMLNet dataset is licensed CC BY-NC 4.0 and is used here strictly non-commercially with attribution to its author.
