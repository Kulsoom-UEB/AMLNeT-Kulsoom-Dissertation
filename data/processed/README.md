# data/processed/

**This folder is empty in the repository by design.**

Every file that belongs here is *generated* by running the pipeline. None of them
are committed, for two reasons: they are reproducible outputs rather than source
material, and several exceed GitHub's file-size limits.

## What appears here after running the pipeline

| File | Created by | Contents |
|---|---|---|
| `cleaned_amlnet.parquet` | `src/step02_data_preparation.py` | 1,090,172 rows after leakage-column removal and cleaning |
| `engineered_amlnet.parquet` | `src/step03_feature_engineering.py` | The same rows with the 11 engineered features added (20 model inputs) |
| `X_train`, `X_val`, `X_test` | `src/step04_split.py` | Stratified 70/15/15 feature splits, seed 42 |
| `y_train`, `y_val`, `y_test` | `src/step04_split.py` | Matching target splits |
| `test_predictions.parquet` | `src/step07_final_testing.py` | Per-transaction risk scores, risk bands and predicted labels for the test set |

## Generating them

From the project root, with the dataset in place at `data/raw/AMLNet.csv`:

```bash
python src/run_all.py
```

This runs steps 00 to 08 in order and takes roughly 25–35 minutes.

## About `test_predictions.parquet` and the dashboard

The Streamlit dashboard (`src/app.py`) reads this file to populate its **Alert
queue** and **Case review** tabs. Without it those two tabs are empty, but the
remaining five — Overview, Explainability, Risk profile, Performance and
Limitations — read the committed CSV tables in `outputs/tables/` and work
immediately after cloning.

A **recorded walkthrough of the complete dashboard**, with all tabs populated, is
available here: [Dashboard demonstration video](https://drive.google.com/file/d/10jZbtqbLWENUH5QFAOESdcQHTcB09NLs/view?usp=drive_link)

## Verifying results without running anything

Everything reported in the paper and supporting material is committed under
`outputs/`:

- `outputs/tables/` — 58 CSV files covering profiling, leakage audit, splits,
  hyperparameter search, validation comparison, final test metrics and
  explainability
- `outputs/figures/` — the 13 figures used in the write-up

So the results can be checked directly from this repository; regenerating them is
optional.
