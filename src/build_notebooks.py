"""
Generate Colab-ready Jupyter notebooks from the src/step*.py modules.
Each notebook carries the CRISP-DM narrative in markdown, installs its
dependencies for Colab, and calls the corresponding step function, so the
notebook and script versions can never drift apart.
"""
from __future__ import annotations
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
NB = ROOT / "notebooks"
NB.mkdir(parents=True, exist_ok=True)
TITLE = ("Explainable Machine Learning for Suspicious Transaction Risk "
         "Detection in Anti-Money Laundering")
COLAB_SETUP = '''# --- Colab / local setup -------------------------------------------------
# Installs dependencies and locates the project. Safe to re-run.
import importlib.util, subprocess, sys
from pathlib import Path
if importlib.util.find_spec("google.colab"):
    subprocess.run([sys.executable, "-m", "pip", "-q", "install",
                    "imbalanced-learn", "xgboost", "shap", "pyarrow", "plotly"], check=False)
    from google.colab import drive          # noqa: F401
    # drive.mount('/content/drive')         # uncomment to persist outputs
CANDIDATES = [Path.cwd() / "AMLNet_Project", Path.cwd(), Path.cwd().parent,
              Path("/content/AMLNet_Project"),
              Path("/content/drive/MyDrive/AMLNet_Project")]
PROJECT_ROOT = next((p for p in CANDIDATES if (p / "src" / "amlnet_common.py").exists()),
                    Path.cwd())
sys.path.append(str(PROJECT_ROOT / "src"))
import amlnet_common as C
C.set_seeds(); C.ensure_dirs()
print("Project root:", C.PROJECT_ROOT)
print("Dataset present:", C.RAW_DATA_PATH.exists())
'''
DATA_CELL = '''# --- Obtain the AMLNet dataset (run once) --------------------------------
# 691 MB. Skips the download if the file is already present.
import subprocess
if not C.RAW_DATA_PATH.exists():
    C.DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    url = ("https://zenodo.org/records/16736515/files/"
           "AMLNet_August%202025.csv?download=1")
    print("Downloading AMLNet (~691 MB)...")
    subprocess.run(["curl", "-L", "--retry", "5", "-o",
                    str(C.RAW_DATA_PATH), url], check=True)
print("Dataset ready:", C.RAW_DATA_PATH.exists())
'''
SPEC = [
    ("00_project_setup", "step00_setup", "Project Setup and Reproducibility",
     "Project setup",
     "Creates the folder structure, fixes the random seed, verifies the raw "
     "AMLNet dataset and records the software environment.",
     ["Deterministic seeding across `random`, `numpy` and `PYTHONHASHSEED`",
      "Dataset presence and size checked against the Zenodo record",
      "Package versions captured for the reproducibility appendix"]),
    ("01_data_audit", "step01_data_understanding",
     "Data Audit and Analysis", "Data Understanding",
     "Audits shape, dtypes, missingness, duplicates, class balance, "
     "categorical levels, identifier cardinality and temporal coverage.",
     ["1,090,172 transactions; 1,745 laundering-positive (0.16 %), ratio ~1:624",
      "Evidence gathered for the leakage decisions taken in step 02",
      "`metadata` is sampled, not fully loaded (~600 MB of the 691 MB file)"]),
    ("02_leakage_removal_cleaning", "step02_data_preparation",
     "Leakage Removal and Cleaning", "Data Preparation",
     "Removes leakage-prone and identifier fields, documents missing-value "
     "handling and duplicates, and writes the cleaned modelling dataset.",
     ["Excluded for leakage: `laundering_typology`, `metadata`, "
      "`fraud_probability`, `isFraud`",
      "Excluded as identifiers: `nameOrig`, `nameDest`",
      "Excluded from modelling: `step` — the simulation time index, not a "
      "meaningful transaction attribute for suspicious transaction risk prediction",
      "Missing numerics imputed inside the training-fitted pipeline, not up front"]),
    ("03_feature_engineering", "step03_feature_engineering",
     "Feature Engineering", "Data Preparation",
     "Creates 11 engineered features from amount and balance fields only.",
     ["Balance movement, amount-to-balance ratios, zero-balance flags, log transforms",
      "No feature derived from any excluded leakage or identifier field",
      "Validated for missing and infinite values"]),
    ("04_split", "step04_split",

     "Train / Validation / Test Split", "Data Preparation",
     "Stratified 70/15/15 split with integrity assertions. Encoding and "
     "scaling are designed to be fitted inside pipelines only.",
     ["Train 763,120 (1,221 positives) · Validation 163,526 (262) · Test 163,526 (262)",
      "No index overlap; stratification preserved in all three subsets",
      "Note: stratified random split, not chronological — see Limitations"]),
    ("05_model_development", "step05_model_development",
     "Model Development and Hyperparameter Tuning", "Modelling",
     "Three algorithms (Logistic Regression, Random Forest, XGBoost) trained "
     "under two imbalance strategies, with RandomizedSearchCV tuning scored by "
     "PR-AUC on the training set only.",
     ["Class weighting: scaling + one-hot **before** the model",
      "SMOTENC: scaling → SMOTENC → one-hot **after** resampling",
      "SMOTENC (not plain SMOTE) because `type` and `category` are categorical",
      "LightGBM removed from project scope",
      "Six tuned model/strategy combinations plus a majority-class benchmark"]),
    ("06_threshold_optimization", "step06_threshold_optimization",
     "Best-Model Selection and Threshold Optimisation", "Evaluation",
     "Optimises the decision threshold for every candidate on validation data, "
     "then selects the best model automatically from the metrics.",
     ["Selection rule: highest PR-AUC, ties broken by F1 then alert volume",
      "Threshold optimised over a 0.01-step grid by maximising validation F1",
      "**Nothing is hardcoded** — the winner is discovered, then frozen to "
      "`models/final_model_config.json`",
      "Test set remains untouched"]),
    ("07_final_testing", "step07_final_testing",
     "Final Test-Set Evaluation", "Evaluation",
     "Applies the frozen model and threshold to the untouched test set exactly "
     "once. No tuning happens here.",
     ["Headline metrics, confusion matrix, classification report",
      "Validation-versus-test comparison as an overfitting check",
      "Reviewer workload framing and per-transaction predictions"]),
    ("08_explainability", "step08_explainability",
     "Explainability: SHAP, Feature and Permutation Importance", "Evaluation",
     "Explains the selected model globally and locally. Works with either "
     "pipeline architecture; one-hot columns are aggregated back to feature groups.",
     ["Built-in importance, permutation importance and SHAP all reported",
      "Local SHAP for true positives, false positives and false negatives",
      "Explanations are model attributions, not evidence of money laundering"]),
]
DASHBOARD_MD = """## Notebook 09: Streamlit Reviewer-Support Dashboard
**CRISP-DM stage:** Deployment (prototype)
The dashboard is a Streamlit application (`dashboard/app.py`) rather than
notebook code, because a notebook cannot demonstrate an interactive reviewer
workflow. It reads **only saved outputs** from steps 06–08, so it opens
instantly and always displays exactly the numbers reported in the paper.
### Launch
```bash
streamlit run dashboard/app.py
```
In Colab, expose it with a tunnel:
```python
!pip -q install streamlit
!streamlit run dashboard/app.py &>/dev/null &
!npx -y localtunnel --port 8501
```
### Tabs
* **Alert queue** — transactions ranked by risk score with an adjustable threshold
* **Case review** — transaction detail plus local SHAP contributions
* **Explainability** — global SHAP, permutation and built-in importance
* **Performance** — final test metrics, automatic selection table, tuning results
* **Limitations** — synthetic-data and non-temporal-split caveats
### Responsible use
Every screen states that output is decision support. A high risk score means the
transaction resembles patterns learned from synthetic data — not evidence, not
an accusation, not a legal determination.
"""
def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.splitlines(True)}
def code(s):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": s.splitlines(True)}
def write(nb, path):
    nb = {"cells": nb,
          "metadata": {"kernelspec": {"display_name": "Python 3",
                                      "language": "python", "name": "python3"},
                       "language_info": {"name": "python", "version": "3.11"},

                       "colab": {"provenance": [], "toc_visible": True}},
          "nbformat": 4, "nbformat_minor": 5}
    path.write_text(json.dumps(nb, indent=1))
def main():
    for i, (fname, mod, title, stage, purpose, points) in enumerate(SPEC):
        num = fname[:2]
        header = (f"# Notebook {num}: {title}\n\n"
                  f"**Project:** {TITLE}\n\n"
                  f"**CRISP-DM stage:** {stage}\n\n"
                  f"**Dataset:** AMLNet (Huda et al., 2025), Zenodo "
                  f"[10.5281/zenodo.16736515](https://doi.org/10.5281/zenodo.16736515)\n\n"
                  f"### Purpose\n\n{purpose}\n\n### Key points\n\n"
                  + "\n".join(f"* {p}" for p in points) + "\n")
        cells = [md(header), code(COLAB_SETUP)]
        if num in ("00", "01"):
            cells.append(code(DATA_CELL))
        cells.append(code(f"import {mod} as step\nstep.main()\n"))
        cells.append(code(
            "# Review the evidence tables this stage produced\n"
            "import pandas as pd\n"
            "from IPython.display import display\n\n"
            "latest = sorted(C.TABLES_DIR.glob('*.csv'),\n"
            "                key=lambda p: p.stat().st_mtime)[-6:]\n"
            "for p in latest:\n"
            "    print('\\n' + '=' * 70); print(p.name); print('=' * 70)\n"
            "    display(pd.read_csv(p).head(15))\n"))
        if num in ("01", "06", "07", "08"):
            pat = {"01": "fig0[123]", "06": "fig0[3456]",
                   "07": "fig0[78]", "08": "fig1[0-2]|fig09"}[num]
            cells.append(code(
                "# Display this stage's figures\n"
                "import re\n"
                "from IPython.display import Image, display\n\n"
                f"pat = re.compile(r'{pat}')\n"
                "for f in sorted(C.FIGURES_DIR.glob('*.png')):\n"
                "    if pat.match(f.name):\n"
                "        print(f.name); display(Image(str(f)))\n"))
        cells.append(md("### Stage complete\n\nEvidence tables are in "
                        "`outputs/tables/` and figures in `outputs/figures/`.\n"))
        write(cells, NB / f"{fname}.ipynb")
        print(f"wrote notebooks/{fname}.ipynb")
    cells = [md(DASHBOARD_MD), code(COLAB_SETUP), code(
        "# Verify the dashboard's inputs exist\n"
        "print('Model      :', C.FINAL_MODEL_PATH.exists())\n"
        "print('Config     :', C.FINAL_CONFIG_PATH.exists())\n"
        "print('Predictions:', (C.DATA_PROCESSED_DIR / 'test_predictions.parquet').exists())\n"
        "print('\\nLaunch: streamlit run dashboard/app.py')\n")]
    write(cells, NB / "09_reviewer_dashboard.ipynb")
    print("wrote notebooks/09_reviewer_dashboard.ipynb")
if __name__ == "__main__":
    main()
