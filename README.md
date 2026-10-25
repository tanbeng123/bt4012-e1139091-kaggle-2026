# BT4012 Kaggle Competition 2026 — Illicit Bitcoin Transaction Detection

Student: **e1139091**. Predicts the probability that each Bitcoin transaction in the test set is illicit, scored by ROC AUC.

## Final submissions (selected for the private leaderboard)

| File | Model | Public AUC |
| --- | --- | --- |
| `exp11c_seedavg5.csv` | Two-stage LightGBM: model B with `extra_trees`, 600 trees, learning rate 0.025, averaged over 5 seeds | 0.96925 |
| `exp11g_rankavg_11abc.csv` | Equal-weight rank average of three 5-seed variants (exp11a, exp11b, exp11c) | 0.96873 |

## How to reproduce

1. **Data.** Download `train.csv`, `test.csv`, `txs_edgelist.csv` and `sample_submission.csv` from the competition's Kaggle Data tab and place them in a `data/` folder at the repository root:

   ```text
   data/
   ├── train.csv
   ├── test.csv
   ├── txs_edgelist.csv
   └── sample_submission.csv
   ```

2. **Environment.** Python 3.13: `pip install -r requirements.txt`
3. **Run `e1139091.ipynb`** (Run All, about 18 minutes; set `RUN_VALIDATION = False` to skip validation, about 6 minutes). It writes both final files to `submissions/final/` and checks them against the submitted copies in `submissions/`.

All random seeds are fixed (model A: 42; model B: 42–46).

## Repository contents

| File | Purpose |
| --- | --- |
| `e1139091.ipynb` | **Final pipeline**: preprocessing, graph feature engineering, forward-in-time validation, and generation of both final submission files |
| `experiments.ipynb` | Full experiment history (Sections 1–17): every experiment behind the final choice, including rejected ones. Runs in about 70 minutes |
| `experiment_log.csv` | Validation AUCs (folds F1–F4) and public scores for every experiment |
| `tuning_results.csv`, `tuning_et_results.csv` | Cached hyperparameter-search results used by `experiments.ipynb` (delete to rerun the searches) |
| `submissions/` | Every file submitted to Kaggle, including the two finals (`exp11c_seedavg5.csv`, `exp11g_rankavg_11abc.csv`) |
| `figures/` | Report figures and the scripts that draw them (`make_report_figures.py` reads `data/train.csv` and `experiment_log.csv`) |
| `requirements.txt` | Python dependencies with exact versions |
