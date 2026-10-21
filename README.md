# BT4012 Kaggle Competition 2026 — Illicit Bitcoin Transaction Detection

Predicts the probability that each Bitcoin transaction in the test set is illicit, scored by ROC AUC.

## Repository contents

| File | Purpose |
| --- | --- |
| `bt4012_solution.ipynb` | End-to-end notebook: data loading, validation, modelling, submission file generation |
| `requirements.txt` | Python dependencies (tested on Python 3.13) |
| `experiment_log.csv` | Cross-validation results for each experiment |
| `submissions/` | Generated submission files |

## How to reproduce

1. Download `train.csv`, `test.csv`, `txs_edgelist.csv` and `sample_submission.csv` from the competition's Kaggle Data tab and place them in the repository root, next to the notebook. They are not committed because of file size limits.
2. Install dependencies: `pip install -r requirements.txt`
3. Run all cells of the notebook. The submission file is written to `submissions/`.

All random seeds are fixed (`SEED = 42`).
