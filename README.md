# Premier League Match Prediction

An experimental machine-learning project for predicting English Premier League match outcomes from historical match data and recent team performance. The project description mentions rolling statistics such as goals, expected goals, shots, corners, cards, and form. The repository contains several season CSVs and a file named `1 match`; inspect these files and their provenance before use.

## Data and setup

- CSV files at the root: `E0.csv` and `E0 (2).csv` through `E0 (5).csv`.
- Confirm season dates, column names, data-source terms, and schema consistency before combining files.
- The root does not list a Python dependency manifest or a verified executable command. Inspect the project file (`1 match`) and its contents to determine the intended environment and workflow.

## Reproducible evaluation

Use only information available before each fixture. Calculate rolling team features from prior matches, sort by date, and use a chronological train/validation/test split. Do not report a model score without its data version, split, baseline, and metric definitions. No test suite or reported result is reproduced by this README.

This is an experimental sports prediction project, not betting advice. Historical patterns do not guarantee future outcomes.


## Reproducible run

The extensionless Python script is `1 match`. Run from the repository root; it reads `E0.csv` and `E0 (2).csv` through `E0 (5).csv`.

```bash
python -m venv .venv
# Activate the environment, then:
python -m pip install --upgrade pip
pip install -r requirements.txt
python "1 match"
```

Keep the season files together. Verify each CSV's source, season, and schema before replacing or combining datasets. Preserve chronological ordering and use only information available before each fixture when validating.
