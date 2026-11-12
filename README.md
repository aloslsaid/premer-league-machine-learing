# Premier League Match Prediction

An experimental three-class classifier for English Premier League match outcomes: home win (H), draw (D), or away win (A). The repository contains multiple season CSV files and a maintained walk-forward feature pipeline.

## Features

For each fixture, rolling team statistics are calculated using matches completed before that fixture only: goals, expected goals, shots, shots on target, corners, fouls, cards and recent points/form. Current-match and post-match statistics are added to team history only after the feature row is created.

## Setup

Python 3.10+:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python train.py
```

The script reads `E0.csv` and any available numbered `E0 (N).csv` files from the repository root, filters the E0 division when the column exists, sorts chronologically and removes duplicate fixtures.

## Evaluation

The maintained pipeline uses chronological train/validation/test partitions. Candidate selection is performed on validation weighted F1; the selected pipeline is refitted on train + validation and evaluated once on the final test period. Team encoding and numeric imputation/scaling are fitted inside the model pipeline.

Metrics and the fitted model are saved in `artifacts/`. No result is claimed until the script has been run against the current checked-in data.

## Important limitations

Match prediction is uncertain. Check the data provider's license and season coverage. Ensure no bookmaker odds or post-match fields enter the feature set. The current pipeline uses historical match statistics only and is not betting advice.

## License

No license is declared. All rights reserved unless the repository owner adds an explicit license.
