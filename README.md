# Premier League Match Outcome Prediction

> A time-aware football analytics experiment that uses historical match statistics and recent team form to classify match outcomes.

## Project goal
The project explores whether a team's recent performance contains useful information for predicting the outcome of an English Premier League fixture. Instead of treating each match as an isolated row, the feature-building process walks through matches chronologically and maintains rolling team history.

## Data pipeline
- Combines multiple season CSV files (`E0.csv` and the numbered copies).
- Filters the English Premier League division.
- Parses match dates and sorts fixtures chronologically.
- Converts match statistics to numeric values and handles selected missing expected-goals fields.
- Maintains separate historical records for each team.
- Builds recent-form features from the previous five matches, including goals for/against, expected goals, shots, shots on target, corners, fouls, cards, and recent results.
- Uses home-team and away-team history to construct a match-level feature set.
- Trains classification models to predict the full-time result.

## Models and evaluation
The source imports and uses Scikit-learn classification tools, including Logistic Regression and tree-based ensemble models. Evaluation uses Accuracy, Precision, Recall, F1-score, and a classification report. These metrics provide different views of performance, particularly when home wins, draws, and away wins are not evenly distributed.

## Install
Python 3.10+ is recommended.
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
python -m pip install --upgrade pip
pip install pandas numpy scikit-learn
```

## Run
Run from the repository root so the five season CSV paths resolve:
```bash
python "1 match"
```
The training file has no extension. Keep the season CSV files beside it unless you update the paths in your own working copy.

## Important methodology note
Chronological feature construction is essential: a fixture must only use information available before that fixture. This project is an experimental prediction model, not a betting system. Football outcomes remain uncertain, and historical performance does not guarantee future accuracy. Re-evaluate on later seasons before drawing conclusions.

## Repository contents
- `1 match` — Python feature-engineering and classification script.
- `E0.csv`, `E0 (2).csv` … `E0 (5).csv` — season-level match data files.

## Skills demonstrated
Sports analytics · Time-aware feature engineering · Rolling statistics · Multi-class classification · Historical data processing · Model evaluation
