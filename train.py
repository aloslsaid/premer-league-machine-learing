from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
FILES = ["E0.csv", "E0 (2).csv", "E0 (3).csv", "E0 (4).csv", "E0 (5).csv"]
OUTPUT_DIR = ROOT / "artifacts"
SEED = 42
STAT_MAP = {
    "GoalsFor": ("FTHG", "FTAG"), "GoalsAgainst": ("FTAG", "FTHG"),
    "xG": ("HxG", "AxG"), "Shots": ("HS", "AS"),
    "ShotsOnTarget": ("HST", "AST"), "Corners": ("HC", "AC"),
    "Fouls": ("HF", "AF"), "YellowCards": ("HY", "AY"),
    "RedCards": ("HR", "AR"),
}


def load_matches():
    frames = []
    for name in FILES:
        path = ROOT / name
        if not path.is_file():
            continue
        frame = pd.read_csv(path)
        if "Div" in frame:
            frame = frame.loc[frame["Div"] == "E0"]
        frames.append(frame)
    if not frames:
        raise FileNotFoundError("No E0 season CSV files were found.")
    df = pd.concat(frames, ignore_index=True)
    required = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
    missing = [c for c in required if c not in df]
    if missing:
        raise ValueError(f"Missing required match columns: {missing}")
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    for col in set(sum((list(v) for v in STAT_MAP.values()), [])):
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"])
    df = df[df["FTR"].isin(["H", "D", "A"])]
    return df.drop_duplicates(["Date", "HomeTeam", "AwayTeam"]).sort_values("Date").reset_index(drop=True)


def build_walk_forward_features(matches):
    history = defaultdict(lambda: {key: deque(maxlen=5) for key in STAT_MAP} | {"Points": deque(maxlen=5)})
    rows = []
    for _, match in matches.iterrows():
        home, away = match["HomeTeam"], match["AwayTeam"]
        features = {"HomeTeam": home, "AwayTeam": away}
        for key in STAT_MAP:
            h_col, a_col = STAT_MAP[key]
            h_value = history[home][key]
            a_value = history[away][key]
            features[f"Home_{key}_Last5"] = float(np.mean(h_value)) if h_value else 0.0
            features[f"Away_{key}_Last5"] = float(np.mean(a_value)) if a_value else 0.0
            features[f"{key}_Diff_Last5"] = features[f"Home_{key}_Last5"] - features[f"Away_{key}_Last5"]
        features["Home_Form_Last5"] = float(sum(history[home]["Points"]))
        features["Away_Form_Last5"] = float(sum(history[away]["Points"]))
        features["Form_Diff_Last5"] = features["Home_Form_Last5"] - features["Away_Form_Last5"]
        features["Target"] = match["FTR"]
        rows.append(features)

        hg, ag = float(match["FTHG"]), float(match["FTAG"])
        for key, (h_col, a_col) in STAT_MAP.items():
            h_val = match.get(h_col, np.nan)
            a_val = match.get(a_col, np.nan)
            if pd.isna(h_val): h_val = hg if key == "GoalsFor" else (ag if key == "GoalsAgainst" else 0.0)
            if pd.isna(a_val): a_val = ag if key == "GoalsFor" else (hg if key == "GoalsAgainst" else 0.0)
            history[home][key].append(float(h_val))
            history[away][key].append(float(a_val))
        if match["FTR"] == "H":
            hp, ap = 3, 0
        elif match["FTR"] == "A":
            hp, ap = 0, 3
        else:
            hp = ap = 1
        history[home]["Points"].append(hp)
        history[away]["Points"].append(ap)
    return pd.DataFrame(rows)


def make_pipeline(estimator, categorical, numeric):
    prep = ColumnTransformer([
        ("teams", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("numbers", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
    ])
    return Pipeline([("preprocess", prep), ("model", estimator)])


def main():
    matches = load_matches()
    data = build_walk_forward_features(matches)
    if len(data) < 60:
        raise ValueError(f"Only {len(data)} feature rows available; at least 60 are required.")
    X, y = data.drop(columns=["Target"]), data["Target"]
    n = len(data)
    train_end, val_end = int(n * 0.70), int(n * 0.85)
    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
    X_trainval, y_trainval = X.iloc[:val_end], y.iloc[:val_end]
    X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]
    categorical = ["HomeTeam", "AwayTeam"]
    numeric = [c for c in X.columns if c not in categorical]
    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=3000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=400, max_depth=10, min_samples_leaf=3, class_weight="balanced", random_state=SEED, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=250, learning_rate=0.05, max_depth=3, random_state=SEED),
    }
    ranked = []
    for name, estimator in candidates.items():
        model = make_pipeline(estimator, categorical, numeric)
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        ranked.append((f1_score(y_val, pred, average="weighted", zero_division=0), name, estimator))
    ranked.sort(key=lambda x: x[0], reverse=True)
    best_val_f1, best_name, best_estimator = ranked[0]
    final_model = make_pipeline(best_estimator, categorical, numeric)
    final_model.fit(X_trainval, y_trainval)
    test_pred = final_model.predict(X_test)
    metrics = {
        "selected_model": best_name,
        "validation_weighted_f1": float(best_val_f1),
        "test_accuracy": float(accuracy_score(y_test, test_pred)),
        "test_weighted_f1": float(f1_score(y_test, test_pred, average="weighted", zero_division=0)),
        "test_rows": int(len(y_test)),
        "feature_rows": int(len(data)),
        "date_start": str(matches["Date"].min().date()),
        "date_end": str(matches["Date"].max().date()),
        "split": {"train": train_end, "validation_end": val_end, "test": n - val_end},
    }
    OUTPUT_DIR.mkdir(exist_ok=True)
    joblib.dump(final_model, OUTPUT_DIR / "premier_league_model.joblib")
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(classification_report(y_test, test_pred, labels=["H", "D", "A"], zero_division=0))


if __name__ == "__main__":
    main()
