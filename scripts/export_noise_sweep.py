"""Fit the four models at several noise levels and export the curves.

The page needs to let a reader raise the noise and watch each model react, so
every fit is computed here once and shipped as JSON rather than described in
prose. Same data generation as the notebook: f1 on [-20, 20], 100 points, a
70/30 split at seed 42, and the five engineered feature columns.
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
OUT = Path(__file__).resolve().parents[1] / "assets" / "noise-sweep.json"


def f1(x):
    return x * np.sin(x) + 2 * x


def features(x):
    return np.column_stack([x, np.sin(x), np.cos(x), x ** 2, x ** 3])


X = np.linspace(-20, 20, 100)
y_clean = f1(X)
X_tr, X_te, y_tr_clean, y_te_clean = train_test_split(
    X, y_clean, train_size=0.7, random_state=42, shuffle=True)

grid = np.linspace(-20, 20, 240)          # dense line for drawing
G = features(grid)

LEVELS = [0, 5, 10, 20, 35, 50, 75, 100]
MODELS = {
    "Linear":     lambda: LinearRegression(),
    "Polynomial": lambda: make_pipeline(PolynomialFeatures(3), LinearRegression()),
    "Ridge":      lambda: make_pipeline(PolynomialFeatures(3), Ridge(alpha=1.0)),
    "Forest":     lambda: RandomForestRegressor(n_estimators=100, random_state=42),
}

payload = {
    "truth": [[round(float(a), 3), round(float(b), 3)] for a, b in zip(grid, f1(grid))],
    "levels": [],
}

for sigma in LEVELS:
    rng = np.random.default_rng(42)
    noise_tr = rng.normal(0, sigma, size=y_tr_clean.size) if sigma else np.zeros_like(y_tr_clean)
    noise_te = rng.normal(0, sigma, size=y_te_clean.size) if sigma else np.zeros_like(y_te_clean)
    y_tr, y_te = y_tr_clean + noise_tr, y_te_clean + noise_te

    scaler = StandardScaler().fit(features(X_tr))
    entry = {
        "sigma": sigma,
        "train": [[round(float(a), 3), round(float(b), 3)] for a, b in zip(X_tr, y_tr)],
        "models": {},
    }
    for name, make in MODELS.items():
        model = make()
        if name == "Forest":
            model.fit(features(X_tr), y_tr)
            curve = model.predict(G)
            pred = model.predict(features(X_te))
        else:
            model.fit(scaler.transform(features(X_tr)), y_tr)
            curve = model.predict(scaler.transform(G))
            pred = model.predict(scaler.transform(features(X_te)))
        entry["models"][name] = {
            "curve": [round(float(v), 3) for v in curve],
            "r2": round(float(r2_score(y_te, pred)), 3),
        }
    payload["levels"].append(entry)
    scores = " ".join(f"{n}={entry['models'][n]['r2']:>7.3f}" for n in MODELS)
    print(f"  sigma {sigma:>3}: {scores}")

payload["grid"] = [round(float(v), 3) for v in grid]
OUT.write_text(json.dumps(payload), encoding="utf-8")
print(f"\nwrote {OUT.name} ({OUT.stat().st_size / 1024:.0f} KB)")
