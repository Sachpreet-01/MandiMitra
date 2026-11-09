import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

metadata = json.loads(Path("models/metadata.json").read_text(encoding="utf-8"))
features = metadata["features"]

X = pd.DataFrame([[0.0] * len(features)], columns=features)

models = [
    "lightgbm_q10.pkl",
    "lightgbm_q50.pkl",
    "lightgbm_q90.pkl",
    "lightgbm_regression.pkl",
    "xgboost_regression.pkl",
]

for filename in models:
    model = joblib.load(Path("models") / filename)
    prediction = np.asarray(model.predict(X)).reshape(-1)

    assert len(prediction) == 1
    assert np.isfinite(prediction).all()

    print(f"{filename}: prediction successful ({prediction[0]:.6f})")

print("\nAll five models passed the prediction smoke test.")
