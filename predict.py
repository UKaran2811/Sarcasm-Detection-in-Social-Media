"""Classify new text with the saved best model.
Usage:  python predict.py "Oh great, another Monday morning meeting!"
        python predict.py            (interactive demo with built-in examples)"""
import sys

import joblib
import pandas as pd

from src.features import ALL_FEATURES, add_features
from src.preprocess import preprocess_df

model = joblib.load("models/best_model.joblib")


def predict(texts):
    df = add_features(preprocess_df(pd.DataFrame({"id": range(len(texts)), "label": 0, "text": texts})))
    p = model.predict_proba(df[["clean"] + ALL_FEATURES])[:, 1]
    return [(t, float(s), "SARCASTIC/IRONIC" if s >= 0.5 else "not sarcastic") for t, s in zip(texts, p)]


if __name__ == "__main__":
    texts = sys.argv[1:] or [
        "Oh great, another Monday morning meeting at 8am. Just what I needed!",
        "Love waking up at 5am to a flat tyre #bestdayever",
        "Wow thanks so much for ignoring my message for three days, really appreciate it",
        "Our new office opens tomorrow at 9am. Come and say hello!",
        "Just finished a 10km run in the park, feeling fantastic",
        "Traffic report: accident on the M25 near junction 10, expect delays",
    ]
    for t, s, lab in predict(texts):
        print(f"{s:.2f}  {lab:17s} | {t}")
