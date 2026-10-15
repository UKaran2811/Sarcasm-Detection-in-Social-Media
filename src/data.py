"""Dataset download, loading and descriptive statistics.

Dataset: SemEval-2018 Task 3 (Subtask A) - Irony/sarcasm detection in English tweets.
Source : https://github.com/Cyvhee/SemEval2018-Task3  (Van Hee, Lefever & Hoste, 2018)
"""
import csv
import os
import re
import urllib.request

import pandas as pd

BASE = "https://raw.githubusercontent.com/Cyvhee/SemEval2018-Task3/master/datasets"
URLS = {
    "train": f"{BASE}/train/SemEval2018-T3-train-taskA_emoji.txt",
    "test": f"{BASE}/goldtest_TaskA/SemEval2018-T3_gold_test_taskA_emoji.txt",
}
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def download(force=False):
    os.makedirs(DATA_DIR, exist_ok=True)
    for name, url in URLS.items():
        path = os.path.join(DATA_DIR, f"{name}.txt")
        if force or not os.path.exists(path) or os.path.getsize(path) < 1000:
            print(f"Downloading {name} -> {path}")
            urllib.request.urlretrieve(url, path)


def load(split):
    path = os.path.join(DATA_DIR, f"{split}.txt")
    if not os.path.exists(path):
        download()
    df = pd.read_csv(path, sep="\t", quoting=csv.QUOTE_NONE, encoding="utf-8")
    df.columns = ["id", "label", "text"]
    df["label"] = df["label"].astype(int)
    return df.reset_index(drop=True)


def describe(df, name):
    """Return a dict of dataset characteristics (used in the report)."""
    toks = df["text"].str.split()
    vocab = set(w.lower() for t in toks for w in t)
    return {
        "split": name,
        "n_tweets": len(df),
        "n_ironic/sarcastic (1)": int(df.label.sum()),
        "n_non-ironic (0)": int((df.label == 0).sum()),
        "% positive class": round(100 * df.label.mean(), 1),
        "avg chars": round(df.text.str.len().mean(), 1),
        "avg tokens": round(toks.str.len().mean(), 1),
        "vocab size (raw)": len(vocab),
        "% with hashtag": round(100 * df.text.str.contains(r"#\w+").mean(), 1),
        "% with @mention": round(100 * df.text.str.contains(r"@\w+").mean(), 1),
        "% with URL": round(100 * df.text.str.contains(r"http\S+").mean(), 1),
        "% with emoji": round(100 * df.text.apply(lambda s: bool(re.search(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]", s))).mean(), 1),
    }
