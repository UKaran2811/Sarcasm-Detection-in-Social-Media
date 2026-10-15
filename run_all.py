"""End-to-end pipeline:  data -> preprocessing -> features -> baseline -> improved ->
evaluation -> ablation -> significance -> error analysis.   Run:  python run_all.py
All tables are written to results/, all figures to results/figures/."""
import json
import os
import time
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score

from src import data, models
from src.evaluate import bootstrap_f1_ci, mcnemar_exact, metrics, scores
from src.features import ALL_FEATURES, FEATURE_GROUPS, add_features
from src.preprocess import preprocess_df

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.abspath(__file__))
RES, FIG = os.path.join(ROOT, "results"), os.path.join(ROOT, "results", "figures")
os.makedirs(FIG, exist_ok=True)
os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)
sns.set_theme(style="whitegrid", context="notebook")
CV = StratifiedKFold(5, shuffle=True, random_state=models.SEED)
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.1f}s]", *a, flush=True)


def md(df, path, floatfmt=".4f"):
    with open(path, "w") as f:
        f.write(df.to_markdown(floatfmt=floatfmt))


# =========================================================== 1. DATA
log("1. Loading data")
data.download()
raw_tr, raw_te = data.load("train"), data.load("test")
stats = pd.DataFrame([data.describe(raw_tr, "train"), data.describe(raw_te, "test")]).set_index("split").T
stats.to_csv(f"{RES}/dataset_stats.csv"); md(stats, f"{RES}/dataset_stats.md", ".1f")
print(stats)

# =========================================================== 2. PREPROCESS + FEATURES
log("2. Preprocessing and feature extraction")
tr = add_features(preprocess_df(raw_tr))
te = add_features(preprocess_df(raw_te))
tr_sw = preprocess_df(raw_tr, remove_stopwords=True)     # for the stop-word ablation
te_sw = preprocess_df(raw_te, remove_stopwords=True)
tr.to_csv(f"{RES}/train_processed.csv", index=False); te.to_csv(f"{RES}/test_processed.csv", index=False)
print(tr[["text", "clean"]].head(5).to_string())
COLS = ["clean"] + ALL_FEATURES
Xtr, ytr, Xte, yte = tr[COLS], tr.label.values, te[COLS], te.label.values

# dataset figure
fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
for a, (d, n) in zip(ax[:2], [(raw_tr, "Train"), (raw_te, "Test")]):
    sns.countplot(x=d.label.map({0: "Non-ironic", 1: "Ironic/Sarcastic"}), ax=a, palette=["#4C72B0", "#DD8452"])
    a.set_title(f"{n} label distribution (n={len(d)})"); a.set_xlabel("")
for k, n in [(1, "Ironic/Sarcastic"), (0, "Non-ironic")]:
    sns.kdeplot(raw_tr[raw_tr.label == k].text.str.split().str.len(), ax=ax[2], label=n, fill=True, alpha=.3)
ax[2].set_title("Tweet length (tokens), train"); ax[2].set_xlim(0, 45); ax[2].legend()
plt.tight_layout(); plt.savefig(f"{FIG}/01_dataset_overview.png", dpi=150); plt.close()

# =========================================================== 3. MODELS
log("3. Training / tuning (5-fold stratified CV on the training set, scoring = F1 of positive class)")
w_grid = [{"ctx": w} for w in (0.15, 0.3, 0.6, 1.0)]
specs = {
    "B1: TF-IDF(word) + Naive Bayes": (models.baseline_nb(), {"clf__alpha": [0.1, 0.3, 0.5, 1.0]}),
    "B2: TF-IDF(word) + LogReg": (models.baseline_lr(), {"clf__C": [0.3, 1, 3, 10]}),
    "I1: Context-only + GradBoost": (models.context_only_hgb(), {"clf__max_iter": [100, 200], "clf__max_depth": [2, 3]}),
    "I2: TF-IDF(word+char)+Context + LogReg": (models.improved_lr(),
                                               {"clf__C": [0.3, 1, 3, 10], "feat__transformer_weights": w_grid}),
    "I3: TF-IDF(word+char)+Context + LinearSVM": (models.improved_svm(),
                                                  {"clf__C": [0.03, 0.1, 0.3, 1], "feat__transformer_weights": w_grid}),
}
fitted, rows, cv_rows, preds, sc = {}, [], [], {}, {}
for name, (pipe, grid) in specs.items():
    gs = GridSearchCV(pipe, grid, scoring="f1", cv=CV, n_jobs=-1, refit=True).fit(Xtr, ytr)
    fitted[name] = gs.best_estimator_
    log(f"   {name}: CV-F1={gs.best_score_:.4f}  best={gs.best_params_}")
    cv_rows.append({"Model": name, "CV F1 (train, 5-fold)": gs.best_score_, "Best params": str(gs.best_params_)})

# Late-fusion (text LR + context GB), using tuned components
c_best = fitted["I2: TF-IDF(word+char)+Context + LogReg"].named_steps["clf"].C
fusion = models.LateFusion(models.improved_lr(C=c_best, ctx_cols=None), models.context_only_hgb(), w=0.6).fit(Xtr, ytr)
name = "I4: Late fusion (Text-LR + Context-GB)"
fitted[name] = fusion
cv_rows.append({"Model": name, "CV F1 (train, 5-fold)": cross_val_score(
    models.LateFusion(models.improved_lr(C=c_best, ctx_cols=None), models.context_only_hgb(), w=0.6),
    Xtr, ytr, cv=CV, scoring="f1", n_jobs=-1).mean(), "Best params": f"C={c_best}, w=0.6"})

for name, m in fitted.items():
    preds[name], sc[name] = m.predict(Xte), scores(m, Xte)
    r = metrics(yte, preds[name], sc[name]); lo, hi = bootstrap_f1_ci(yte, preds[name])
    rows.append({"Model": name, **r, "F1 95% CI": f"[{lo:.3f}, {hi:.3f}]"})
res = pd.DataFrame(rows).set_index("Model")
cv_df = pd.DataFrame(cv_rows).set_index("Model")
res = res.join(cv_df[["CV F1 (train, 5-fold)"]])
res.to_csv(f"{RES}/model_comparison.csv"); md(res.drop(columns=[]), f"{RES}/model_comparison.md")
cv_df.to_csv(f"{RES}/cv_and_best_params.csv")
print(res.round(4).to_string())

# comparison bar chart
plot = res[["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]].copy()
plot.index = [i.split(":")[0] for i in plot.index]
ax = plot.plot(kind="bar", figsize=(10, 4.5), width=.8, colormap="viridis"); ax.set_ylim(0.4, 0.9)
ax.set_title("Test-set performance (SemEval-2018 Task 3A gold test, n=784)"); plt.xticks(rotation=0)
plt.legend(ncol=5, loc="upper center", fontsize=8); plt.tight_layout(); plt.savefig(f"{FIG}/02_model_comparison.png", dpi=150); plt.close()

# ROC + confusion matrices
best = res["F1"].idxmax(); log(f"   Best model on test (F1): {best}")
fig, ax = plt.subplots(figsize=(5.5, 5))
for n_ in fitted:
    fpr, tpr, _ = roc_curve(yte, sc[n_]); ax.plot(fpr, tpr, label=f"{n_.split(':')[0]} (AUC={res.loc[n_, 'ROC-AUC']:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=.8); ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
ax.set_title("ROC curves (test)"); ax.legend(fontsize=8); plt.tight_layout(); plt.savefig(f"{FIG}/03_roc.png", dpi=150); plt.close()
fig, ax = plt.subplots(1, 2, figsize=(9, 3.8))
for a, n_ in zip(ax, ["B2: TF-IDF(word) + LogReg", best]):
    ConfusionMatrixDisplay.from_predictions(yte, preds[n_], display_labels=["Non-iro", "Ironic"], cmap="Blues", ax=a, colorbar=False)
    a.set_title(n_.split(":")[0] + " - " + n_.split(":")[1].strip()[:32], fontsize=9)
plt.tight_layout(); plt.savefig(f"{FIG}/04_confusion_matrices.png", dpi=150); plt.close()

# =========================================================== 4. SIGNIFICANCE
log("4. McNemar significance tests (vs. baseline B2)")
sig = []
for n_ in fitted:
    if n_ != "B2: TF-IDF(word) + LogReg":
        b, c, p = mcnemar_exact(yte, preds["B2: TF-IDF(word) + LogReg"], preds[n_])
        sig.append({"Model vs B2": n_, "B2 right/model wrong": b, "B2 wrong/model right": c, "p-value": p})
sig = pd.DataFrame(sig).set_index("Model vs B2"); sig.to_csv(f"{RES}/mcnemar.csv"); md(sig, f"{RES}/mcnemar.md", ".4f"); print(sig)

# =========================================================== 5. ABLATION
log("5. Ablation study (best LR hyper-parameters, 5-fold CV on train + test)")
bp = fitted["I2: TF-IDF(word+char)+Context + LogReg"].get_params()
C_, W_ = bp["clf__C"], bp["feat__transformer_weights"]["ctx"]
G = FEATURE_GROUPS
abl_cfg = {
    "word TF-IDF only (= B2)": dict(use_char=False, ctx_cols=None),
    "word + char TF-IDF": dict(ctx_cols=None),
    "+ sentiment/incongruity only": dict(ctx_cols=G["sentiment_incongruity"]),
    "+ pragmatic/stylistic only": dict(ctx_cols=G["pragmatic_stylistic"]),
    "+ structural only": dict(ctx_cols=G["structural"]),
    "FULL (all context features)": dict(ctx_cols=ALL_FEATURES),
    "FULL - sentiment/incongruity": dict(ctx_cols=G["pragmatic_stylistic"] + G["structural"]),
    "FULL - pragmatic/stylistic": dict(ctx_cols=G["sentiment_incongruity"] + G["structural"]),
    "FULL - structural": dict(ctx_cols=G["sentiment_incongruity"] + G["pragmatic_stylistic"]),
    "context features only (LR)": None,
}
abl = []
for k, cfg in abl_cfg.items():
    m = models.context_only_lr(C_) if cfg is None else models.improved_lr(C=C_, ctx_weight=W_, **cfg)
    cvf = cross_val_score(m, Xtr, ytr, cv=CV, scoring="f1", n_jobs=-1)
    m.fit(Xtr, ytr); r = metrics(yte, m.predict(Xte), scores(m, Xte))
    abl.append({"Configuration": k, "CV F1 mean": cvf.mean(), "CV F1 std": cvf.std(), "Test Acc": r["Accuracy"],
                "Test P": r["Precision"], "Test R": r["Recall"], "Test F1": r["F1"]})
# pre-processing ablation: stop-word removal on baseline
for nm, a_, b_ in [("B2 + stop-word removal", tr_sw, te_sw)]:
    m = models.baseline_lr(bp["clf__C"]); Xa, Xb = a_[["clean"]], b_[["clean"]]
    cvf = cross_val_score(m, Xa, ytr, cv=CV, scoring="f1", n_jobs=-1); m.fit(Xa, ytr); r = metrics(yte, m.predict(Xb))
    abl.append({"Configuration": nm, "CV F1 mean": cvf.mean(), "CV F1 std": cvf.std(), "Test Acc": r["Accuracy"],
                "Test P": r["Precision"], "Test R": r["Recall"], "Test F1": r["F1"]})
abl = pd.DataFrame(abl).set_index("Configuration"); abl.to_csv(f"{RES}/ablation.csv"); md(abl, f"{RES}/ablation.md"); print(abl.round(4).to_string())
fig, ax = plt.subplots(figsize=(8, 4.8)); a2 = abl.iloc[::-1]
ax.barh(a2.index, a2["Test F1"], color="#4C72B0", label="Test F1")
ax.barh(a2.index, a2["CV F1 mean"], color="#DD8452", height=.35, label="CV F1")
ax.set_xlim(0.4, 0.8); ax.set_title("Ablation: contribution of feature groups"); ax.legend(); plt.tight_layout()
plt.savefig(f"{FIG}/05_ablation.png", dpi=150); plt.close()

# =========================================================== 6. FEATURE INSPECTION
log("6. Most informative features")
lr = fitted["I2: TF-IDF(word+char)+Context + LogReg"]
names = np.array(lr.named_steps["feat"].get_feature_names_out()); coef = lr.named_steps["clf"].coef_[0]
fi = pd.DataFrame({"feature": names, "coef": coef})
ctx_fi = fi[fi.feature.str.startswith("ctx__")].assign(feature=lambda d: d.feature.str[5:]).sort_values("coef")
word_fi = fi[fi.feature.str.startswith("word__")].assign(feature=lambda d: d.feature.str[6:]).sort_values("coef")
ctx_fi.to_csv(f"{RES}/context_feature_coefficients.csv", index=False)
fig, ax = plt.subplots(1, 2, figsize=(13, 6))
top = pd.concat([ctx_fi.head(8), ctx_fi.tail(8)]); ax[0].barh(top.feature, top.coef, color=np.where(top.coef > 0, "#DD8452", "#4C72B0"))
ax[0].set_title("Context features: LR coefficients (scaled)")
top = pd.concat([word_fi.head(12), word_fi.tail(12)]); ax[1].barh(top.feature, top.coef, color=np.where(top.coef > 0, "#DD8452", "#4C72B0"))
ax[1].set_title("Word n-grams: LR coefficients (+ = ironic)"); plt.tight_layout(); plt.savefig(f"{FIG}/06_feature_importance.png", dpi=150); plt.close()
print("Top ironic words:", list(word_fi.tail(15).feature[::-1])); print("Top non-ironic words:", list(word_fi.head(15).feature))
print(ctx_fi.to_string())

# =========================================================== 7. ERROR ANALYSIS
log("7. Error analysis for:", best)
te_e = te.copy(); te_e["pred"] = preds[best]; te_e["score"] = sc[best]; te_e["correct"] = te_e.pred == te_e.label
te_e["error_type"] = np.select([(te_e.label == 1) & (te_e.pred == 0), (te_e.label == 0) & (te_e.pred == 1)], ["FN (missed irony)", "FP (false alarm)"], "correct")
te_e.loc[te_e.error_type != "correct"].sort_values("score").to_csv(f"{RES}/errors_test.csv", index=False)
buckets = {
    "short (<=8 tokens)": te_e.n_tokens <= 8, "long (>=20 tokens)": te_e.n_tokens >= 20,
    "has hashtag": te_e.n_hashtags > 0, "no hashtag": te_e.n_hashtags == 0,
    "has emoji/emoticon": te_e.n_emoji > 0, "has URL": te_e.has_url > 0, "has @mention": te_e.n_mentions > 0,
    "positive text sentiment": te_e.sent_text > 0.2, "negative text sentiment": te_e.sent_text < -0.2,
    "neutral text sentiment": te_e.sent_text.abs() <= 0.2,
    "sentiment incongruity present": te_e.seg_incongruity > 0, "text-vs-hashtag contrast": te_e.contrast_text_hashtag > 0,
    "has '!' ": te_e.n_exclaim > 0, "has interjection": te_e.n_interjections > 0,
}
eb = pd.DataFrame([{"Subset": k, "n": int(v.sum()), "% ironic (gold)": 100 * te_e[v].label.mean(),
                    "Error rate %": 100 * (~te_e[v].correct).mean(),
                    "FN rate % (of ironic)": 100 * ((te_e[v].error_type == "FN (missed irony)").sum() / max(te_e[v].label.sum(), 1)),
                    "FP rate % (of non-ironic)": 100 * ((te_e[v].error_type == "FP (false alarm)").sum() / max((te_e[v].label == 0).sum(), 1))}
                   for k, v in buckets.items()]).set_index("Subset")
eb.to_csv(f"{RES}/error_by_subset.csv"); md(eb, f"{RES}/error_by_subset.md", ".1f"); print(eb.round(1).to_string())
print("Overall error rate: %.1f%%   FN=%d  FP=%d" % (100 * (~te_e.correct).mean(), (te_e.error_type.str.startswith("FN")).sum(), (te_e.error_type.str.startswith("FP")).sum()))
fig, ax = plt.subplots(figsize=(8, 5)); eb["Error rate %"].sort_values().plot(kind="barh", ax=ax, color="#C44E52")
ax.axvline(100 * (~te_e.correct).mean(), color="k", ls="--", lw=1, label="overall"); ax.legend(); ax.set_title("Error rate by tweet subset (test)")
plt.tight_layout(); plt.savefig(f"{FIG}/07_error_by_subset.png", dpi=150); plt.close()
for et in ["FN (missed irony)", "FP (false alarm)"]:
    print(f"\n--- most confident {et} ---")
    sub = te_e[te_e.error_type == et].sort_values("score", ascending=(et.startswith("FN")))
    for _, r in sub.head(8).iterrows(): print(f"  p={r.score:.2f} | {r.text[:150]}")

# =========================================================== 8. SAVE
joblib.dump(fitted[best], f"{ROOT}/models/best_model.joblib")
json.dump({"best_model": best, "test_metrics": res.loc[best].drop("F1 95% CI").to_dict()}, open(f"{RES}/best_model.json", "w"), indent=2)
log("DONE. Best model saved to models/best_model.joblib")
