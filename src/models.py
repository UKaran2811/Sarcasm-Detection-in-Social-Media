"""Model definitions: baselines and the improved context-aware classifiers."""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import LinearSVC

from .features import ALL_FEATURES

SEED = 42


def _word_tfidf():
    return TfidfVectorizer(token_pattern=r"\S+", ngram_range=(1, 2), min_df=2, sublinear_tf=True)


def _char_tfidf():
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True)


def _ctx_block():
    return Pipeline([("scale", StandardScaler()),
                     ("clip", FunctionTransformer(lambda x: np.clip(x, -5, 5), feature_names_out="one-to-one"))])


def build_features(use_word=True, use_char=True, ctx_cols=None, ctx_weight=1.0):
    parts, weights = [], {}
    if use_word:
        parts.append(("word", _word_tfidf(), "clean"))
    if use_char:
        parts.append(("char", _char_tfidf(), "clean"))
    if ctx_cols:
        parts.append(("ctx", _ctx_block(), list(ctx_cols)))
        weights["ctx"] = ctx_weight
    return ColumnTransformer(parts, transformer_weights=weights or None)


# ---------------- Baselines ----------------
def baseline_nb(alpha=0.5):
    return Pipeline([("feat", build_features(True, False)), ("clf", MultinomialNB(alpha=alpha))])


def baseline_lr(C=1.0):
    return Pipeline([("feat", build_features(True, False)),
                     ("clf", LogisticRegression(C=C, max_iter=2000, random_state=SEED))])


# ---------------- Improved (contextual) ----------------
def improved_lr(C=1.0, ctx_weight=0.3, ctx_cols=ALL_FEATURES, use_word=True, use_char=True):
    return Pipeline([("feat", build_features(use_word, use_char, ctx_cols, ctx_weight)),
                     ("clf", LogisticRegression(C=C, max_iter=3000, random_state=SEED))])


def improved_svm(C=0.1, ctx_weight=0.3, ctx_cols=ALL_FEATURES):
    return Pipeline([("feat", build_features(True, True, ctx_cols, ctx_weight)),
                     ("clf", LinearSVC(C=C, random_state=SEED))])


def context_only_hgb():
    return Pipeline([("feat", ColumnTransformer([("ctx", "passthrough", list(ALL_FEATURES))])),
                     ("clf", HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200,
                                                            l2_regularization=1.0, random_state=SEED))])


def context_only_lr(C=1.0):
    return Pipeline([("feat", build_features(False, False, ALL_FEATURES, 1.0)),
                     ("clf", LogisticRegression(C=C, max_iter=2000, random_state=SEED))])


class LateFusion(BaseEstimator, ClassifierMixin):
    """Average probabilities of a text model (LR on TF-IDF) and a context model (gradient boosting)."""

    def __init__(self, text_model=None, ctx_model=None, w=0.5):
        self.text_model, self.ctx_model, self.w = text_model, ctx_model, w

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.t_ = clone(self.text_model).fit(X, y)
        self.c_ = clone(self.ctx_model).fit(X, y)
        return self

    def predict_proba(self, X):
        return self.w * self.t_.predict_proba(X) + (1 - self.w) * self.c_.predict_proba(X)

    def decision_function(self, X):
        return self.predict_proba(X)[:, 1] - 0.5

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
