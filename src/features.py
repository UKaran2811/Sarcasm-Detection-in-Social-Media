"""Contextual feature extraction.

'Context' in a single tweet (no conversation thread is available in this dataset) is
modelled at three levels:
  (1) SENTIMENT CONTEXT / INCONGRUITY - sarcasm is often a positive sentiment placed in a
      negative situation. We compare the polarity of: clauses within the tweet, the main
      text vs. its hashtags, and the main text vs. its emoji.
  (2) PRAGMATIC / STYLISTIC CUES      - interjections, intensifiers, laughter, elongation,
      '!', '?!', '...', quotation marks, ALL-CAPS words (prosody substitutes in text).
  (3) STRUCTURAL CONTEXT              - hashtags, mentions, URLs, emoji counts/positions.
"""
import math
import re

import emoji
import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from .preprocess import HASHTAG_RE, split_hashtag

_vader = SentimentIntensityAnalyzer()
LEX = _vader.lexicon
# words usable for scanning un-segmented hashtags (#worstmorning)
LEX_LONG = {w: v for w, v in LEX.items() if len(w) >= 4 and w.isalpha()}

INTERJ = {"oh", "wow", "yay", "ugh", "ah", "aw", "aww", "oops", "yeah", "yep", "well", "gee",
          "hmm", "meh", "omg", "ha", "haha", "hahaha", "lol", "lmao", "duh", "yippee", "whoa", "ooh", "ohh"}
INTENS = {"so", "totally", "absolutely", "really", "very", "completely", "just", "such", "truly",
          "definitely", "obviously", "literally", "perfectly", "clearly", "utterly", "extremely", "super"}
LAUGH_RE = re.compile(r"\b(?:ha){2,}h?\b|\blol+\b|\blmao+\b|\brofl\b|\bhehe+\b", re.I)
EMOJI_RE = re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]")
EMOTICON_RE = re.compile(r"[:;=8][-o*']?[)(\]\[dDpP/\\|]")
CLAUSE_SPLIT = re.compile(r"[.!?;,\u2026]+|\s+(?:but|however|though|yet|although|except)\s+|\s-\s", re.I)


def _norm(score):
    return score / math.sqrt(score * score + 15)


def vader_compound(text):
    return _vader.polarity_scores(text)["compound"] if text.strip() else 0.0


def hashtag_sentiment(tags):
    """Sentiment of hashtags; un-segmented tags are scanned for lexicon words."""
    total = 0.0
    for tag in tags:
        parts = split_hashtag(tag).lower().split()
        for p in parts:
            if p in LEX:
                total += LEX[p]
            else:                                    # substring scan, e.g. 'worstmorning'
                hits = [v for w, v in LEX_LONG.items() if w in p]
                if hits:
                    total += max(hits, key=abs)
    return _norm(total) if total else 0.0


def _opposite(a, b, thr=0.2):
    return float(abs(a) >= thr and abs(b) >= thr and a * b < 0)


def extract_one(raw_norm):
    tags = HASHTAG_RE.findall(raw_norm)
    no_tags = HASHTAG_RE.sub(" ", raw_norm)
    emojis = EMOJI_RE.findall(raw_norm) + EMOTICON_RE.findall(raw_norm)
    text_only = EMOJI_RE.sub(" ", no_tags).replace("URL", " ").replace("@user", " ")
    words = re.findall(r"[A-Za-z']+", raw_norm)
    lw = [w.lower() for w in words]
    n_words = max(len(words), 1)

    # ---------- (1) sentiment context / incongruity ----------
    s_text = vader_compound(text_only)
    s_hash = hashtag_sentiment(tags)
    s_emoji = vader_compound(" ".join(emojis)) if emojis else 0.0
    s_all = vader_compound(raw_norm)
    clauses = [c for c in CLAUSE_SPLIT.split(text_only) if c and c.strip()]
    seg = [vader_compound(c) for c in clauses] or [0.0]
    seg_max, seg_min = max(seg), min(seg)
    signs = [np.sign(s) for s in seg if abs(s) >= 0.2]
    flips = sum(1 for a, b in zip(signs, signs[1:]) if a != b)
    pos_w = sum(1 for w in lw if LEX.get(w, 0) > 0)
    neg_w = sum(1 for w in lw if LEX.get(w, 0) < 0)

    f = {
        "sent_text": s_text, "sent_hashtag": s_hash, "sent_emoji": s_emoji, "sent_all": s_all,
        "seg_max": seg_max, "seg_min": seg_min,
        "seg_incongruity": (seg_max - seg_min) if (seg_max > 0.2 and seg_min < -0.2) else 0.0,
        "seg_flips": flips,
        "n_pos_words": pos_w, "n_neg_words": neg_w,
        "pos_and_neg_words": float(pos_w > 0 and neg_w > 0),
        "contrast_text_hashtag": _opposite(s_text, s_hash),
        "gap_text_hashtag": abs(s_text - s_hash) if (s_hash != 0 and tags) else 0.0,
        "contrast_text_emoji": _opposite(s_text, s_emoji),
        "pos_text_neg_context": float(s_text > 0.2 and (s_hash < -0.2 or s_emoji < -0.2)),
    }

    # ---------- (2) pragmatic / stylistic cues ----------
    f.update({
        "n_exclaim": raw_norm.count("!"),
        "n_question": raw_norm.count("?"),
        "has_mixed_punct": float(bool(re.search(r"[!?]{2,}", raw_norm)) and "?" in raw_norm and "!" in raw_norm),
        "n_ellipsis": len(re.findall(r"\.{2,}|\u2026", raw_norm)),
        "n_quotes": raw_norm.count('"') // 2 + raw_norm.count("\u201c"),
        "caps_ratio": sum(1 for c in raw_norm if c.isupper()) / max(len(raw_norm), 1),
        "n_allcaps_words": sum(1 for w in words if len(w) > 1 and w.isupper() and w not in ("URL",)),
        "n_elongated": len(re.findall(r"(\w)\1{2,}", raw_norm)),
        "n_interjections": sum(1 for w in lw if w in INTERJ),
        "n_intensifiers": sum(1 for w in lw if w in INTENS),
        "n_laughter": len(LAUGH_RE.findall(raw_norm)),
        "starts_interjection": float(bool(lw) and lw[0] in INTERJ),
    })

    # ---------- (3) structural context ----------
    f.update({
        "n_tokens": len(words),
        "log_chars": math.log1p(len(raw_norm)),
        "n_hashtags": len(tags),
        "ends_with_hashtag": float(bool(re.search(r"#\w+\s*$", raw_norm))),
        "n_mentions": raw_norm.count("@user"),
        "has_url": float("URL" in raw_norm),
        "n_emoji": len(emojis),
        "emoji_at_end": float(bool(re.search(r"([\U0001F300-\U0001FAFF\u2600-\u27BF]|[:;=8][-o*']?[)(\]\[dDpP/\\|])\s*$", raw_norm))),
    })
    return f


FEATURE_GROUPS = {
    "sentiment_incongruity": ["sent_text", "sent_hashtag", "sent_emoji", "sent_all", "seg_max", "seg_min",
                              "seg_incongruity", "seg_flips", "n_pos_words", "n_neg_words", "pos_and_neg_words",
                              "contrast_text_hashtag", "gap_text_hashtag", "contrast_text_emoji",
                              "pos_text_neg_context"],
    "pragmatic_stylistic": ["n_exclaim", "n_question", "has_mixed_punct", "n_ellipsis", "n_quotes", "caps_ratio",
                            "n_allcaps_words", "n_elongated", "n_interjections", "n_intensifiers", "n_laughter",
                            "starts_interjection"],
    "structural": ["n_tokens", "log_chars", "n_hashtags", "ends_with_hashtag", "n_mentions", "has_url",
                   "n_emoji", "emoji_at_end"],
}
ALL_FEATURES = [c for g in FEATURE_GROUPS.values() for c in g]


def add_features(df):
    feats = pd.DataFrame([extract_one(t) for t in df["raw_norm"]], index=df.index)
    return pd.concat([df, feats[ALL_FEATURES]], axis=1)
