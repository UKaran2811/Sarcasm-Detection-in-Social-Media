"""NLP preprocessing for noisy social-media text.

Steps (in order):
 1. HTML unescape, '||' (tweet line-break marker in this corpus) -> sentence break
 2. Mask label-leaking hashtags (#sarcasm, #irony, #not ...) - avoids shortcut learning
 3. URL -> 'url', @mention -> '@user'
 4. Hashtag segmentation (#NoReligion -> 'no religion'); hashtags are kept as words
    because they often carry the *situational context* of a sarcastic tweet
 5. Emoji -> textual tokens (emoji_face_with_tears_of_joy)
 6. Lower-casing, elongation normalisation (soooo -> soo)
 7. Regex tokenisation that keeps emoticons and punctuation runs ('!!!', '...', '?!')
Stop-words are KEPT by default: words like 'so', 'just', 'oh', 'really' are strong
sarcasm markers (an ablation in run_all.py shows the effect of removing them).
"""
import html
import re

import emoji
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

LEAK_TAGS = {"sarcasm", "sarcastic", "irony", "ironic", "not", "sarcasmalert", "satire"}

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
ELONG_RE = re.compile(r"(\w)\1{2,}")
TOKEN_RE = re.compile(
    r"emoji_\w+"                                 # converted emoji
    r"|[:;=8][-o*']?[)(\]\[dDpP/\\|]"             # emoticons :) ;-) :P
    r"|[a-z]+(?:'[a-z]+)?"                        # words / contractions
    r"|\d+"                                       # numbers
    r"|[!?]{2,}|\.{2,}|[!?.,\"]"                  # punctuation runs / marks
)
CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def mask_leak_tags(text):
    return HASHTAG_RE.sub(lambda m: "" if m.group(1).lower() in LEAK_TAGS else m.group(0), text)


def normalise_raw(text):
    """Light normalisation that preserves case/punctuation (used by feature extractor)."""
    t = html.unescape(text)
    t = t.replace("||", " . ")
    t = mask_leak_tags(t)
    t = URL_RE.sub(" URL ", t)
    t = MENTION_RE.sub(" @user ", t)
    return re.sub(r"\s+", " ", t).strip()


def split_hashtag(tag):
    return CAMEL_RE.sub(" ", tag).replace("_", " ")


def clean_text(raw_norm, remove_stopwords=False):
    t = HASHTAG_RE.sub(lambda m: " " + split_hashtag(m.group(1)) + " ", raw_norm)
    t = emoji.demojize(t, delimiters=(" emoji_", " "))
    t = t.replace("URL", " url ").replace("@user", " user ")
    t = ELONG_RE.sub(r"\1\1", t.lower())
    tokens = TOKEN_RE.findall(t)
    if remove_stopwords:
        tokens = [w for w in tokens if w not in ENGLISH_STOP_WORDS]
    return " ".join(tokens)


def preprocess_df(df, remove_stopwords=False):
    out = df.copy()
    out["raw_norm"] = out["text"].apply(normalise_raw)
    out["clean"] = out["raw_norm"].apply(lambda s: clean_text(s, remove_stopwords))
    return out
