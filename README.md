# Sarcasm Detection in Social Media Text Using Contextual Features

NLP mini-project. Binary classification of English tweets as **ironic/sarcastic vs. not**, comparing
TF-IDF baselines with an improved approach that adds *contextual features* (sentiment incongruity,
hashtag/emoji context, pragmatic cues) and a late-fusion model.

## Quick start
```bash
pip install -r requirements.txt
python run_all.py            # downloads data, trains, evaluates, writes results/ and figures (≈5 min, 1 CPU)
python predict.py "Oh great, another Monday morning meeting!"   # try the saved model
```
Everything is deterministic (seed 42). The dataset is downloaded automatically from GitHub on first run.

## Project layout
```
run_all.py          full pipeline (data -> preprocessing -> features -> models -> evaluation -> ablation -> error analysis)
predict.py          inference with the saved best model
src/data.py         download / load / dataset statistics
src/preprocess.py   tweet normalisation + tokenisation
src/features.py     contextual feature extraction (34 features in 3 groups)
src/models.py       baselines, improved models, late-fusion classifier
src/evaluate.py     metrics, bootstrap CI, McNemar test
data/               train.txt, test.txt (downloaded)
results/            all tables (.csv/.md), run_log.txt, errors_test.csv, figures/
models/             best_model.joblib
REPORT.md / .docx   full written report (methodology, results, analysis, references)
```

## Dataset
SemEval-2018 Task 3 (Subtask A), *Irony detection in English tweets* - Van Hee, Lefever & Hoste (2018).
Source: https://github.com/Cyvhee/SemEval2018-Task3 . 3,834 training tweets (1,911 ironic / 1,923 not) and
784 gold test tweets (311 / 473). Details and caveats are in `REPORT.md` (section 3).

## Notes / honesty box
* The corpus is annotated for **irony** (sarcasm is its most common sub-type, collected through
  #sarcasm/#irony/#not hashtags). The report discusses this label-scope caveat.
* Only single-tweet context is available (no reply threads). "Contextual" therefore means intra-tweet context:
  clause-level sentiment, text-vs-hashtag and text-vs-emoji contrast, pragmatic cues.
* The saved demo model is trained on tweets; short texts with no cues can be misclassified (see limitations).
