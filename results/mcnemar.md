| Model vs B2                               |   B2 right/model wrong |   B2 wrong/model right |   p-value |
|:------------------------------------------|-----------------------:|-----------------------:|----------:|
| B1: TF-IDF(word) + Naive Bayes            |                42.0000 |                49.0000 |    0.5296 |
| I1: Context-only + GradBoost              |               109.0000 |               128.0000 |    0.2423 |
| I2: TF-IDF(word+char)+Context + LogReg    |                55.0000 |                75.0000 |    0.0953 |
| I3: TF-IDF(word+char)+Context + LinearSVM |                74.0000 |                89.0000 |    0.2728 |
| I4: Late fusion (Text-LR + Context-GB)    |                46.0000 |                76.0000 |    0.0084 |