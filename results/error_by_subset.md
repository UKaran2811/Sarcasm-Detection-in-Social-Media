| Subset                        |     n |   % ironic (gold) |   Error rate % |   FN rate % (of ironic) |   FP rate % (of non-ironic) |
|:------------------------------|------:|------------------:|---------------:|------------------------:|----------------------------:|
| short (<=8 tokens)            | 144.0 |              39.6 |           31.2 |                    15.8 |                        41.4 |
| long (>=20 tokens)            | 155.0 |              45.2 |           29.0 |                    25.7 |                        31.8 |
| has hashtag                   | 362.0 |              38.4 |           26.2 |                    24.5 |                        27.4 |
| no hashtag                    | 422.0 |              40.8 |           32.7 |                    34.3 |                        31.6 |
| has emoji/emoticon            | 105.0 |              32.4 |           21.0 |                    14.7 |                        23.9 |
| has URL                       | 197.0 |              19.8 |           18.8 |                    79.5 |                         3.8 |
| has @mention                  | 295.0 |              28.5 |           28.8 |                    56.0 |                        18.0 |
| positive text sentiment       | 326.0 |              43.3 |           27.6 |                    14.9 |                        37.3 |
| negative text sentiment       | 161.0 |              34.2 |           36.0 |                    47.3 |                        30.2 |
| neutral text sentiment        | 297.0 |              38.7 |           28.6 |                    40.0 |                        21.4 |
| sentiment incongruity present |  38.0 |              47.4 |           28.9 |                    22.2 |                        35.0 |
| text-vs-hashtag contrast      |  45.0 |              55.6 |           37.8 |                    16.0 |                        65.0 |
| has '!'                       | 119.0 |              39.5 |           31.9 |                    21.3 |                        38.9 |
| has interjection              |  80.0 |              48.8 |           27.5 |                    17.9 |                        36.6 |