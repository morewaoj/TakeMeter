# Fine-Tuned DistilBERT Metrics

Base model: `distilbert-base-uncased`

Loss: class-weighted cross entropy using weights computed from `data/train.csv` only.

Dataset splits: `data/train.csv`, `data/validation.csv`, `data/test.csv`

## Class Weights

| Label | Weight |
| --- | ---: |
| `analysis` | 0.6437 |
| `hot_take` | 1.5556 |
| `reaction_noise` | 1.2444 |

## Validation Metrics

Examples: 36

Accuracy: 0.7778

Macro F1: 0.7127

### Per-Class Metrics

| Label | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| `analysis` | 0.8182 | 1.0000 | 0.9000 |
| `hot_take` | 0.6667 | 0.5000 | 0.5714 |
| `reaction_noise` | 0.7500 | 0.6000 | 0.6667 |

### Macro Averages

| Precision | Recall | F1 |
| ---: | ---: | ---: |
| 0.7449 | 0.7000 | 0.7127 |

### Confusion Matrix

Rows are gold labels; columns are predicted labels.

| Gold \ Predicted | analysis | hot_take | reaction_noise |
| --- | ---: | ---: | ---: |
| `analysis` | 18 | 0 | 0 |
| `hot_take` | 2 | 4 | 2 |
| `reaction_noise` | 2 | 2 | 6 |

## Final Test Metrics

Examples: 36

Accuracy: 0.8056

Macro F1: 0.7395

### Per-Class Metrics

| Label | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| `analysis` | 0.8636 | 1.0000 | 0.9268 |
| `hot_take` | 0.6250 | 0.6250 | 0.6250 |
| `reaction_noise` | 0.8333 | 0.5556 | 0.6667 |

### Macro Averages

| Precision | Recall | F1 |
| ---: | ---: | ---: |
| 0.7740 | 0.7269 | 0.7395 |

### Confusion Matrix

Rows are gold labels; columns are predicted labels.

| Gold \ Predicted | analysis | hot_take | reaction_noise |
| --- | ---: | ---: | ---: |
| `analysis` | 19 | 0 | 0 |
| `hot_take` | 2 | 5 | 1 |
| `reaction_noise` | 1 | 3 | 5 |
