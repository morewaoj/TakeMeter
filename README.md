# TakeMeter

TakeMeter is an AI201 Week 3 project for classifying NBA discourse quality. The goal is to distinguish useful basketball analysis from unsupported hot takes and low-signal reaction noise.

## Label Taxonomy

The classifier uses three labels:

| Label | Meaning | Typical Signals |
| --- | --- | --- |
| `analysis` | Reasoned NBA commentary that explains a claim with evidence, context, or basketball logic. | Mentions scheme, efficiency, matchup context, lineup data, shot profile, injuries, pace, or role. |
| `hot_take` | Strong or exaggerated NBA opinion with little support, often framed as certainty. | Absolute claims, overreaction to small samples, legacy debates, unsupported rankings, sweeping predictions. |
| `reaction_noise` | Low-information emotional reaction, meme-like response, or short post that does not make a meaningful basketball claim. | One-liners, all-caps reactions, jokes, vague complaints, emojis, empty hype. |

## Dataset

The frozen annotated dataset is in `data/takemeter_dataset.csv`.

Columns:

- `text`: NBA discourse sample to classify.
- `label`: One of `analysis`, `hot_take`, or `reaction_noise`.
- `notes`: Short reason for the label.

## Final Dataset Statistics

| Label | Count |
| --- | ---: |
| `analysis` | 124 |
| `hot_take` | 52 |
| `reaction_noise` | 64 |
| **Total** | **240** |

## Baseline Prompt

The baseline prompt is in `prompts/baseline_prompt.txt`. It will be used to compare a general prompt-only classifier against a later fine-tuned model.

## GroqCloud Baseline Setup

Before running the GroqCloud zero-shot baseline, create a local `.env` file from `.env.example` or export `GROQ_API_KEY` in your shell. Do not commit real API keys.

The baseline uses GroqCloud's OpenAI-compatible chat completions API with `llama-3.1-8b-instant`. It should classify only `data/test.csv` and save outputs to `results/baseline_predictions.csv` and `results/baseline_metrics.md`.

If GroqCloud returns a `403` or access-block error, do not keep retrying from the same environment. Run the baseline from an environment where GroqCloud API access is allowed, then save the same result files in `results/`.

## How to Run the Groq Baseline Locally

Install dependencies:

```bash
pip install groq pandas scikit-learn python-dotenv
```

Create a local `.env` file:

```env
GROQ_API_KEY=your_key_here
```

Run the baseline:

```bash
python scripts/run_groq_baseline.py
```

Expected output files:

- `results/baseline_predictions.csv`
- `results/baseline_metrics.md`

## Fine-Tuning Pipeline

Completed steps:

1. Created a frozen 240-example annotated dataset.
2. Split the dataset into stratified train, validation, and test sets.
3. Fine-tuned DistilBERT on the training set.
4. Used validation results during training.
5. Used the test set only for final evaluation.

## Model Information

| Item | Value |
| --- | --- |
| Model | `distilbert-base-uncased` |
| Task | 3-class text classification |
| Labels | `analysis`, `hot_take`, `reaction_noise` |
| Loss | Weighted CrossEntropyLoss |

Weighted CrossEntropyLoss was used to address class imbalance so the minority labels mattered more during training.

## Reproducing the Model

The trained model artifacts are not included in the repository because they exceed GitHub size limits.

To regenerate them:

1. Install requirements.
2. Run:

```bash
python scripts/train_distilbert.py
```

The model will be saved locally to `models/distilbert-takemeter` and used by `app/app.py`.

## Baseline Comparison

The GroqCloud zero-shot baseline is prepared but has not been run successfully in this environment because GroqCloud API access returned a 403/access-block response. No baseline results are reported here.

Planned comparison after the baseline is run from an allowed environment:

| System | Dataset Split | Metrics |
| --- | --- | --- |
| Baseline prompt classifier | Held-out test set | TBD |
| Fine-tuned DistilBERT classifier | Same held-out test set | Completed |

Do not invent baseline results. Baseline metrics should be added only after `results/baseline_metrics.md` exists.

## Final Evaluation Results

| Metric | Result |
| --- | ---: |
| Validation Accuracy | 0.7778 |
| Validation Macro F1 | 0.7127 |
| Test Accuracy | 0.8056 |
| Test Macro F1 | 0.7395 |

## Confusion Matrix

Rows are gold labels; columns are predicted labels.

| Gold \ Predicted | analysis | hot_take | reaction_noise |
| --- | ---: | ---: | ---: |
| `analysis` | 19 | 0 | 0 |
| `hot_take` | 2 | 5 | 1 |
| `reaction_noise` | 1 | 3 | 5 |

## Error Pattern Summary

The main remaining errors involve ambiguity between `hot_take` and `reaction_noise`, especially when a post mixes humor, sarcasm, and basketball judgment. The model also sometimes treats short basketball-adjacent posts as `analysis` when they mention basketball concepts but do not provide enough reasoning.

### Specific Misclassified Examples

| Post | Gold Label | Predicted Label | Explanation |
| --- | --- | --- | --- |
| "That defensive possession after the missed free throw was worse than the miss itself." | `hot_take` | `analysis` | The model likely focused on the phrase "defensive possession" and treated it as analysis, but the post does not explain the defensive breakdown. |
| "The stat nerds are going to hate this, but he just does not pass the eye test." | `hot_take` | `analysis` | The model likely treated the stats-versus-eye-test framing as basketball reasoning, but "eye test" is vague and unsupported. |
| "The hot mic picked the perfect time to expose that huddle." | `reaction_noise` | `analysis` | The model likely focused on "huddle" as a basketball/coaching cue, but the post is mainly a broadcast or game-thread reaction. |

The recurring pattern is that the model sometimes upgrades basketball-adjacent words into `analysis` even when the post lacks real reasoning.

## Demo

TakeMeter includes a working interface for classifying NBA posts. The demo classifies posts into `analysis`, `hot_take`, and `reaction_noise`, and discusses both successful predictions and model errors.

## AI Usage Reflection

I used AI assistance to scaffold the project structure, draft the initial planning document, define the three-label taxonomy, and create the first baseline prompt. I also used AI to generate candidate NBA discussion posts for dataset collection, but I did not treat those labels as final automatically.

AI was especially useful during label stress-testing and QA review. It helped identify ambiguous examples, near-duplicates, and possible label inconsistencies before the final dataset was frozen. I reviewed those suggestions and made human decisions where needed, including overriding or refining labels for borderline examples.

The biggest annotation issue we discovered was the overlap between `hot_take` and `reaction_noise`. Some posts were both funny and judgmental, so I added a decision rule: if the main purpose is evaluating a player, coach, team, strategy, or outcome, it should be `hot_take`; if the main purpose is humor, venting, celebration, frustration, sarcasm, or emotional reaction, it should be `reaction_noise`.

AI also helped write and revise the training and evaluation scripts, but the reported metrics come only from actual saved outputs. I did not invent baseline results; the GroqCloud baseline was prepared, but API access was blocked in this environment.

## Spec Reflection

My original plan was to build the project step by step: define the taxonomy, create a dataset, run a zero-shot baseline, then fine-tune a model. The final implementation followed most of that structure, but the order changed because the GroqCloud baseline could not be run from this environment due to an access-block error. Instead of inventing baseline numbers, I kept the baseline workflow documented and moved forward with the fine-tuned DistilBERT model.

The first DistilBERT run exposed an important modeling problem: the model predicted `analysis` for every test example. That showed the class imbalance was affecting learning, because `analysis` was the largest class. To address that, I updated the training script to compute class weights from `data/train.csv` only and use weighted CrossEntropyLoss. The weighted model produced better validation and test macro F1, and it predicted all three classes instead of collapsing to one label.

The project still has limitations. The dataset is small, and several errors are still caused by the same ambiguity we found during annotation: short posts can sit between `hot_take` and `reaction_noise`, and basketball-adjacent wording can make a weak post look like `analysis`. Even with those limits, the final project satisfies the core spec: it includes a label taxonomy, annotated dataset, documented fine-tuning pipeline, prepared baseline workflow, evaluation results, planning document, demo notes, AI usage reflection, and this spec reflection.
