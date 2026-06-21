# TakeMeter Planning

## Project Goal

Build a small NBA discourse quality classifier that labels short basketball commentary as:

- `analysis`
- `hot_take`
- `reaction_noise`

The project should show the full machine learning workflow: taxonomy design, annotation, baseline prompting, fine-tuning plan, evaluation plan, and reflection.

## Community Choice

NBA online discourse is a strong fit for this classification task because fans constantly post short reactions, detailed analysis, and exaggerated opinions in the same spaces. Game threads, recap comments, social posts, and podcast discussions often mix useful basketball reasoning with emotional reactions, which makes the community realistic for a discourse quality classifier.

## Scope Guardrails

- Do not claim a model has been trained until a real fine-tuning job is completed.
- Do not invent evaluation metrics.
- Keep the first version small and understandable.
- Prefer clear labels and clean examples over a large noisy dataset.

## Label Taxonomy

### `analysis`

Use this label when the post makes a basketball claim and gives reasoning, evidence, or context.

Examples of evidence:

- Matchup details.
- Shot quality.
- Defensive coverage.
- Lineup fit.
- Efficiency stats.
- Injury or schedule context.

Example posts:

- "The Wolves slowed the game down by keeping two bigs near the rim, which made Dallas rely more on contested pull-up threes."
- "Brunson's scoring mattered, but the bigger issue was New York forcing switches until the help defense opened corner looks."

### `hot_take`

Use this label when the post makes a strong NBA claim without enough support.

Common patterns:

- Extreme ranking claims.
- Legacy arguments with no evidence.
- Certainty after one game.
- Dismissing a player, coach, or team with no context.

Example posts:

- "One playoff loss proves this team was fake all season."
- "He is already the best point guard ever and nobody else is close."

### `reaction_noise`

Use this label when the post is mostly emotional reaction, joke, hype, complaint, or filler.

Common patterns:

- Very short reaction posts.
- Meme-like comments.
- All-caps responses.
- No clear basketball claim.

Example posts:

- "This game is unserious."
- "NO WAY THAT JUST HAPPENED."

## Hard Edge Case

Edge case: "This coach is awful; those late-game lineups made no sense."

This could be `hot_take` because it makes a strong unsupported claim, or `analysis` because it gestures toward a tactical issue. Decision rule: label it `hot_take` unless the post explains which lineup choice was flawed and why it affected the game. A basketball-related complaint is not enough for `analysis` without specific reasoning.

If a post contains both judgment and humor or emotion, label it `hot_take` when the primary purpose is evaluating a player, coach, team, strategy, or basketball outcome. Label it `reaction_noise` when the primary purpose is humor, venting, celebration, frustration, sarcasm, or emotional reaction.

## Dataset Plan

Current status: small starter annotated CSV exists.

Next dataset steps:

1. Add more examples from NBA game threads, recap comments, podcast-style snippets, and social posts.
2. Keep labels balanced across the three classes.
3. Add difficult borderline cases.
4. Review annotations for consistency.
5. Create train, validation, and test splits.

Target future dataset size:

- Starter milestone: 30 to 60 labeled examples.
- Fine-tuning milestone: 150 or more labeled examples, if time allows.

## Baseline Plan

Use `prompts/baseline_prompt.txt` to classify test examples with a prompt-only model.

Record:

- Predicted label.
- Gold label.
- Whether the prediction was correct.
- Short error note for incorrect predictions.

No baseline results have been generated yet.

## Fine-Tuning Pipeline Documentation

Planned pipeline:

1. Validate CSV labels against the allowed taxonomy.
2. Split rows into train, validation, and test sets.
3. Convert train and validation rows into model fine-tuning examples.
4. Start a fine-tuning job.
5. Save the fine-tuned model identifier.
6. Run the fine-tuned model on the held-out test set.
7. Compare fine-tuned results against the baseline prompt.

Future implementation files may include:

- `scripts/validate_dataset.py`
- `scripts/prepare_finetune_data.py`
- `scripts/run_baseline.py`
- `scripts/evaluate.py`

These scripts have not been created yet.

## Evaluation Report Plan

Future evaluation should include:

- Accuracy.
- Per-label precision.
- Per-label recall.
- Per-label F1.
- Macro F1.
- Confusion matrix.
- Error analysis with specific misclassified examples.

These metrics fit the project because accuracy gives a simple overall score, F1 shows whether the classifier performs well across all labels, and the confusion matrix reveals which labels are being mixed up. This is especially important for borderline cases like `hot_take` versus `reaction_noise`.

Metrics placeholder:

| Metric | Value |
| --- | --- |
| Accuracy | TBD |
| Macro F1 | TBD |
| `analysis` precision | TBD |
| `analysis` recall | TBD |
| `hot_take` precision | TBD |
| `hot_take` recall | TBD |
| `reaction_noise` precision | TBD |
| `reaction_noise` recall | TBD |

## Definition of Success

The project will be considered successful if the final evaluated classifier:

- Reaches at least 75% accuracy on the held-out test set.
- Reaches at least 0.70 F1 for each individual label.
- Shows no severe collapse into a single majority label.
- Improves over the baseline prompt on macro F1 or provides a clear error analysis explaining why it does not.

## Demo Video Notes

Suggested demo flow:

1. Show the problem and why NBA discourse quality is interesting.
2. Explain the three labels.
3. Open the dataset and show annotation notes.
4. Show the baseline prompt.
5. Later, show model comparison results after real evaluation exists.
6. End with limitations and next steps.

## AI Usage Log

Initial AI assistance:

- Created project scaffold.
- Drafted label taxonomy.
- Created starter annotated dataset.
- Wrote baseline prompt.
- Added placeholders for fine-tuning, evaluation, demo, and reflection sections.

Future updates should log any AI help with code, prompts, data generation, evaluation, or writing.

## Spec Reflection

The current scaffold satisfies the assignment structure without overclaiming progress. The taxonomy is explicit, the dataset is annotated, the baseline and fine-tuning plans are documented, and evaluation sections are ready for real results later.

Main risks to address next:

- The starter dataset is too small for reliable model evaluation.
- Some NBA discourse can be ambiguous between `hot_take` and `reaction_noise`.
- The project needs real baseline predictions before comparing against fine-tuning.
