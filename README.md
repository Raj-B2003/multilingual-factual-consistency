# Multilingual Factual Consistency — ROME/MEMIT

Research prototype for factual model editing using ROME and MEMIT with
GPT-2 XL and CounterFact.

## Pipeline

CounterFact -> ROME / MEMIT -> efficacy -> specificity -> cross-lingual pilot
-> results -> Hugging Face -> Gradio Space

## Current prototype

- GPT-2 XL
- CounterFact
- ROME
- MEMIT
- Exact Match
- Levenshtein Ratio
- 0.80 success threshold
- Specificity pilot
- English -> Hindi exploratory transfer
- Hugging Face model artifact
- Gradio deployment

## Important scope

The current notebook is a free-Colab pilot. It does not claim completion
of the full 4-Romance + 3-Indic + ELFI + ELFO + three-checkpoint TIES study.

MEMIT uses a 10,000-sample pilot covariance configuration because the original
100,000-sample run exceeded the practical free-Colab execution window.

## Results

See:
- results/model_editing_results.csv
- results/model_editing_summary.csv

## Deployment

The deployment files are under deployment/.

## Large artifacts

Model weights and datasets are not stored in GitHub. They are kept in
Hugging Face / Google Drive.
