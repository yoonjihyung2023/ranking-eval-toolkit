A compact ranking evaluation toolkit for CTR / RecSys portfolio projects, covering AUC, LogLoss, Precision@K, Recall@K, and NDCG@K.

# ranking-eval-toolkit

Reusable ranking metrics toolkit for CTR/RecSys experiments  
Metrics: AUC, LogLoss, Precision@K, Recall@K, NDCG@K  
Purpose: reproducible offline evaluation for ranking models

## Why this repo matters
- Reusable evaluation utilities for ranking and recommendation experiments
- Covers both pointwise metrics and top-K ranking metrics
- Useful for CTR/RecSys offline validation and benchmarking
- Lightweight portfolio repo that complements model training/serving projects

## Metrics
- AUC
- LogLoss
- Precision@K
- Recall@K
- NDCG@K

## Quickstart
```bash
python example.py
pytest -q
Example use cases

Compare offline ranking quality across multiple models

Evaluate CTR/RecSys experiments with a reusable metric toolkit

Build cleaner experiment reports for portfolio or interview demos

## Portfolio Role

This repo demonstrates the **evaluation** part of my Ads/RecSys ML portfolio.

It complements:

- `ctr-seqrec-avazu` ??training / offline benchmark
- `ctr-api` ??FastAPI / Docker model serving
- `ranking-eval-toolkit` ??reusable ranking and CTR evaluation metrics

Together, the portfolio story is:

- train ??evaluate ??serve

## Metrics Covered

- AUC
- LogLoss
- Precision@K
- Recall@K
- NDCG@K


## Usage example

```python
from ranking_eval_toolkit import (
    auc_score,
    log_loss_score,
    precision_at_k,
    recall_at_k,
    ndcg_at_k,
)

y_true = [1, 0, 1, 0, 1]
y_score = [0.9, 0.2, 0.8, 0.1, 0.7]

print("AUC:", auc_score(y_true, y_score))
print("LogLoss:", log_loss_score(y_true, y_score))
print("Precision@3:", precision_at_k(y_true, y_score, k=3))
print("Recall@3:", recall_at_k(y_true, y_score, k=3))
print("NDCG@3:", ndcg_at_k(y_true, y_score, k=3))
```

Run the example:

```bash
python example.py
```

