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
