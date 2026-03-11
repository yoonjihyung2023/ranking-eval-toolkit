# ranking-eval-toolkit

Lightweight ranking evaluation toolkit for CTR/RecSys experiments.  
Focus: reproducible offline metrics for ranking models.  
Includes practical evaluation utilities for ML portfolio projects.

## What this repo does
- Computes common ranking metrics
- Supports reproducible offline evaluation
- Helps compare model outputs consistently
- Designed as a small portfolio-friendly utility repo

## Currently included
- AUC
- LogLoss
- Precision@K
- Recall@K
- NDCG@K

## Quickstart
```bash
python example.py
Run tests
pytest -q
Example output
AUC: 1.0
LogLoss: 0.202737
Precision@3: 1.0
Recall@3: 1.0
NDCG@3: 1.0
Project structure
ranking_eval/
  __init__.py
  metrics.py
tests/
  test_metrics.py
example.py
Why this repo matters

Model training is not enough.
Reliable ML work also needs clear, repeatable evaluation.

This repo is meant to show:

evaluation mindset

ranking metric understanding

reusable ML tooling habit

Roadmap

 Add MAP

 Add pytest setup

 Add usage examples
