# ranking-eval-toolkit

**Offline ranking/CTR evaluation utilities for trustworthy model benchmarking**  
**Focus:** AUC, LogLoss, Precision@K, Recall@K, NDCG@K, HitRate  
**Why this repo matters:** evaluation quality is as important as model quality

## One-line
A lightweight toolkit for evaluating ranking and CTR models with clear, reusable offline metrics.

## What this repo covers
- **CTR metrics**: AUC, LogLoss
- **Ranking metrics**: Precision@K, Recall@K, NDCG@K, HitRate
- **Evaluation utilities**: top-k helpers, batch scoring helpers, metric aggregation
- **Trustworthy benchmarking mindset**: metric definitions should be explicit, reproducible, and easy to inspect

## Why this repo matters
- Shows that I care not only about training models, but also about **how results are evaluated**
- Complements **`ctr-seqrec-avazu`** by separating **evaluation logic** from the benchmark project
- Makes common ranking metrics easier to reuse across recommendation and CTR experiments
- Provides interview-friendly code for discussing offline evaluation choices

## Example metrics
| Category | Metrics |
|---|---|
| CTR / classification | **AUC**, **LogLoss** |
| Ranking / recommendation | **Precision@K**, **Recall@K**, **NDCG@K**, **HitRate@K** |

## Example usage
`from ranking_eval_toolkit import auc_score, logloss_score, ndcg_at_k`

## Initial scope
1. `auc_score`
2. `logloss_score`
3. `precision_at_k`
4. `recall_at_k`
5. `ndcg_at_k`
6. `hitrate_at_k`

## Project goal
This repo is built to show practical understanding of **offline evaluation for ranking and recommendation systems**, with reusable code that is easy to read, test, and extend.
