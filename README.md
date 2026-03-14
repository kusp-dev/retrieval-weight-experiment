# Gradient-Free Retrieval Weight Learning via Thompson Sampling

![Tests](https://img.shields.io/badge/tests-238_passed-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-80%25-green)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-Apache_2.0-blue)

Online optimization of retrieval weights (recency, importance, relevance) in LLM-based agent memory systems using dimension-specific self-assessment feedback.

> **Paper:** [Gradient-Free Retrieval Weight Learning via Thompson Sampling with LLM Self-Assessment](paper/) -- arXiv preprint.

## Key Results

Results from the v3 experiment (1,200 episodes: 300 tasks x 4 conditions, 205 domain-specific skills, 12 weight presets).

| Condition | Mean Tokens | Token Reduction | NDCG@5 | MRR | Best Arm | Posterior |
|-----------|-------------|-----------------|--------|-----|----------|-----------|
| C1 Control | 5,517 | -- | 0.223 | 0.302 | -- | -- |
| C2 Dim Feedback | 4,824 | -12.6% | 0.251 | 0.331 | pure\_relevance | 0.709 |
| C3 Full System | 4,830 | -12.5% | 0.314 | 0.411 | pure\_relevance | 0.749 |
| C4 Qualitative | 4,429 | -19.7% | 0.245 | 0.313 | pure\_importance | 0.472 |

Feedback-guided retrieval reduces token consumption by 12--20% vs. control (Kruskal--Wallis H=33.87, p<0.001; all control-vs-treatment comparisons significant after Bonferroni correction). The full system (C3) achieves the strongest retrieval quality (+40.8% NDCG@5 vs. control, +35.8% MRR), while the qualitative condition (C4) yields the largest token savings.

### What Drives the Token Reduction?

Better-weighted retrieval produces more focused prompts, yielding shorter LLM responses. The control condition averages 3,813 output tokens per episode vs. 2,537 for the best treatment -- a 33% reduction in output verbosity. Input tokens actually *increase* slightly in treatment conditions (more relevant context retrieved), but the output savings dominate.

## How It Works

The system learns optimal retrieval weights online using Thompson Sampling over 12 discrete weight presets. After each task, the LLM rates how useful each retrieval dimension (recency, importance, relevance) was -- this "input assessment" provides the reward signal. Unlike output-level self-assessment, which leads to reward hacking (Pan et al., 2024), assessing input quality produces stable, informative feedback.

### Experimental Conditions

| # | Condition | Bandit | Feedback | Explanation Embedding |
|---|-----------|--------|----------|-----------------------|
| 1 | Control | No (fixed equal weights) | None | No |
| 2 | Dimension Feedback | Yes | Likert 1--5 | No |
| 3 | Full System | Yes | Likert 1--5 | Yes |
| 4 | Qualitative | Yes | Free-text -> anchor embedding | No |

### Architecture

```
Task Corpus                    Skill Library
(300 tasks, 5 themes,          (205 skills, 10 domains)
 3 difficulty levels)                |
        |                            v
        +---->  Hybrid Search (BM25 via FTS5 + Dense via Qwen3 + RRF Fusion)
                        |
                        v
                Weight-Scored Ranking (TS-selected preset applied
                to recency, importance, relevance dimensions)
                        |
                        v
                LLM Execution (MiniMax M2.5)
                        |
                        v
                Feedback Parsing (Likert regex / anchor embedding)
                        |
                        v
                Bandit Posterior Update (Beta(alpha, beta) per arm)
                        |
                        +---> loop
```

### Weight Presets (12 Arms)

The bandit selects from 12 presets spanning the weight simplex `[recency, importance, relevance]`:

| Preset | Weights | Role |
|--------|---------|------|
| pure\_relevance | [0.05, 0.05, 0.90] | Semantic match |
| pure\_recency | [0.85, 0.05, 0.10] | Freshness |
| pure\_importance | [0.05, 0.85, 0.10] | Track record |
| relevance\_heavy | [0.15, 0.15, 0.70] | Topic-driven |
| recency\_heavy | [0.60, 0.15, 0.25] | Fast-changing domains |
| importance\_heavy | [0.15, 0.65, 0.20] | Reliability-critical |
| recency\_relevance | [0.40, 0.10, 0.50] | Fresh + relevant |
| importance\_relevance | [0.10, 0.40, 0.50] | Proven + relevant |
| recency\_importance | [0.40, 0.45, 0.15] | Fresh + proven |
| balanced | [0.33, 0.33, 0.34] | Park et al. baseline |
| diversity\_seeker | [0.45, 0.45, 0.10] | Anti-relevance |
| exploration | [0.25, 0.25, 0.50] | Moderate |

## Setup

### Prerequisites

- Python >= 3.11
- [uv](https://docs.astral.sh/uv/) package manager
- MiniMax API key ([platform.minimax.chat](https://www.minimax.chat/))
- ~1.2 GB disk for Qwen3-Embedding-0.6B (auto-downloads on first run)
- Optional: [Langfuse](https://langfuse.com/) account for tracing

### Installation

```bash
git clone https://github.com/kusp-dev/retrieval-weight-experiment.git
cd retrieval-weight-experiment
uv sync
```

### Configuration

```bash
cp .env.example .env
# Edit .env and set MINIMAX_API_KEY
# Optionally add Langfuse keys for observability
```

## Usage

### Run the experiment

```bash
uv run python scripts/run_full_experiment.py --dry-run  # preflight check
uv run python scripts/run_full_experiment.py             # full run (~24h)
```

### Docker

```bash
docker compose up experiment                    # run experiment
docker compose --profile test up integration-test  # run integration tests
```

### Analyze results

```bash
uv run python scripts/analyze_experiment.py     # generates figures + tables in results/
```

### Monitor progress

```bash
uv run python scripts/monitor.py --watch 30     # CLI dashboard (30s refresh)
uv run python scripts/dashboard.py              # web dashboard (Flask on port 5050)
```

### Run tests

```bash
uv run pytest tests/                            # unit tests
uv run pytest tests/ --cov=src                  # with coverage
uv run python scripts/run_integration_test.py   # API integration test (requires key)
```

## Project Structure

```
retrieval-weight-experiment/
├── configs/
│   └── experiment.yaml                 # Conditions, presets, hyperparameters
│
├── data/
│   ├── tasks/
│   │   ├── corpus_v3.json              # Task corpus (300 tasks, 50E/100M/150H)
│   │   └── corpus_annotated.json       # Tasks with ground-truth skill annotations
│   ├── skills/
│   │   └── library_v3.json             # Skill library (205 skills, 10 domains)
│   ├── ground_truth_v3.json            # Ground truth (task-skill assignments)
│   └── warm_start_priors.json          # Tier-0 priors for warm-start transfer
│
├── src/
│   ├── db/
│   │   └── schema.py                   # SQLite schema (14 tables, FTS5 indexes)
│   ├── embeddings/
│   │   └── embedder.py                 # Qwen3-Embedding-0.6B wrapper
│   ├── experiment/
│   │   ├── bandit.py                   # Thompson Sampling (Beta posteriors, 12 arms)
│   │   ├── feedback_parser.py          # Likert regex + anchor embedding parser
│   │   ├── rate_control.py             # Sliding window + step budget enforcement
│   │   ├── runner.py                   # Main experiment orchestrator
│   │   └── scoring.py                  # Dynamic recency/importance scoring
│   ├── llm/
│   │   └── minimax_client.py           # MiniMax M2.5 client
│   ├── observability/
│   │   └── tracing.py                  # Langfuse integration (graceful no-op)
│   ├── search/
│   │   └── hybrid.py                   # BM25 (FTS5) + dense + RRF fusion
│   └── utils/
│       └── config.py                   # YAML configuration loader
│
├── scripts/
│   ├── run_full_experiment.py          # Launcher (lockfile, signal handling, resume)
│   ├── run_integration_test.py         # Real API integration test
│   ├── analyze_experiment.py           # Post-experiment analysis + figure generation
│   ├── dashboard.py                    # Flask monitoring backend (SSE, live metrics)
│   ├── monitor.py                      # CLI monitoring tool
│   ├── load_skills.py                  # Skill library loader (JSON -> SQLite + FTS5)
│   ├── annotate_ground_truth.py        # Ground truth annotation for NDCG/MRR
│   └── warm_start.py                   # Tier-0 prior extraction
│
├── tests/                              # 12 test modules (unit + integration)
├── paper/                              # LaTeX source (preprint format)
├── results/                            # Pre-computed v3 experiment results
│   ├── analysis_summary.json           # Aggregate metrics per condition
│   ├── primary_metrics.csv             # Per-condition summary
│   ├── statistical_tests.json          # All statistical tests
│   └── figures/                        # Analysis plots (15 figures)
│
├── Dockerfile
├── docker-compose.yaml
├── pyproject.toml
├── CITATION.cff
└── LICENSE                             # Apache 2.0
```

## Results Data

Pre-computed results from the v3 experiment (1,200 episodes) are in `results/`:

- **`analysis_summary.json`** -- aggregate metrics, convergence analysis, theme breakdowns, ground truth evaluation, Jaccard similarity, anchor validation, and cumulative regret
- **`primary_metrics.csv`** -- per-condition summary (tokens, NDCG@5, MRR, parse rates, best arm)
- **`statistical_tests.json`** -- Kruskal--Wallis omnibus test, pairwise Mann--Whitney U tests with Bonferroni and Benjamini--Hochberg corrections, bootstrap confidence intervals, Cohen's d effect sizes
- **`figures/`** -- 15 analysis plots (reward boxplots, convergence curves, theme heatmaps, Jaccard trajectories, bandit posteriors, difficulty comparisons, etc.)
- **`rewards_c[1-4].csv`** -- per-episode reward traces for each condition

### Statistical Tests Summary

| Comparison | Token Diff | 95% CI | Cohen's d | p (Bonferroni) |
|------------|-----------|--------|-----------|----------------|
| C1 vs C2 | -695 | [-1052, -351] | -0.32 (small) | 0.0019 |
| C1 vs C3 | -687 | [-1041, -347] | -0.32 (small) | 0.0025 |
| C1 vs C4 | -1,088 | [-1418, -773] | -0.54 (medium) | <0.0001 |
| C2 vs C3 | +5 | [-293, +296] | 0.00 (negligible) | 1.0 |
| C2 vs C4 | -396 | [-667, -134] | -0.24 (small) | 0.30 |
| C3 vs C4 | -404 | [-667, -139] | -0.25 (small) | 0.15 |

### Ground Truth Retrieval Quality

| Condition | Hit Rate | Mean GT in Top-5 | NDCG@5 (graded) |
|-----------|----------|-------------------|------------------|
| C1 Control | 59.7% | 72.7% | 0.221 |
| C2 Dim Feedback | 59.3% | 77.3% | 0.243 |
| C3 Full System | 67.3% | 91.7% | 0.316 |
| C4 Qualitative | 58.0% | 77.7% | 0.240 |

### Convergence

All bandit conditions stabilize by episode 281 (out of 300). C2 and C3 converge to **pure\_relevance**, while C4 converges to **pure\_importance** -- an interpretable specialization where structured Likert feedback learns semantic matching while free-text feedback favors skill track records.

## Citation

If you use this code or data, please cite:

```bibtex
@article{dirocco2026retrieval,
  title={Gradient-Free Retrieval Weight Learning via Thompson Sampling
         with {LLM} Self-Assessment},
  author={DiRocco, Alfonso A. V.},
  year={2026},
  note={arXiv preprint}
}
```

## License

Code is released under the [Apache License 2.0](LICENSE). Data files in `data/` are released under [CC BY 4.0](data/LICENSE).
