#!/usr/bin/env python3
# DEPRECATED: Use data/generate_ground_truth_v3.py for v3 ground truth generation
"""
Annotate the 250-task corpus with ground_truth_skills and expected_approach.

Ground truth annotations are used to compute:
- NDCG@k: did the retrieval system rank relevant skills higher?
- MRR: was the best (primary) skill found first?

Each task maps to 1-3 skills:
- Primary skill: most essential for the task (listed first)
- Secondary skills: add value but not strictly necessary

The expected_approach field describes how the agent should combine
the skills to solve the task.

Usage:
    python scripts/annotate_ground_truth.py

Reads:  data/tasks/corpus.json, data/skills/library.json
Writes: data/tasks/corpus_annotated.json
"""

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS_PATH = ROOT / "data" / "tasks" / "corpus_v3.json"
SKILLS_PATH = ROOT / "data" / "skills" / "library_v3.json"
OUTPUT_PATH = ROOT / "data" / "tasks" / "corpus_annotated_v3.json"


# ============================================================
# GROUND TRUTH ANNOTATIONS
# ============================================================
# Format: task_id -> (ground_truth_skills, expected_approach)
# Skills listed in priority order (primary first).

ANNOTATIONS = {
    # ============================================================
    # ML FUNDAMENTALS (task_001 - task_050)
    # ============================================================
    # --- Easy (001-020) ---
    "task_001": (
        ["research_003", "creative_001"],
        "Use Algorithm Deep Dive to explain backprop mechanics (chain rule, gradient flow), then"
        "Technical Tutorial Development to structure the explanation clearly.",
    ),
    "task_002": (
        ["research_002", "synthesis_002"],
        "Use Comparative Technical Analysis to structure the SGD vs Adam comparison, with"
        "Trade-off Analysis for convergence/memory/use-case dimensions.",
    ),
    "task_003": (
        ["research_003", "research_006"],
        "Use Algorithm Deep Dive for scaled dot-product attention math, supplemented by Formal"
        "Definition for Q/K/V notation.",
    ),
    "task_004": (
        ["research_004", "creative_003"],
        "Use Architecture Pattern Catalog to describe CNN layers, with Architecture Diagram Design"
        "to visually map layer interactions and data flow.",
    ),
    "task_005": (
        ["research_003", "research_006"],
        "Use Algorithm Deep Dive for L1/L2 penalty mechanics, Formal Definition for the geometric"
        "interpretation.",
    ),
    "task_006": (
        ["research_002", "research_006"],
        "Use Comparative Technical Analysis to compare three loss functions, Formal Definition for"
        "mathematical objectives.",
    ),
    "task_007": (
        ["research_002", "research_003"],
        "Use Comparative Technical Analysis for RNN vs LSTM, Algorithm Deep Dive for vanishing"
        "gradient and gating mechanisms.",
    ),
    "task_008": (
        ["research_003", "creative_002"],
        "Use Algorithm Deep Dive for batch norm mechanics, Analogy-Based Explanation to make the"
        "normalization intuition clear.",
    ),
    "task_009": (
        ["research_003", "creative_005"],
        "Use Algorithm Deep Dive for transfer learning theory, Technical Blog Post Writing for"
        "accessible practitioner-oriented explanation with concrete example.",
    ),
    "task_010": (
        ["research_003", "research_002"],
        "Use Algorithm Deep Dive for few-shot learning concepts, Comparative Technical Analysis to"
        "contrast meta-learning vs in-context learning.",
    ),
    "task_011": (
        ["research_003", "creative_002"],
        "Use Algorithm Deep Dive for dropout mechanics (training vs inference), Analogy-Based"
        "Explanation for intuitive understanding.",
    ),
    "task_012": (
        ["research_003", "research_012"],
        "Use Algorithm Deep Dive for NAS search strategies, Scalability and Complexity Analysis"
        "for computational cost considerations.",
    ),
    "task_013": (
        ["research_002", "research_003"],
        "Use Comparative Technical Analysis for batch/stochastic/mini-batch comparison, Algorithm"
        "Deep Dive for convergence properties.",
    ),
    "task_014": (
        ["research_002", "creative_004"],
        "Use Comparative Technical Analysis for activation function comparison, Example Generation"
        "for use-case examples.",
    ),
    "task_015": (
        ["research_003", "synthesis_004"],
        "Use Algorithm Deep Dive for bias-variance tradeoff, Abstraction Ladder to connect"
        "low-level overfitting symptoms to high-level modeling concepts.",
    ),
    "task_016": (
        ["research_002", "research_003"],
        "Use Comparative Technical Analysis for batch/layer/instance norm comparison, Algorithm"
        "Deep Dive for normalization axis mechanics.",
    ),
    "task_017": (
        ["research_003", "research_006"],
        "Use Algorithm Deep Dive for Xavier/He initialization derivation, Formal Definition for"
        "mathematical foundations.",
    ),
    "task_018": (
        ["research_006", "creative_004"],
        "Use Formal Definition to define classification vs regression precisely, Example"
        "Generation for loss/metric examples.",
    ),
    "task_019": (
        ["research_003", "evaluation_001"],
        "Use Algorithm Deep Dive for k-fold mechanics, Metric Selection for generalization error"
        "estimation.",
    ),
    "task_020": (
        ["research_003", "creative_002"],
        "Use Algorithm Deep Dive for embedding layer mechanics, Analogy-Based Explanation for"
        "semantic relationship intuition.",
    ),
    # --- Medium (021-040) ---
    "task_021": (
        ["research_003", "research_009"],
        "Use Algorithm Deep Dive for gradient chain derivation, Failure Mode Analysis for"
        "vanishing gradient diagnosis and residual connection mitigation.",
    ),
    "task_022": (
        ["research_002", "synthesis_002"],
        "Use Comparative Technical Analysis for AdaGrad vs Proximal SGD vs Adam, Trade-off"
        "Analysis for sparsity handling mechanisms.",
    ),
    "task_023": (
        ["research_003", "evaluation_009"],
        "Use Algorithm Deep Dive for multi-head attention mechanics, Ablation Study Design for the"
        "redundancy experiment.",
    ),
    "task_024": (
        ["research_009", "creative_007"],
        "Use Failure Mode Analysis to analyze filter redundancy, Problem Reformulation to reframe"
        "whether redundancy is a bug or a feature.",
    ),
    "task_025": (
        ["research_003", "research_006"],
        "Use Algorithm Deep Dive for focal loss derivation and comparison, Formal Definition for"
        "mathematical difference.",
    ),
    "task_026": (
        ["research_003", "evaluation_003"],
        "Use Algorithm Deep Dive for dropout rate effects, A/B Test Design for principled"
        "hyperparameter selection with validation.",
    ),
    "task_027": (
        ["research_002", "creative_004"],
        "Use Comparative Technical Analysis for meta-learning vs transfer learning, Example"
        "Generation for concrete MAML scenario.",
    ),
    "task_028": (
        ["synthesis_010", "research_012"],
        "Use Design Space Exploration for NAS search space definition, Scalability and Complexity"
        "Analysis for compute constraints.",
    ),
    "task_029": (
        ["synthesis_002", "research_012"],
        "Use Trade-off Analysis for memory vs computation trade-off, Scalability and Complexity"
        "Analysis for batch size interactions.",
    ),
    "task_030": (
        ["research_003", "synthesis_002"],
        "Use Algorithm Deep Dive for FP16/FP32 and loss scaling mechanics, Trade-off Analysis for"
        "speedup vs stability risks.",
    ),
    "task_031": (
        ["operational_003", "creative_009"],
        "Use Pipeline Design for curriculum ordering strategy, Scenario Planning for difficulty"
        "progression adaptation.",
    ),
    "task_032": (
        ["operational_003", "research_003"],
        "Use Pipeline Design for distillation process, Algorithm Deep Dive for hard/soft target"
        "mechanics and objective weighting.",
    ),
    "task_033": (
        ["research_003", "evaluation_003"],
        "Use Algorithm Deep Dive for adversarial training mechanics, A/B Test Design for clean vs"
        "robust accuracy frontier experiment.",
    ),
    "task_034": (
        ["creative_008", "synthesis_008"],
        "Use Visualization Design for attention pattern visualization, Assumption Surfacing to"
        "distinguish genuine reasoning from noise.",
    ),
    "task_035": (
        ["operational_003", "evaluation_004"],
        "Use Pipeline Design for augmentation pipeline, Bias Detection for validating augmentation"
        "doesn't introduce bias.",
    ),
    "task_036": (
        ["evaluation_003", "operational_004"],
        "Use A/B Test Design for the (batch_size, lr) experiment, Configuration Management for"
        "systematic hyperparameter exploration.",
    ),
    "task_037": (
        ["research_002", "research_003"],
        "Use Comparative Technical Analysis for EWC vs replay buffers, Algorithm Deep Dive for"
        "mechanisms and computational costs.",
    ),
    "task_038": (
        ["research_002", "synthesis_002"],
        "Use Comparative Technical Analysis for pruning vs quantization, Trade-off Analysis for"
        "latency vs accuracy and combination strategies.",
    ),
    "task_039": (
        ["research_003", "synthesis_010"],
        "Use Algorithm Deep Dive for stacking/meta-model mechanics, Design Space Exploration for"
        "sophisticated ensemble strategies.",
    ),
    "task_040": (
        ["research_002", "synthesis_002"],
        "Use Comparative Technical Analysis across 5 imbalance methods, Trade-off Analysis for"
        "bias-variance and computational implications.",
    ),
    # --- Hard (041-050) ---
    "task_041": (
        ["research_004", "operational_003", "creative_003"],
        "Use Architecture Pattern Catalog for system architecture, Pipeline Design for end-to-end"
        "pipeline, Architecture Diagram Design for visual system documentation.",
    ),
    "task_042": (
        ["research_001", "research_005", "research_003"],
        "Use Systematic Literature Review for Kaplan/Hoffmann papers, Empirical Evidence Synthesis"
        "for scaling law results, Algorithm Deep Dive for Chinchilla analysis.",
    ),
    "task_043": (
        ["research_006", "research_003"],
        "Use Formal Definition for VC dimension/PAC learning bounds, Algorithm Deep Dive for"
        "regularization-generalization connection.",
    ),
    "task_044": (
        ["research_004", "operational_003", "research_012"],
        "Use Architecture Pattern Catalog for federated learning design, Pipeline Design for"
        "communication protocol, Scalability Analysis for client management.",
    ),
    "task_045": (
        ["research_003", "evaluation_003", "research_006"],
        "Use Algorithm Deep Dive for causal inference theory, A/B Test Design for interventional"
        "experiment, Formal Definition for causal graph notation.",
    ),
    "task_046": (
        ["research_004", "synthesis_002", "research_002"],
        "Use Architecture Pattern Catalog for system design, Trade-off Analysis for accuracy vs"
        "interpretability, Comparative Analysis for model types.",
    ),
    "task_047": (
        ["evaluation_005", "research_009", "synthesis_005"],
        "Use Benchmark Protocol Design for robust evaluation, Failure Mode Analysis for benchmark"
        "gaming diagnosis, Gap Analysis for identifying what benchmarks miss vs real-world.",
    ),
    "task_048": (
        ["research_004", "research_003", "synthesis_002"],
        "Use Architecture Pattern Catalog for MTL system design, Algorithm Deep Dive for"
        "conflicting gradient handling, Trade-off Analysis for sharing strategies.",
    ),
    "task_049": (
        ["research_002", "synthesis_010", "research_003"],
        "Use Comparative Technical Analysis for search methods, Design Space Exploration for"
        "meta-learning approach, Algorithm Deep Dive for Bayesian optimization.",
    ),
    "task_050": (
        ["research_004", "synthesis_006", "research_003"],
        "Use Architecture Pattern Catalog for adaptation framework, Conceptual Framework Building"
        "for unifying LoRA/adapters/multi-task concepts, Algorithm Deep Dive for"
        "parameter-efficient mechanics.",
    ),
    # ============================================================
    # AGENT ARCHITECTURES (task_051 - task_100)
    # ============================================================
    # --- Easy (051-070) ---
    "task_051": (
        ["research_004", "creative_004"],
        "Use Architecture Pattern Catalog for ReAct framework components, Example Generation for a"
        "demonstration prompt.",
    ),
    "task_052": (
        ["research_003", "creative_004"],
        "Use Algorithm Deep Dive for CoT mechanics, Example Generation for concrete CoT"
        "demonstration.",
    ),
    "task_053": (
        ["research_007", "creative_004"],
        "Use Taxonomy and Classification to categorize tool types, Example Generation for three"
        "tool examples.",
    ),
    "task_054": (
        ["research_004", "research_007"],
        "Use Architecture Pattern Catalog for AutoGPT components, Taxonomy to classify"
        "memory/planning/execution subsystems.",
    ),
    "task_055": (
        ["research_002", "research_003"],
        "Use Comparative Technical Analysis for BFS/DFS/A*, Algorithm Deep Dive for heuristic"
        "definition.",
    ),
    "task_056": (
        ["research_007", "creative_010"],
        "Use Taxonomy for episodic/semantic/procedural classification, Concept Map Construction to"
        "map relationships between memory types and agent behaviors.",
    ),
    "task_057": (
        ["research_003", "creative_004"],
        "Use Algorithm Deep Dive for reflection mechanisms, Example Generation for concrete"
        "implementation.",
    ),
    "task_058": (
        ["research_003", "research_006"],
        "Use Algorithm Deep Dive for function calling flow, Formal Definition for JSON schema"
        "format.",
    ),
    "task_059": (
        ["research_004", "creative_004"],
        "Use Architecture Pattern Catalog for chaining patterns, Example Generation for two"
        "concrete examples.",
    ),
    "task_060": (
        ["research_009", "operational_001"],
        "Use Failure Mode Analysis for error types, System Debugging for recovery strategies.",
    ),
    "task_061": (
        ["evaluation_001", "evaluation_002"],
        "Use Metric Selection for agent evaluation metrics, Evaluation Rubric Design for"
        "measurement criteria.",
    ),
    "task_062": (
        ["research_002", "research_012"],
        "Use Comparative Technical Analysis for CoT vs ToT, Scalability and Complexity for"
        "exponential cost management.",
    ),
    "task_063": (
        ["research_007", "creative_004"],
        "Use Taxonomy for grounding mechanism types, Example Generation for concrete mechanism"
        "examples.",
    ),
    "task_064": (
        ["research_004", "synthesis_002"],
        "Use Architecture Pattern Catalog for state management design, Trade-off Analysis for"
        "completeness vs context limits.",
    ),
    "task_065": (
        ["research_003", "evaluation_001"],
        "Use Algorithm Deep Dive for uncertainty quantification methods, Metric Selection for"
        "confidence expression.",
    ),
    "task_066": (
        ["research_009", "operational_001"],
        "Use Failure Mode Analysis for ReAct loop failures, System Debugging Protocol for infinite"
        "loop detection and breaking.",
    ),
    "task_067": (
        ["synthesis_002", "research_004"],
        "Use Trade-off Analysis for reasoning vs tool-calling balance, Architecture Pattern"
        "Catalog for interleaving design.",
    ),
    "task_068": (
        ["research_004", "operational_003"],
        "Use Architecture Pattern Catalog for hierarchical decomposition, Pipeline Design for"
        "task/subtask/atomic action levels.",
    ),
    "task_069": (
        ["synthesis_002", "operational_003"],
        "Use Trade-off Analysis for verbatim vs summarize vs discard decisions, Pipeline Design"
        "for memory management strategy.",
    ),
    "task_070": (
        ["evaluation_002", "research_004"],
        "Use Evaluation Rubric Design for reflection criteria, Architecture Pattern Catalog for"
        "reflection mechanism structure.",
    ),
    # --- Medium (071-090) ---
    "task_071": (
        ["research_006", "operational_008"],
        "Use Formal Definition for function specification format, Testing Strategy for validation"
        "approach.",
    ),
    "task_072": (
        ["research_004", "operational_003"],
        "Use Architecture Pattern Catalog for branching chain design, Pipeline Design for state"
        "management across branches.",
    ),
    "task_073": (
        ["operational_001", "research_009"],
        "Use System Debugging for recovery strategy, Failure Mode Analysis for error loop"
        "prevention.",
    ),
    "task_074": (
        ["research_004", "operational_003"],
        "Use Architecture Pattern Catalog for multi-agent communication, Pipeline Design for"
        "coordination protocol.",
    ),
    "task_075": (
        ["operational_005", "synthesis_002"],
        "Use Resource Estimation for cost modeling,"
        " Trade-off Analysis for quality vs cost balance.",
    ),
    "task_076": (
        ["research_004", "operational_006"],
        "Use Architecture Pattern Catalog for guardrail design, Monitoring and Observability for"
        "safety monitors.",
    ),
    "task_077": (
        ["operational_001", "operational_006"],
        "Use System Debugging for failure investigation, Monitoring and Observability for logging"
        "system design.",
    ),
    "task_078": (
        ["operational_003", "operational_005"],
        "Use Pipeline Design for task queue and agent pool, Resource Estimation for capacity"
        "planning.",
    ),
    "task_079": (
        ["synthesis_002", "research_004"],
        "Use Trade-off Analysis for tool cost-benefit, Architecture Pattern Catalog for dynamic"
        "selection mechanism.",
    ),
    "task_080": (
        ["evaluation_006", "synthesis_007"],
        "Use Error Analysis for failure pattern categorization, Theme Clustering for insight"
        "extraction from failure logs.",
    ),
    "task_081": (
        ["research_004", "operational_003"],
        "Use Architecture Pattern Catalog for agent roles and quality gates, Pipeline Design for"
        "orchestration flow.",
    ),
    "task_082": (
        ["synthesis_007", "research_003"],
        "Use Theme Clustering for behavioral pattern identification, Algorithm Deep Dive for"
        "causal modeling approach.",
    ),
    "task_083": (
        ["evaluation_003", "research_005"],
        "Use A/B Test Design for emergence measurement experiments, Empirical Evidence Synthesis"
        "for analyzing results.",
    ),
    "task_084": (
        ["research_004", "evaluation_003"],
        "Use Architecture Pattern Catalog for self-improvement system, A/B Test Design for"
        "generalization validation.",
    ),
    "task_085": (
        ["evaluation_005", "research_009"],
        "Use Benchmark Protocol Design for robustness measurement, Failure Mode Analysis for"
        "adversarial defense design.",
    ),
    "task_086": (
        ["operational_003", "evaluation_002"],
        "Use Pipeline Design for curriculum progression, Evaluation Rubric Design for task"
        "difficulty and progression criteria.",
    ),
    "task_087": (
        ["research_003", "research_005"],
        "Use Algorithm Deep Dive for meta-learning encoding, Empirical Evidence Synthesis for"
        "optimal prompt analysis.",
    ),
    "task_088": (
        ["research_004", "operational_003"],
        "Use Architecture Pattern Catalog for collaborative framework, Pipeline Design for role"
        "and communication protocols.",
    ),
    "task_089": (
        ["research_003", "research_006"],
        "Use Algorithm Deep Dive for verification methods, Formal Definition for tractability"
        "analysis.",
    ),
    "task_090": (
        ["synthesis_010", "research_004"],
        "Use Design Space Exploration for architecture decision space, Architecture Pattern"
        "Catalog for per-task adaptation.",
    ),
    # --- Hard (091-100) ---
    "task_091": (
        ["research_003", "research_005", "research_006"],
        "Use Algorithm Deep Dive for IRL algorithms, Empirical Evidence Synthesis for trajectory"
        "analysis, Formal Definition for reward function formulation.",
    ),
    "task_092": (
        ["operational_006", "operational_003", "research_004"],
        "Use Monitoring Design for health metrics, Pipeline Design for adaptation pipeline,"
        "Architecture Pattern Catalog for system structure.",
    ),
    "task_093": (
        ["research_004", "operational_003", "synthesis_003"],
        "Use Architecture Pattern Catalog for knowledge graph design, Pipeline Design for graph"
        "maintenance, Multi-Source Integration for conflict resolution.",
    ),
    "task_094": (
        ["operational_003", "operational_005", "research_012"],
        "Use Pipeline Design for distributed framework, Resource Estimation for capacity planning,"
        "Scalability Analysis for fault tolerance.",
    ),
    "task_095": (
        ["operational_003", "evaluation_008", "synthesis_007"],
        "Use Pipeline Design for improvement pipeline, Human Evaluation Design for correction"
        "collection, Theme Clustering for suboptimal decision patterns.",
    ),
    "task_096": (
        ["evaluation_005", "evaluation_002", "research_011"],
        "Use Benchmark Protocol Design for benchmark suite, Evaluation Rubric Design for success"
        "criteria, Dataset and Benchmark Analysis for difficulty calibration.",
    ),
    "task_097": (
        ["research_004", "operational_008", "synthesis_001"],
        "Use Architecture Pattern Catalog for skill composition, Testing Strategy for correctness"
        "validation, Cross-Domain Pattern Extraction for skill interactions.",
    ),
    "task_098": (
        ["creative_006", "research_003", "evaluation_003"],
        "Use Thought Experiment for counterfactual generation, Algorithm Deep Dive for impact"
        "measurement, A/B Test Design for contrastive training.",
    ),
    "task_099": (
        ["research_004", "research_012", "synthesis_002"],
        "Use Architecture Pattern Catalog for privacy mechanisms, Scalability Analysis for"
        "privacy-capability tradeoff, Trade-off Analysis for mechanism selection.",
    ),
    "task_100": (
        ["evaluation_004", "evaluation_002", "research_004"],
        "Use Bias Detection for bias identification, Evaluation Rubric Design for fairness"
        "auditing, Architecture Pattern Catalog for mitigation system.",
    ),
    # ============================================================
    # RETRIEVAL SYSTEMS (task_101 - task_150)
    # ============================================================
    # --- Easy (101-120) ---
    "task_101": (
        ["research_004", "research_002"],
        "Use Architecture Pattern Catalog for RAG pipeline components, Comparative Technical"
        "Analysis for RAG vs fine-tuning.",
    ),
    "task_102": (
        ["research_002", "research_003"],
        "Use Comparative Technical Analysis for dense vs sparse, Algorithm Deep Dive for embedding"
        "nearest neighbor mechanics.",
    ),
    "task_103": (
        ["research_011", "evaluation_001"],
        "Use Dataset and Benchmark Analysis for MTEB benchmarks, Metric Selection for nDCG/MRR"
        "explanation.",
    ),
    "task_104": (
        ["research_003", "creative_004"],
        "Use Algorithm Deep Dive for query expansion techniques, Example Generation for concrete"
        "expansion example.",
    ),
    "task_105": (
        ["research_002", "research_004"],
        "Use Comparative Technical Analysis for FAISS/Pinecone/Weaviate, Architecture Pattern"
        "Catalog for indexing structures.",
    ),
    "task_106": (
        ["research_002", "synthesis_002"],
        "Use Comparative Technical Analysis for chunking strategies, Trade-off Analysis for"
        "granularity trade-offs.",
    ),
    "task_107": (
        ["research_003", "synthesis_002"],
        "Use Algorithm Deep Dive for cross-encoder vs bi-encoder, Trade-off Analysis for"
        "computational trade-off.",
    ),
    "task_108": (
        ["research_002", "research_004"],
        "Use Comparative Technical Analysis for graph vs embedding retrieval, Architecture Pattern"
        "Catalog for knowledge graph structure.",
    ),
    "task_109": (
        ["research_003", "research_006"],
        "Use Algorithm Deep Dive for BM25 formula and parameters, Formal Definition for"
        "mathematical comparison with TF-IDF.",
    ),
    "task_110": (
        ["evaluation_001", "research_006"],
        "Use Metric Selection for nDCG/MRR/MAP explanation, Formal Definition for mathematical"
        "definitions.",
    ),
    "task_111": (
        ["research_002", "research_004"],
        "Use Comparative Technical Analysis for RRF vs learned weights, Architecture Pattern"
        "Catalog for hybrid search design.",
    ),
    "task_112": (
        ["research_009", "synthesis_005"],
        "Use Failure Mode Analysis for cold start diagnosis, Gap Analysis for identifying what's"
        "missing and how to bootstrap.",
    ),
    "task_113": (
        ["research_002", "research_004"],
        "Use Comparative Technical Analysis for personalization approaches, Architecture Pattern"
        "Catalog for system design.",
    ),
    "task_114": (
        ["research_003", "evaluation_004"],
        "Use Algorithm Deep Dive for feedback loop mechanics, Bias Detection for position bias"
        "handling.",
    ),
    "task_115": (
        ["research_003", "research_002"],
        "Use Algorithm Deep Dive for semantic search mechanics, Comparative Technical Analysis for"
        "vs keyword search.",
    ),
    "task_116": (
        ["research_004", "operational_003"],
        "Use Architecture Pattern Catalog for multi-stage RAG, Pipeline Design for multi-hop"
        "retrieval orchestration.",
    ),
    "task_117": (
        ["research_009", "evaluation_003"],
        "Use Failure Mode Analysis for temporal bias diagnosis, A/B Test Design for correction"
        "experiments.",
    ),
    "task_118": (
        ["operational_003", "research_003"],
        "Use Pipeline Design for expansion system, Algorithm Deep Dive for intent analysis and"
        "term weighting.",
    ),
    "task_119": (
        ["research_002", "operational_005"],
        "Use Comparative Technical Analysis for HNSW/IVF/SCANN, Resource Estimation for"
        "memory/speed constraints.",
    ),
    "task_120": (
        ["research_003", "operational_003"],
        "Use Algorithm Deep Dive for semantic boundary detection, Pipeline Design for chunking"
        "algorithm.",
    ),
    # --- Medium (121-140) ---
    "task_121": (
        ["operational_003", "research_003"],
        "Use Pipeline Design for LTR system, Algorithm Deep Dive for ranking model training.",
    ),
    "task_122": (
        ["research_009", "operational_001"],
        "Use Failure Mode Analysis for retrieval failure detection, System Debugging for recovery"
        "mechanisms.",
    ),
    "task_123": (
        ["research_003", "research_004"],
        "Use Algorithm Deep Dive for TransE/GCN embeddings, Architecture Pattern Catalog for graph"
        "query system.",
    ),
    "task_124": (
        ["research_004", "research_003"],
        "Use Architecture Pattern Catalog for multimodal retrieval design, Algorithm Deep Dive for"
        "CLIP-style alignment.",
    ),
    "task_125": (
        ["synthesis_002", "research_003"],
        "Use Trade-off Analysis for recency vs quality balance, Algorithm Deep Dive for temporal"
        "weighting scheme.",
    ),
    "task_126": (
        ["operational_003", "evaluation_005"],
        "Use Pipeline Design for domain adaptation, Benchmark Protocol Design for domain-specific"
        "evaluation.",
    ),
    "task_127": (
        ["operational_005", "research_012"],
        "Use Resource Estimation for capacity constraints, Scalability Analysis for distribution"
        "and quantization strategy.",
    ),
    "task_128": (
        ["research_003", "synthesis_002"],
        "Use Algorithm Deep Dive for MMR/DPP, Trade-off Analysis for diversity vs relevance.",
    ),
    "task_129": (
        ["operational_003", "synthesis_003"],
        "Use Pipeline Design for multi-document retrieval, Multi-Source Integration for passage"
        "combination.",
    ),
    "task_130": (
        ["research_002", "research_003"],
        "Use Comparative Technical Analysis for multilingual approaches, Algorithm Deep Dive for"
        "cross-lingual embeddings.",
    ),
    "task_131": (
        ["operational_002", "operational_003"],
        "Use Performance Profiling for pipeline optimization, Pipeline Design for end-to-end RAG"
        "system.",
    ),
    "task_132": (
        ["research_004", "research_003"],
        "Use Architecture Pattern Catalog for adaptive system design, Algorithm Deep Dive for"
        "query complexity estimation.",
    ),
    "task_133": (
        ["creative_008", "evaluation_008"],
        "Use Visualization Design for explanation generation, Human Evaluation Design for"
        "explanation validation.",
    ),
    "task_134": (
        ["synthesis_003", "synthesis_008"],
        "Use Multi-Source Integration for contradiction detection, Assumption Surfacing for"
        "conflict resolution strategy.",
    ),
    "task_135": (
        ["operational_003", "research_003"],
        "Use Pipeline Design for continual learning system, Algorithm Deep Dive for incremental"
        "embedding updates.",
    ),
    "task_136": (
        ["research_002", "evaluation_005"],
        "Use Comparative Technical Analysis for code retrieval approaches, Benchmark Protocol"
        "Design for code search evaluation.",
    ),
    "task_137": (
        ["research_003", "synthesis_002"],
        "Use Algorithm Deep Dive for homomorphic encryption/MPC, Trade-off Analysis for"
        "privacy-utility measurement.",
    ),
    "task_138": (
        ["research_003", "creative_006"],
        "Use Algorithm Deep Dive for causal graph mechanics, Thought Experiment for counterfactual"
        "reasoning design.",
    ),
    "task_139": (
        ["research_003", "evaluation_003"],
        "Use Algorithm Deep Dive for Thompson Sampling/RL weight learning, A/B Test Design for"
        "dynamic vs static weight comparison.",
    ),
    "task_140": (
        ["evaluation_005", "research_009"],
        "Use Benchmark Protocol Design for robustness evaluation framework, Failure Mode Analysis"
        "for degradation analysis.",
    ),
    # --- Hard (141-150) ---
    "task_141": (
        ["research_003", "research_005", "synthesis_010"],
        "Use Algorithm Deep Dive for meta-learning, Empirical Evidence Synthesis for"
        "hyperparameter dataset, Design Space Exploration for task encoding.",
    ),
    "task_142": (
        ["operational_003", "research_003", "evaluation_003"],
        "Use Pipeline Design for retrieval-augmented fine-tuning, Algorithm Deep Dive for"
        "curriculum design, A/B Test for broad vs targeted retrieval.",
    ),
    "task_143": (
        ["research_003", "research_004", "evaluation_005"],
        "Use Algorithm Deep Dive for GNN mechanics, Architecture Pattern Catalog for graph"
        "retrieval design, Benchmark Protocol for performance comparison.",
    ),
    "task_144": (
        ["evaluation_004", "evaluation_001", "research_004"],
        "Use Bias Detection for retrieval bias analysis, Metric Selection for fairness metrics,"
        "Architecture Pattern Catalog for mitigation design.",
    ),
    "task_145": (
        ["evaluation_005", "research_005", "research_003"],
        "Use Benchmark Protocol for generalization testing, Empirical Evidence Synthesis for"
        "cross-domain results, Algorithm Deep Dive for domain-agnostic design.",
    ),
    "task_146": (
        ["research_003", "research_004", "evaluation_003"],
        "Use Algorithm Deep Dive for instruction-tuned embeddings, Architecture Pattern Catalog"
        "for conditioned retrieval, A/B Test for instruction comparison.",
    ),
    "task_147": (
        ["research_004", "evaluation_005", "synthesis_008"],
        "Use Architecture Pattern Catalog for fact verification retrieval, Benchmark Protocol for"
        "evidence evaluation, Assumption Surfacing for fact validation.",
    ),
    "task_148": (
        ["operational_002", "operational_005", "research_003"],
        "Use Performance Profiling for mobile optimization, Resource Estimation for mobile"
        "constraints, Algorithm Deep Dive for quantization/distillation.",
    ),
    "task_149": (
        ["research_003", "evaluation_003", "synthesis_001"],
        "Use Algorithm Deep Dive for interaction detection, A/B Test for synergy measurement,"
        "Cross-Domain Pattern Extraction for interaction exploitation.",
    ),
    "task_150": (
        ["evaluation_005", "evaluation_008", "research_011"],
        "Use Benchmark Protocol for benchmark design, Human Evaluation Design for annotation"
        "collection, Dataset and Benchmark Analysis for baseline establishment.",
    ),
    # ============================================================
    # MULTI-AGENT SYSTEMS (task_151 - task_200)
    # ============================================================
    # --- Easy (151-170) ---
    "task_151": (
        ["research_007", "creative_004"],
        "Use Taxonomy for role classification, Example Generation for specialization example.",
    ),
    "task_152": (
        ["research_002", "research_004"],
        "Use Comparative Technical Analysis for communication protocols, Architecture Pattern"
        "Catalog for system design.",
    ),
    "task_153": (
        ["research_002", "synthesis_002"],
        "Use Comparative Technical Analysis for allocation strategies, Trade-off Analysis for"
        "centralized vs decentralized trade-offs.",
    ),
    "task_154": (
        ["research_007", "creative_004"],
        "Use Taxonomy for coordination mechanism classification, Example Generation for concrete"
        "mechanism examples.",
    ),
    "task_155": (
        ["research_002", "research_003"],
        "Use Comparative Technical Analysis for consensus algorithms, Algorithm Deep Dive for"
        "Byzantine fault tolerance.",
    ),
    "task_156": (
        ["research_002", "synthesis_002"],
        "Use Comparative Technical Analysis for conflict resolution approaches, Trade-off Analysis"
        "for approach selection.",
    ),
    "task_157": (
        ["research_012", "research_002"],
        "Use Scalability and Complexity Analysis for communication scaling, Comparative Technical"
        "Analysis for protocol comparison.",
    ),
    "task_158": (
        ["research_003", "creative_004"],
        "Use Algorithm Deep Dive for emergence mechanics, Example Generation for two concrete"
        "emergence examples.",
    ),
    "task_159": (
        ["research_002", "research_003"],
        "Use Comparative Technical Analysis for learning approaches, Algorithm Deep Dive for"
        "federated/decentralized learning.",
    ),
    "task_160": (
        ["research_003", "creative_004"],
        "Use Algorithm Deep Dive for debate framework mechanics, Example Generation for concrete"
        "debate example.",
    ),
    "task_161": (
        ["research_002", "synthesis_002"],
        "Use Comparative Technical Analysis for hierarchical vs flat, Trade-off Analysis for"
        "communication/responsiveness/robustness.",
    ),
    "task_162": (
        ["research_004", "operational_003"],
        "Use Architecture Pattern Catalog for shared knowledge system, Pipeline Design for"
        "consistency maintenance.",
    ),
    "task_163": (
        ["operational_006", "research_004"],
        "Use Monitoring and Observability for audit trail design, Architecture Pattern Catalog for"
        "accountability mechanism.",
    ),
    "task_164": (
        ["operational_003", "research_004"],
        "Use Pipeline Design for coordination protocol, Architecture Pattern Catalog for message"
        "types and error handling.",
    ),
    "task_165": (
        ["research_003", "operational_005"],
        "Use Algorithm Deep Dive for optimization formulation, Resource Estimation for task-skill"
        "matching.",
    ),
    "task_166": (
        ["synthesis_010", "research_002"],
        "Use Design Space Exploration for team composition, Comparative Technical Analysis for"
        "diversity effects.",
    ),
    "task_167": (
        ["synthesis_002", "operational_003"],
        "Use Trade-off Analysis for information priority, Pipeline Design for efficient"
        "communication protocol.",
    ),
    "task_168": (
        ["research_009", "operational_001"],
        "Use Failure Mode Analysis for failure detection, System Debugging for task reassignment"
        "and recovery.",
    ),
    "task_169": (
        ["research_002", "research_003"],
        "Use Comparative Technical Analysis for planning approaches, Algorithm Deep Dive for"
        "decentralized planning algorithms.",
    ),
    "task_170": (
        ["evaluation_002", "research_003"],
        "Use Evaluation Rubric Design for debate format criteria, Algorithm Deep Dive for voting"
        "and collusion detection.",
    ),
    # --- Medium (171-190) ---
    "task_171": (
        ["research_003", "synthesis_002"],
        "Use Algorithm Deep Dive for game-theoretic incentive design, Trade-off Analysis for"
        "cooperation stability.",
    ),
    "task_172": (
        ["synthesis_003", "synthesis_008"],
        "Use Multi-Source Integration for observation merging, Assumption Surfacing for"
        "contradiction handling.",
    ),
    "task_173": (
        ["synthesis_002", "research_004"],
        "Use Trade-off Analysis for individual vs team goal balance, Architecture Pattern Catalog"
        "for alignment framework.",
    ),
    "task_174": (
        ["creative_009", "research_003"],
        "Use Scenario Planning for specialization emergence, Algorithm Deep Dive for beneficial"
        "specialization mechanisms.",
    ),
    "task_175": (
        ["research_003", "research_002"],
        "Use Algorithm Deep Dive for MARL algorithms, Comparative Technical Analysis for"
        "centralized vs decentralized training.",
    ),
    "task_176": (
        ["operational_003", "operational_005"],
        "Use Pipeline Design for distributed processing, Resource Estimation for data distribution"
        "and straggler handling.",
    ),
    "task_177": (
        ["evaluation_002", "research_003"],
        "Use Evaluation Rubric Design for trust metrics, Algorithm Deep Dive for reputation update"
        "mechanisms.",
    ),
    "task_178": (
        ["research_002", "research_003"],
        "Use Comparative Technical Analysis for optimization approaches, Algorithm Deep Dive for"
        "federated optimization.",
    ),
    "task_179": (
        ["research_012", "research_004"],
        "Use Scalability Analysis for 1,000-agent communication, Architecture Pattern Catalog for"
        "hierarchical/gossip design.",
    ),
    "task_180": (
        ["operational_008", "operational_001"],
        "Use Testing Strategy Design for multi-agent testing, System Debugging for non-determinism"
        "and race condition handling.",
    ),
    "task_181": (
        ["operational_003", "research_004", "evaluation_003"],
        "Use Pipeline Design for large-scale protocol, Architecture Pattern Catalog for deadlock"
        "detection, A/B Test for simulation validation.",
    ),
    "task_182": (
        ["creative_009", "evaluation_003"],
        "Use Scenario Planning for strategy emergence, A/B Test Design for strategy measurement"
        "experiments.",
    ),
    "task_183": (
        ["research_005", "evaluation_001"],
        "Use Empirical Evidence Synthesis for diversity analysis, Metric Selection for diversity"
        "metrics.",
    ),
    "task_184": (
        ["research_003", "research_012"],
        "Use Algorithm Deep Dive for Byzantine fault tolerance, Scalability Analysis for"
        "adversarial agent tolerance bounds.",
    ),
    "task_185": (
        ["evaluation_002", "research_003"],
        "Use Evaluation Rubric Design for debate format criteria, Algorithm Deep Dive for"
        "game-theoretic justification.",
    ),
    "task_186": (
        ["research_003", "research_006"],
        "Use Algorithm Deep Dive for differential equation modeling, Formal Definition for"
        "equilibria formalization.",
    ),
    "task_187": (
        ["research_003", "research_002"],
        "Use Algorithm Deep Dive for prediction market mechanics, Comparative Technical Analysis"
        "for vs voting/aggregation.",
    ),
    "task_188": (
        ["research_003", "research_006"],
        "Use Algorithm Deep Dive for multi-agent IRL, Formal Definition for reward function and"
        "interdependency formalization.",
    ),
    "task_189": (
        ["research_003", "evaluation_001"],
        "Use Algorithm Deep Dive for emergent communication, Metric Selection for protocol"
        "interpretability metrics.",
    ),
    "task_190": (
        ["synthesis_010", "synthesis_002"],
        "Use Design Space Exploration for organization structure options, Trade-off Analysis for"
        "reorganization cost vs benefit.",
    ),
    # --- Hard (191-200) ---
    "task_191": (
        ["evaluation_004", "research_006", "synthesis_002"],
        "Use Bias Detection for fairness analysis, Formal Definition for fairness formalization,"
        "Trade-off Analysis for distributional fairness.",
    ),
    "task_192": (
        ["research_006", "operational_008", "research_003"],
        "Use Formal Definition for safety/liveness properties, Testing Strategy for verification"
        "approach, Algorithm Deep Dive for formal methods.",
    ),
    "task_193": (
        ["research_004", "research_012", "operational_003"],
        "Use Architecture Pattern Catalog for knowledge sharing design, Scalability Analysis for"
        "communication overhead, Pipeline Design for consistency.",
    ),
    "task_194": (
        ["research_003", "creative_009", "operational_003"],
        "Use Algorithm Deep Dive for robust planning, Scenario Planning for contingency design,"
        "Pipeline Design for reactive replanning.",
    ),
    "task_195": (
        ["research_003", "evaluation_003", "synthesis_001"],
        "Use Algorithm Deep Dive for collaboration meta-learning, A/B Test for novel team testing,"
        "Cross-Domain Pattern for adaptable strategies.",
    ),
    "task_196": (
        ["research_003", "research_006", "research_012"],
        "Use Algorithm Deep Dive for consensus algorithms, Formal Definition for convergence"
        "guarantees, Scalability Analysis for communication complexity.",
    ),
    "task_197": (
        ["evaluation_001", "evaluation_005", "research_005"],
        "Use Metric Selection for collective intelligence metrics, Benchmark Protocol for"
        "comparison framework, Empirical Evidence Synthesis for analysis.",
    ),
    "task_198": (
        ["research_003", "research_006", "evaluation_003"],
        "Use Algorithm Deep Dive for mechanism design, Formal Definition for incentive"
        "compatibility, A/B Test for manipulation resistance.",
    ),
    "task_199": (
        ["operational_003", "research_003", "research_012"],
        "Use Pipeline Design for temporal coordination, Algorithm Deep Dive for synchronization"
        "mechanisms, Scalability Analysis for timing constraints.",
    ),
    "task_200": (
        ["operational_003", "research_004", "operational_006"],
        "Use Pipeline Design for long-horizon orchestration, Architecture Pattern Catalog for role"
        "evolution, Monitoring Design for progress tracking.",
    ),
    # ============================================================
    # EVALUATION METHODS (task_201 - task_250)
    # ============================================================
    # --- Easy (201-220) ---
    "task_201": (
        ["evaluation_002", "research_009"],
        "Use Evaluation Rubric Design for LLM judge rubric and prompt, Failure Mode Analysis for"
        "judge bias failure modes.",
    ),
    "task_202": (
        ["research_011", "research_007"],
        "Use Dataset and Benchmark Analysis for benchmark descriptions, Taxonomy for benchmark"
        "classification.",
    ),
    "task_203": (
        ["evaluation_008", "evaluation_002"],
        "Use Human Evaluation Design for protocol structure, Evaluation Rubric Design for"
        "annotator instructions.",
    ),
    "task_204": (
        ["evaluation_007", "research_006"],
        "Use Statistical Significance Testing for agreement metrics, Formal Definition for"
        "Kappa/Alpha formulas.",
    ),
    "task_205": (
        ["evaluation_001", "research_002"],
        "Use Metric Selection for BLEU/ROUGE/METEOR explanation, Comparative Technical Analysis"
        "for limitations comparison.",
    ),
    "task_206": (
        ["evaluation_004", "evaluation_001"],
        "Use Bias Detection for bias identification methods, Metric Selection for demographic"
        "parity and equalized odds.",
    ),
    "task_207": (
        ["evaluation_005", "creative_004"],
        "Use Benchmark Protocol Design for robustness test design, Example Generation for image"
        "classifier test examples.",
    ),
    "task_208": (
        ["evaluation_001", "research_003"],
        "Use Metric Selection for calibration metrics, Algorithm Deep Dive for calibration"
        "improvement methods.",
    ),
    "task_209": (
        ["evaluation_003", "evaluation_007"],
        "Use A/B Test Design for model comparison, Statistical Significance Testing for sample"
        "size and significance.",
    ),
    "task_210": (
        ["evaluation_004", "synthesis_002"],
        "Use Bias Detection for fairness metric definitions, Trade-off Analysis for metric"
        "conflicts.",
    ),
    "task_211": (
        ["research_007", "creative_004"],
        "Use Taxonomy for interpretability vs explainability distinction, Example Generation for"
        "three techniques.",
    ),
    "task_212": (
        ["research_009", "creative_004"],
        "Use Failure Mode Analysis for reward hacking mechanisms, Example Generation for concrete"
        "hacking examples.",
    ),
    "task_213": (
        ["research_002", "evaluation_007"],
        "Use Comparative Technical Analysis for cross-validation strategies, Statistical"
        "Significance Testing for data leakage pitfalls.",
    ),
    "task_214": (
        ["evaluation_007", "research_006"],
        "Use Statistical Significance Testing for p-values and power analysis, Formal Definition"
        "for mathematical formulation.",
    ),
    "task_215": (
        ["evaluation_002", "evaluation_004"],
        "Use Evaluation Rubric Design for multi-dimensional judge system, Bias Detection for bias"
        "mitigation in judging.",
    ),
    "task_216": (
        ["evaluation_005", "evaluation_002"],
        "Use Benchmark Protocol Design for benchmark structure, Evaluation Rubric Design for"
        "difficulty levels and annotations.",
    ),
    "task_217": (
        ["evaluation_008", "operational_005"],
        "Use Human Evaluation Design for annotation strategy, Resource Estimation for"
        "crowdsourcing cost planning.",
    ),
    "task_218": (
        ["evaluation_001", "research_002"],
        "Use Metric Selection for hallucination metrics, Comparative Technical Analysis for"
        "detection method comparison.",
    ),
    "task_219": (
        ["evaluation_001", "evaluation_007"],
        "Use Metric Selection for metric sensitivity analysis, Statistical Significance Testing"
        "for ranking stability.",
    ),
    "task_220": (
        ["operational_006", "evaluation_001"],
        "Use Monitoring and Observability for drift detection system, Metric Selection for drift"
        "metrics and thresholds.",
    ),
    # --- Medium (221-240) ---
    "task_221": (
        ["evaluation_005", "evaluation_004"],
        "Use Benchmark Protocol Design for adversarial robustness benchmark, Bias Detection for"
        "robustness vs clean accuracy balance.",
    ),
    "task_222": (
        ["evaluation_004", "evaluation_001"],
        "Use Bias Detection for multi-class fairness metrics, Metric Selection for constraint"
        "selection.",
    ),
    "task_223": (
        ["evaluation_006", "synthesis_009"],
        "Use Error Analysis for systematic categorization, Root Cause Analysis for failure mode"
        "taxonomy.",
    ),
    "task_224": (
        ["evaluation_001", "research_002"],
        "Use Metric Selection for efficiency metrics, Comparative Technical Analysis for"
        "cross-difficulty comparison.",
    ),
    "task_225": (
        ["evaluation_003", "synthesis_008"],
        "Use A/B Test Design for confounding factor analysis, Assumption Surfacing for identifying"
        "hidden confounders.",
    ),
    "task_226": (
        ["evaluation_005", "evaluation_003"],
        "Use Benchmark Protocol Design for few-shot evaluation protocol, A/B Test Design for"
        "sample size variation.",
    ),
    "task_227": (
        ["evaluation_002", "evaluation_008"],
        "Use Evaluation Rubric Design for long-form criteria, Human Evaluation Design for combined"
        "judgment protocol.",
    ),
    "task_228": (
        ["research_010", "evaluation_010"],
        "Use Research Gap Identification for identifying suppressed negative results,"
        "Reproducibility Assessment for fair statistical reporting.",
    ),
    "task_229": (
        ["synthesis_002", "creative_008"],
        "Use Trade-off Analysis for Pareto frontier, Visualization Design for multi-objective"
        "visualization.",
    ),
    "task_230": (
        ["evaluation_005", "evaluation_003"],
        "Use Benchmark Protocol Design for domain shift evaluation, A/B Test Design for"
        "in-vs-out-distribution comparison.",
    ),
    "task_231": (
        ["evaluation_005", "evaluation_002", "research_007"],
        "Use Benchmark Protocol for framework structure, Evaluation Rubric for capability"
        "criteria, Taxonomy for capability classification.",
    ),
    "task_232": (
        ["evaluation_005", "evaluation_004"],
        "Use Benchmark Protocol for unbiased benchmark design, Bias Detection for gaming and data"
        "leakage prevention.",
    ),
    "task_233": (
        ["evaluation_008", "evaluation_007"],
        "Use Human Evaluation Design for agreement study, Statistical Significance Testing for"
        "disagreement analysis.",
    ),
    "task_234": (
        ["evaluation_004", "operational_005"],
        "Use Bias Detection for fairness auditing, Resource Estimation for sampling strategy at"
        "scale.",
    ),
    "task_235": (
        ["evaluation_003", "evaluation_007"],
        "Use A/B Test Design for causal inference, Statistical Significance Testing for"
        "difference-in-differences analysis.",
    ),
    "task_236": (
        ["evaluation_006", "evaluation_004"],
        "Use Error Analysis for shortcut detection, Bias Detection for spurious correlation"
        "identification.",
    ),
    "task_237": (
        ["evaluation_001", "synthesis_002"],
        "Use Metric Selection for aggregation approach, Trade-off Analysis for metric weighting"
        "and conflict resolution.",
    ),
    "task_238": (
        ["evaluation_005", "evaluation_003"],
        "Use Benchmark Protocol for generalization tests, A/B Test Design for extrapolation"
        "evaluation.",
    ),
    "task_239": (
        ["synthesis_002", "operational_005"],
        "Use Trade-off Analysis for evaluation cost-benefit, Resource Estimation for cheaper proxy"
        "identification.",
    ),
    "task_240": (
        ["research_005", "synthesis_007"],
        "Use Empirical Evidence Synthesis for leaderboard meta-analysis, Theme Clustering for"
        "success pattern extraction.",
    ),
    # --- Hard (241-250) ---
    "task_241": (
        ["synthesis_002", "evaluation_001", "research_005"],
        "Use Trade-off Analysis for interpretability-performance frontier, Metric Selection for"
        "measurement approach, Empirical Evidence Synthesis for evidence.",
    ),
    "task_242": (
        ["operational_008", "evaluation_005", "research_009"],
        "Use Testing Strategy for stress test design, Benchmark Protocol for adversarial"
        "benchmarks, Failure Mode Analysis for cascading failures.",
    ),
    "task_243": (
        ["research_005", "evaluation_008", "evaluation_001"],
        "Use Empirical Evidence Synthesis for method comparison, Human Evaluation Design for"
        "agreement study, Metric Selection for correlation metrics.",
    ),
    "task_244": (
        ["evaluation_005", "evaluation_001", "research_008"],
        "Use Benchmark Protocol for temporal evaluation, Metric Selection for forgetting/transfer"
        "metrics, Historical Evolution Analysis for learning trajectory.",
    ),
    "task_245": (
        ["evaluation_010", "operational_007", "evaluation_007"],
        "Use Reproducibility Assessment for framework design, Documentation and Runbook for"
        "release protocol, Statistical Significance for replication analysis.",
    ),
    "task_246": (
        ["evaluation_005", "research_009", "evaluation_004"],
        "Use Benchmark Protocol for backdoor evaluation, Failure Mode Analysis for attack"
        "detection, Bias Detection for targeted misclassification.",
    ),
    "task_247": (
        ["synthesis_002", "research_003", "evaluation_003"],
        "Use Trade-off Analysis for privacy-accuracy trade-off, Algorithm Deep Dive for"
        "differential privacy, A/B Test for federated evaluation.",
    ),
    "task_248": (
        ["evaluation_005", "evaluation_003", "research_005"],
        "Use Benchmark Protocol for sample efficiency evaluation, A/B Test Design for data size"
        "variation, Empirical Evidence Synthesis for cross-model comparison.",
    ),
    "task_249": (
        ["research_006", "evaluation_005", "operational_008"],
        "Use Formal Definition for axiom formulation, Benchmark Protocol for axiom testing,"
        "Testing Strategy for violation detection.",
    ),
    "task_250": (
        ["evaluation_005", "evaluation_009", "evaluation_006"],
        "Use Benchmark Protocol for evaluation framework, Ablation Study Design for component"
        "testing, Error Analysis for failure mode analysis.",
    ),
}


def validate_annotations(tasks, skills, annotations):
    """Validate annotation quality and coverage."""
    errors = []
    warnings = []

    task_ids = {t["task_id"] for t in tasks}
    skill_ids = {s["skill_id"] for s in skills}

    # Check all tasks are annotated
    missing = task_ids - set(annotations.keys())
    if missing:
        errors.append(f"Missing annotations for {len(missing)} tasks: {sorted(missing)[:5]}...")

    # Check no extra annotations
    extra = set(annotations.keys()) - task_ids
    if extra:
        errors.append(f"Annotations for non-existent tasks: {sorted(extra)}")

    # Check all skill references are valid
    all_referenced_skills = set()
    for task_id, (skills_list, approach) in annotations.items():
        for sid in skills_list:
            if sid not in skill_ids:
                errors.append(f"{task_id}: references non-existent skill {sid}")
            all_referenced_skills.add(sid)

        # Check 1-3 skills per task
        if len(skills_list) < 1 or len(skills_list) > 3:
            errors.append(f"{task_id}: has {len(skills_list)} skills (expected 1-3)")

        # Check no duplicate skills
        if len(skills_list) != len(set(skills_list)):
            errors.append(f"{task_id}: has duplicate skills")

        # Check approach is non-empty
        if not approach or len(approach) < 20:
            warnings.append(f"{task_id}: approach is too short ({len(approach)} chars)")

    # Check every skill is used at least once
    unused = skill_ids - all_referenced_skills
    if unused:
        warnings.append(f"Unused skills ({len(unused)}): {sorted(unused)}")

    # Skill usage distribution
    skill_counter = Counter()
    for task_id, (skills_list, approach) in annotations.items():
        for sid in skills_list:
            skill_counter[sid] += 1

    return errors, warnings, skill_counter


def annotate_corpus(tasks, annotations):
    """Add ground_truth_skills and expected_approach to task corpus."""
    annotated = []
    for task in tasks:
        task_copy = dict(task)
        skills_list, approach = annotations[task["task_id"]]
        task_copy["ground_truth_skills"] = skills_list
        task_copy["expected_approach"] = approach
        annotated.append(task_copy)
    return annotated


def main():
    # Load data
    with open(CORPUS_PATH) as f:
        tasks = json.load(f)
    with open(SKILLS_PATH) as f:
        skills = json.load(f)

    print(f"Loaded {len(tasks)} tasks, {len(skills)} skills")
    print(f"Annotations defined for {len(ANNOTATIONS)} tasks")

    # Validate
    errors, warnings, skill_counter = validate_annotations(tasks, skills, ANNOTATIONS)

    if errors:
        print(f"\n❌ ERRORS ({len(errors)}):")
        for e in errors:
            print(f"  - {e}")
        return False

    if warnings:
        print(f"\n⚠️  WARNINGS ({len(warnings)}):")
        for w in warnings:
            print(f"  - {w}")

    # Coverage stats
    print("\n📊 Coverage Stats:")
    print(f"  Tasks annotated: {len(ANNOTATIONS)}/{len(tasks)}")
    print(f"  Skills used: {len(skill_counter)}/{len(skills)}")

    # Skills per task distribution
    skills_per_task = [len(v[0]) for v in ANNOTATIONS.values()]
    print(
        f"  Skills per task: min={min(skills_per_task)}, max={max(skills_per_task)}, "
        f"avg={sum(skills_per_task) / len(skills_per_task):.1f}"
    )

    # Skill usage distribution
    print("\n📈 Skill Usage (top 10):")
    for sid, count in skill_counter.most_common(10):
        skill_title = next(s["title"] for s in skills if s["skill_id"] == sid)
        print(f"  {sid:20s} ({count:3d}x) {skill_title}")

    print("\n📉 Skill Usage (bottom 10):")
    for sid, count in skill_counter.most_common()[-10:]:
        skill_title = next(s["title"] for s in skills if s["skill_id"] == sid)
        print(f"  {sid:20s} ({count:3d}x) {skill_title}")

    # Annotate and write
    annotated = annotate_corpus(tasks, ANNOTATIONS)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(annotated, f, indent=2)

    print(f"\n✅ Wrote annotated corpus to {OUTPUT_PATH}")
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
