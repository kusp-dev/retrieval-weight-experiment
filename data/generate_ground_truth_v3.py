#!/usr/bin/env python3
"""
Generate ground truth annotations mapping 300 tasks to their most relevant skills.
Each task maps to 1-3 skills based on content overlap analysis.
"""

import json
from collections import Counter

# Load both files
from pathlib import Path

_DIR = Path(__file__).parent

with open(_DIR / "tasks" / "corpus_v3.json") as f:
    tasks = json.load(f)

with open(_DIR / "skills" / "library_v3.json") as f:
    skills = json.load(f)

# Build skill index by id for reference
skill_by_id = {s["skill_id"]: s for s in skills}

# ============================================================================
# GROUND TRUTH ANNOTATIONS
# ============================================================================
# Format: task_id -> (primary_skill_id, [relevant_skill_ids])
# Rules:
# - 1-3 skills per task (prefer 2-3 for medium/hard, 1-2 for easy)
# - primary_skill_id is the single most relevant
# - Domain-specific skills over generic meta-skills
# - Every domain-specific skill used at least once
# ============================================================================

annotations = {
    # ========================================================================
    # ML FUNDAMENTALS (tasks 001-050, 251-260)
    # ========================================================================
    # task_001: Explain Backpropagation Algorithm (easy)
    "task_001": ("mlf_002", ["mlf_002"]),
    # task_002: Compare SGD vs Adam Optimizer (hard)
    "task_002": ("mlf_008", ["mlf_008", "mlf_010", "research_002"]),
    # task_003: Explain Attention Mechanism (easy)
    "task_003": ("mlf_005", ["mlf_005"]),
    # task_004: Define CNN Architecture (easy)
    "task_004": ("mlf_015", ["mlf_015", "mlf_034"]),
    # task_005: Explain Regularization Techniques (easy)
    "task_005": ("mlf_019", ["mlf_019"]),
    # task_006: Understanding Loss Functions (hard)
    "task_006": ("mlf_013", ["mlf_013", "mlf_031", "research_002"]),
    # task_007: RNN vs LSTM Comparison (hard)
    "task_007": ("mlf_001", ["mlf_001", "mlf_004", "mlf_003"]),
    # task_008: Batch Normalization Overview (easy)
    "task_008": ("mlf_012", ["mlf_012"]),
    # task_009: What is Transfer Learning (easy)
    "task_009": ("mlf_030", ["mlf_030"]),
    # task_010: Few-Shot Learning Basics (easy)
    "task_010": ("mlf_030", ["mlf_030", "creative_004"]),
    # task_011: Dropout Regularization Mechanism (easy)
    "task_011": ("mlf_011", ["mlf_011"]),
    # task_012: Neural Architecture Search Intro (hard)
    "task_012": ("mlf_015", ["mlf_015", "research_012", "synthesis_010"]),
    # task_013: Gradient Descent Variants (hard)
    "task_013": ("mlf_010", ["mlf_010", "mlf_008", "research_002"]),
    # task_014: Activation Functions in Deep Learning (hard)
    "task_014": ("mlf_017", ["mlf_017", "research_002"]),
    # task_015: Overfitting and Underfitting (hard)
    "task_015": ("mlf_029", ["mlf_029", "mlf_019", "mlf_011"]),
    # task_016: Normalization Techniques Overview (hard)
    "task_016": ("mlf_032", ["mlf_032", "mlf_012", "research_002"]),
    # task_017: Initialization Strategies (hard)
    "task_017": ("mlf_018", ["mlf_018", "mlf_003"]),
    # task_018: Classification vs Regression (easy)
    "task_018": ("mlf_013", ["mlf_013", "research_006"]),
    # task_019: Cross-Validation for Model Selection (hard)
    "task_019": ("evm_024", ["evm_024", "mlf_029", "evaluation_003"]),
    # task_020: Embedding Layers in Neural Networks (easy)
    "task_020": ("mlf_014", ["mlf_014", "mlf_023"]),
    # task_021: Advanced Backpropagation Analysis (medium)
    "task_021": ("mlf_002", ["mlf_002", "mlf_003", "mlf_016"]),
    # task_022: Design Optimizer for Sparse Data (medium)
    "task_022": ("mlf_008", ["mlf_008", "mlf_010"]),
    # task_023: Multi-Head Attention Deep Dive (medium)
    "task_023": ("mlf_005", ["mlf_005", "mlf_033", "evaluation_009"]),
    # task_024: Analyzing CNN Filter Learning (medium)
    "task_024": ("mlf_015", ["mlf_015", "mlf_026", "synthesis_009"]),
    # task_025: Loss Function Landscape Analysis (medium)
    "task_025": ("mlf_031", ["mlf_031", "mlf_013", "research_006"]),
    # task_026: Dropout Rate Optimization (medium)
    "task_026": ("mlf_011", ["mlf_011", "mlf_029"]),
    # task_027: Meta-Learning vs Transfer Learning (medium)
    "task_027": ("mlf_030", ["mlf_030", "research_002"]),
    # task_028: NAS Search Space Design (medium)
    "task_028": ("synthesis_010", ["synthesis_010", "mlf_015", "creative_009"]),
    # task_029: Gradient Checkpointing Trade-offs (medium)
    "task_029": ("mlf_002", ["mlf_002", "operational_005", "synthesis_002"]),
    # task_030: Mixed Precision Training (medium)
    "task_030": ("mlf_027", ["mlf_027", "mlf_009"]),
    # task_031: Curriculum Learning Strategy (medium)
    "task_031": ("mlf_024", ["mlf_024", "creative_009"]),
    # task_032: Knowledge Distillation Pipeline (medium)
    "task_032": ("mlf_025", ["mlf_025", "operational_003"]),
    # task_033: Adversarial Training Analysis (medium)
    "task_033": ("mlf_029", ["mlf_029", "evm_021", "evaluation_003"]),
    # task_034: Explainability via Attention Visualization (medium)
    "task_034": ("mlf_033", ["mlf_033", "mlf_005"]),
    # task_035: Data Augmentation Strategy Design (medium)
    "task_035": ("mlf_024", ["mlf_024", "mlf_015", "creative_009"]),
    # task_036: Hyperparameter Interaction Effects (medium)
    "task_036": ("mlf_009", ["mlf_009", "mlf_008", "evaluation_003"]),
    # task_037: Continual Learning Without Catastrophic Forgetting (medium)
    "task_037": ("mlf_030", ["mlf_030", "mlf_019", "research_002"]),
    # task_038: Model Pruning vs Quantization (medium)
    "task_038": ("mlf_026", ["mlf_026", "mlf_027", "research_002"]),
    # task_039: Ensemble Methods Beyond Voting (medium)
    "task_039": ("mlf_028", ["mlf_028", "synthesis_002"]),
    # task_040: Class Imbalance Mitigation Comparative Analysis (medium)
    "task_040": ("mlf_031", ["mlf_031", "mlf_013", "research_002"]),
    # task_041: System Design: Image Classification Pipeline (hard)
    "task_041": ("operational_003", ["operational_003", "mlf_015", "operational_006"]),
    # task_042: Scaling Laws and Compute Optimal Architecture Search (hard)
    "task_042": ("research_012", ["research_012", "mlf_007", "research_005"]),
    # task_043: Generalization Bounds and Empirical Risk Minimization (hard)
    "task_043": ("research_006", ["research_006", "mlf_029", "mlf_019"]),
    # task_044: Designing Federated Learning System (hard)
    "task_044": ("mlf_030", ["mlf_030", "mag_006", "research_004"]),
    # task_045: Causal Inference in Deep Learning (hard)
    "task_045": ("research_005", ["research_005", "evaluation_003", "mlf_029"]),
    # task_046: Interpretable ML System Design (hard)
    "task_046": ("synthesis_002", ["synthesis_002", "mlf_029", "creative_009"]),
    # task_047: Benchmark Gaming and Robust Evaluation (hard)
    "task_047": ("evm_034", ["evm_034", "research_011", "evm_009"]),
    # task_048: Multi-Task Learning Architecture Design (hard)
    "task_048": ("mlf_030", ["mlf_030", "mlf_016", "synthesis_010"]),
    # task_049: Automating Hyperparameter Tuning (hard)
    "task_049": ("mlf_008", ["mlf_008", "mlf_009", "research_002"]),
    # task_050: Foundation Model Adaptation Framework (hard)
    "task_050": ("mlf_030", ["mlf_030", "mlf_025", "operational_003"]),
    # ========================================================================
    # AGENT ARCHITECTURES (tasks 051-100, 261-270)
    # ========================================================================
    # task_051: Explain ReAct Framework (easy)
    "task_051": ("aga_001", ["aga_001"]),
    # task_052: Chain-of-Thought Prompting (easy)
    "task_052": ("aga_005", ["aga_005"]),
    # task_053: Tool Use in Language Models (easy)
    "task_053": ("aga_002", ["aga_002"]),
    # task_054: AutoGPT Agent Components (hard)
    "task_054": ("aga_011", ["aga_011", "aga_022", "aga_020"]),
    # task_055: Planning Algorithms for Agents (easy)
    "task_055": ("aga_020", ["aga_020", "aga_004"]),
    # task_056: Memory Systems in Agents (easy)
    "task_056": ("aga_022", ["aga_022", "aga_003"]),
    # task_057: Reflection and Self-Improvement (easy)
    "task_057": ("aga_008", ["aga_008"]),
    # task_058: Function Calling and API Integration (easy)
    "task_058": ("aga_021", ["aga_021", "aga_002"]),
    # task_059: Prompt Chaining Patterns (easy)
    "task_059": ("aga_031", ["aga_031"]),
    # task_060: Error Handling in Agents (easy)
    "task_060": ("aga_018", ["aga_018"]),
    # task_061: Agent Evaluation Metrics (hard)
    "task_061": ("evm_028", ["evm_028", "aga_029", "evm_027"]),
    # task_062: Tree-of-Thought vs Chain-of-Thought (hard)
    "task_062": ("aga_004", ["aga_004", "aga_005", "research_002"]),
    # task_063: Grounding LLM Outputs (easy)
    "task_063": ("aga_014", ["aga_014", "aga_002"]),
    # task_064: Agent State Management (hard)
    "task_064": ("aga_010", ["aga_010", "aga_003", "aga_032"]),
    # task_065: Uncertainty Quantification in Agents (hard)
    "task_065": ("aga_035", ["aga_035", "aga_025", "evm_010"]),
    # task_066: Advanced ReAct Analysis (hard)
    "task_066": ("aga_001", ["aga_001", "aga_007", "research_009"]),
    # task_067: Combining CoT with Tool Use (hard)
    "task_067": ("aga_001", ["aga_001", "aga_005", "aga_002"]),
    # task_068: Hierarchical Task Decomposition (hard)
    "task_068": ("aga_011", ["aga_011", "aga_030", "aga_020"]),
    # task_069: Memory Optimization for Long Tasks (hard)
    "task_069": ("aga_010", ["aga_010", "aga_032", "aga_022"]),
    # task_070: Reflection Mechanism Design (hard)
    "task_070": ("aga_008", ["aga_008", "aga_007", "mag_021"]),
    # task_071: Function Specification and Validation (medium)
    "task_071": ("aga_002", ["aga_002", "aga_027", "aga_021"]),
    # task_072: Prompt Chaining with Branching (medium)
    "task_072": ("aga_031", ["aga_031", "aga_013"]),
    # task_073: Robust Error Recovery Strategy (medium)
    "task_073": ("aga_018", ["aga_018", "aga_027", "operational_001"]),
    # task_074: Multi-Agent Coordination Patterns (medium)
    "task_074": ("mag_005", ["mag_005", "mag_004", "aga_031"]),
    # task_075: Cost Control in Agentic Systems (medium)
    "task_075": ("aga_029", ["aga_029", "aga_016", "evm_027"]),
    # task_076: Safety and Alignment in Agents (medium)
    "task_076": ("aga_012", ["aga_012", "aga_025"]),
    # task_077: Agent Behavior Logging and Debugging (medium)
    "task_077": ("operational_006", ["operational_006", "aga_018", "mag_032"]),
    # task_078: Scaling Agents to Many Tasks (medium)
    "task_078": ("mag_024", ["mag_024", "mag_013", "operational_003"]),
    # task_079: Dynamic Tool Selection (medium)
    "task_079": ("aga_034", ["aga_034", "aga_002"]),
    # task_080: Learning from Agent Failures (medium)
    "task_080": ("aga_008", ["aga_008", "research_009", "synthesis_009"]),
    # task_081: Complex System Orchestration with Agents (medium)
    "task_081": ("mag_008", ["mag_008", "aga_011", "operational_003"]),
    # task_082: Agent Behavior Prediction and Control (medium)
    "task_082": ("aga_033", ["aga_033", "aga_007", "evm_028"]),
    # task_083: Emergent Behaviors in Multi-Agent Systems (medium)
    "task_083": ("mag_019", ["mag_019", "mag_032"]),
    # task_084: Agent Self-Improvement Without Retraining (medium)
    "task_084": ("aga_008", ["aga_008", "aga_015", "aga_006"]),
    # task_085: Adversarial Robustness of Agents (medium)
    "task_085": ("aga_012", ["aga_012", "aga_033", "evm_021"]),
    # task_086: Agent Curriculum Learning (medium)
    "task_086": ("aga_011", ["aga_011", "aga_020", "mlf_024"]),
    # task_087: Meta-Learning for Agent Prompting (medium)
    "task_087": ("aga_006", ["aga_006", "aga_019", "aga_024"]),
    # task_088: Collaborative Agent Frameworks (medium)
    "task_088": ("mag_005", ["mag_005", "mag_007", "mag_004"]),
    # task_089: Verifiable Reasoning in Agents (medium)
    "task_089": ("aga_033", ["aga_033", "aga_005", "aga_017"]),
    # task_090: Adaptive Agent Architectures (medium)
    "task_090": ("aga_020", ["aga_020", "aga_034", "synthesis_010"]),
    # task_091: Inverse Reinforcement Learning for Agent Goals (hard)
    "task_091": ("evm_028", ["evm_028", "aga_007", "research_006"]),
    # task_092: Real-Time Agent Monitoring and Adaptation (hard)
    "task_092": ("operational_006", ["operational_006", "aga_029", "mag_035"]),
    # task_093: Agent Knowledge Graph Construction (hard)
    "task_093": ("aga_022", ["aga_022", "ret_024", "creative_010"]),
    # task_094: Scalable Agent Execution Framework (hard)
    "task_094": ("mag_024", ["mag_024", "mag_029", "operational_005"]),
    # task_095: Human-in-the-Loop Agent Improvement (hard)
    "task_095": ("aga_025", ["aga_025", "mag_014", "aga_008"]),
    # task_096: Agent Capability Benchmarking (hard)
    "task_096": ("evaluation_005", ["evaluation_005", "evm_028", "evm_032"]),
    # task_097: Compositional Reasoning in Agents (hard)
    "task_097": ("aga_023", ["aga_023", "aga_011", "aga_004"]),
    # task_098: Counterfactual Reasoning for Agent Improvement (hard)
    "task_098": ("mag_032", ["mag_032", "aga_008", "evm_028"]),
    # task_099: Privacy-Preserving Agent Collaboration (hard)
    "task_099": ("mag_033", ["mag_033", "mag_006", "aga_012"]),
    # task_100: Agent Fairness and Bias Detection (hard)
    "task_100": ("evm_020", ["evm_020", "aga_012", "evm_033"]),
    # ========================================================================
    # RETRIEVAL SYSTEMS (tasks 101-150, 271-280)
    # ========================================================================
    # task_101: RAG Pipeline Fundamentals (easy)
    "task_101": ("ret_012", ["ret_012"]),
    # task_102: Dense vs Sparse Retrieval (easy)
    "task_102": ("ret_030", ["ret_030", "ret_001"]),
    # task_103: Embedding Models for Retrieval (easy)
    "task_103": ("ret_003", ["ret_003", "ret_008", "evm_003"]),
    # task_104: Query Expansion Techniques (easy)
    "task_104": ("ret_022", ["ret_022", "ret_007"]),
    # task_105: Vector Databases Basics (easy)
    "task_105": ("ret_005", ["ret_005", "ret_026"]),
    # task_106: Chunking Strategies for Documents (easy)
    "task_106": ("ret_002", ["ret_002", "ret_009"]),
    # task_107: Re-ranking and Cross-Encoders (easy)
    "task_107": ("ret_006", ["ret_006", "ret_021"]),
    # task_108: Knowledge Graphs for Retrieval (hard)
    "task_108": ("ret_024", ["ret_024", "ret_030", "research_002"]),
    # task_109: BM25 Algorithm Details (easy)
    "task_109": ("ret_001", ["ret_001"]),
    # task_110: Evaluation Metrics for Retrieval (easy)
    "task_110": ("ret_008", ["ret_008", "evm_001", "evm_002"]),
    # task_111: Hybrid Search Best Practices (hard)
    "task_111": ("ret_004", ["ret_004", "ret_035", "ret_010"]),
    # task_112: Cold Start Problem in Retrieval (hard)
    "task_112": ("ret_003", ["ret_003", "ret_019", "research_009"]),
    # task_113: Personalization in Retrieval (hard)
    "task_113": ("ret_015", ["ret_015", "ret_032", "research_002"]),
    # task_114: Retrieval Feedback Loops (hard)
    "task_114": ("ret_032", ["ret_032", "evm_022", "evm_023"]),
    # task_115: Semantic Search Fundamentals (easy)
    "task_115": ("ret_030", ["ret_030", "ret_003"]),
    # task_116: Advanced RAG Architecture (hard)
    "task_116": ("ret_028", ["ret_028", "ret_016", "aga_028"]),
    # task_117: Embedding Space Analysis (hard)
    "task_117": ("ret_019", ["ret_019", "ret_003", "synthesis_009"]),
    # task_118: Query Expansion Strategy Design (hard)
    "task_118": ("ret_022", ["ret_022", "ret_007", "ret_014"]),
    # task_119: Vector Database Index Optimization (hard)
    "task_119": ("ret_005", ["ret_005", "ret_026", "ret_013"]),
    # task_120: Semantic Chunking Strategy (hard)
    "task_120": ("ret_002", ["ret_002", "ret_009", "ret_031"]),
    # task_121: Learned Ranking Pipeline (medium)
    "task_121": ("ret_006", ["ret_006", "ret_011", "ret_008"]),
    # task_122: Handling Retrieval Failures (medium)
    "task_122": ("ret_029", ["ret_029", "ret_016", "aga_018"]),
    # task_123: Knowledge Graph Embedding for Retrieval (medium)
    "task_123": ("ret_024", ["ret_024", "mlf_014", "ret_017"]),
    # task_124: Multimodal Retrieval System (medium)
    "task_124": ("ret_023", ["ret_023", "ret_003"]),
    # task_125: Temporal Awareness in Retrieval (medium)
    "task_125": ("ret_015", ["ret_015", "ret_034"]),
    # task_126: Domain Adaptation for Embeddings (medium)
    "task_126": ("ret_019", ["ret_019", "ret_003", "mlf_035"]),
    # task_127: Retrieval at Scale with Constraints (medium)
    "task_127": ("ret_026", ["ret_026", "ret_013", "ret_020"]),
    # task_128: Diversity in Retrieval Results (medium)
    "task_128": ("ret_025", ["ret_025", "ret_006"]),
    # task_129: Retrieval for Long-Form Generation (medium)
    "task_129": ("ret_025", ["ret_025", "ret_014", "ret_012"]),
    # task_130: Cross-Lingual Retrieval (medium)
    "task_130": ("ret_003", ["ret_003", "ret_023", "mlf_035"]),
    # task_131: End-to-End RAG System Optimization (medium)
    "task_131": ("ret_012", ["ret_012", "ret_020", "operational_003"]),
    # task_132: Adaptive Retrieval based on Query Complexity (medium)
    "task_132": ("ret_028", ["ret_028", "ret_014", "ret_016"]),
    # task_133: Retrieval with Explanations (medium)
    "task_133": ("ret_033", ["ret_033", "ret_006", "ret_008"]),
    # task_134: Handling Contradictory Retrieved Information (medium)
    "task_134": ("ret_016", ["ret_016", "ret_025", "synthesis_003"]),
    # task_135: Continual Learning for Retrieval (medium)
    "task_135": ("ret_019", ["ret_019", "ret_032", "evm_030"]),
    # task_136: Retrieval for Code Search (medium)
    "task_136": ("ret_030", ["ret_030", "ret_003", "ret_018"]),
    # task_137: Privacy-Preserving Dense Retrieval (medium)
    "task_137": ("ret_013", ["ret_013", "ret_003"]),
    # task_138: Retrieval with Causal Reasoning (medium)
    "task_138": ("ret_014", ["ret_014", "ret_024"]),
    # task_139: Dynamic Weighting of Retrieval Dimensions (medium)
    "task_139": ("ret_035", ["ret_035", "ret_004", "evm_026"]),
    # task_140: Evaluating Retrieval System Robustness (medium)
    "task_140": ("ret_029", ["ret_029", "evm_021", "ret_008"]),
    # task_141: Meta-Learning for Retrieval Hyperparameters (hard)
    "task_141": ("ret_035", ["ret_035", "ret_004", "evm_026"]),
    # task_142: Retrieval-Augmented Fine-Tuning (hard)
    "task_142": ("ret_012", ["ret_012", "ret_028", "mlf_030"]),
    # task_143: Graph Neural Networks for Retrieval (hard)
    "task_143": ("ret_024", ["ret_024", "mlf_014", "research_004"]),
    # task_144: Retrieval System Fairness and Bias (hard)
    "task_144": ("evm_020", ["evm_020", "ret_029", "evm_033"]),
    # task_145: Zero-Shot Retrieval Generalization (hard)
    "task_145": ("ret_003", ["ret_003", "mlf_035", "evm_030"]),
    # task_146: Instruction-Tuned Retrieval Models (hard)
    "task_146": ("ret_032", ["ret_032", "ret_019", "ret_003"]),
    # task_147: Retrieval for Fact Verification (hard)
    "task_147": ("ret_016", ["ret_016", "ret_014", "ret_012"]),
    # task_148: Efficient Dense Retrieval for Mobile (hard)
    "task_148": ("ret_013", ["ret_013", "mlf_027", "mlf_026"]),
    # task_149: Retrieval Interaction Effects (hard)
    "task_149": ("ret_022", ["ret_022", "ret_014", "ret_035"]),
    # task_150: Benchmark Development for Advanced Retrieval (hard)
    "task_150": ("evaluation_005", ["evaluation_005", "ret_008", "evm_032"]),
    # ========================================================================
    # MULTI-AGENT SYSTEMS (tasks 151-200, 281-290)
    # ========================================================================
    # task_151: Agent Roles and Specialization (easy)
    "task_151": ("mag_007", ["mag_007"]),
    # task_152: Agent Communication Protocols (easy)
    "task_152": ("mag_020", ["mag_020", "mag_005"]),
    # task_153: Task Allocation in Multi-Agent Systems (easy)
    "task_153": ("mag_022", ["mag_022", "mag_004"]),
    # task_154: Coordination Mechanisms (easy)
    "task_154": ("mag_001", ["mag_001", "mag_022"]),
    # task_155: Consensus Algorithms (easy)
    "task_155": ("mag_006", ["mag_006"]),
    # task_156: Conflict Resolution in Teams (easy)
    "task_156": ("mag_011", ["mag_011", "mag_006"]),
    # task_157: Agent Scalability (easy)
    "task_157": ("mag_024", ["mag_024", "mag_020", "research_012"]),
    # task_158: Emergent Behavior (easy)
    "task_158": ("mag_019", ["mag_019"]),
    # task_159: Agent Learning from Interactions (hard)
    "task_159": ("mag_015", ["mag_015", "mag_028", "mag_030"]),
    # task_160: Debate Framework for Multi-Agent Reasoning (easy)
    "task_160": ("mag_003", ["mag_003"]),
    # task_161: Hierarchical vs Flat Organization (easy)
    "task_161": ("mag_016", ["mag_016"]),
    # task_162: Shared Knowledge Bases (hard)
    "task_162": ("mag_012", ["mag_012", "mag_001", "mag_026"]),
    # task_163: Agent Accountability (hard)
    "task_163": ("mag_033", ["mag_033", "mag_032", "operational_006"]),
    # task_164: Advanced Coordination Protocol Design (hard)
    "task_164": ("mag_008", ["mag_008", "mag_031", "mag_005"]),
    # task_165: Task Allocation Optimization (hard)
    "task_165": ("mag_022", ["mag_022", "mag_013", "operational_005"]),
    # task_166: Heterogeneous Agent Teams (hard)
    "task_166": ("mag_007", ["mag_007", "mag_004", "mag_034"]),
    # task_167: Communication Bandwidth Constraints (hard)
    "task_167": ("mag_020", ["mag_020", "mag_023", "research_012"]),
    # task_168: Handling Agent Failures (hard)
    "task_168": ("mag_035", ["mag_035", "mag_009", "mag_031"]),
    # task_169: Multi-Agent Planning Algorithms (hard)
    "task_169": ("mag_018", ["mag_018", "mag_004", "research_002"]),
    # task_170: Debate and Voting for Accuracy (hard)
    "task_170": ("mag_003", ["mag_003", "mag_006", "mag_011"]),
    # task_171: Incentive Design for Cooperation (medium)
    "task_171": ("mag_015", ["mag_015", "mag_027"]),
    # task_172: Knowledge Synthesis from Multiple Agents (medium)
    "task_172": ("mag_034", ["mag_034", "synthesis_003", "mag_012"]),
    # task_173: Multi-Agent Goal Alignment (medium)
    "task_173": ("mag_011", ["mag_011", "mag_006", "mag_027"]),
    # task_174: Emergent Specialization (medium)
    "task_174": ("mag_019", ["mag_019", "mag_007", "mag_010"]),
    # task_175: Multi-Agent Reinforcement Learning (medium)
    "task_175": ("mag_027", ["mag_027", "mag_006", "mag_018"]),
    # task_176: Distributed Data Processing with Agents (medium)
    "task_176": ("mag_029", ["mag_029", "mag_013", "mag_024"]),
    # task_177: Trust and Reputation in Agent Teams (medium)
    "task_177": ("mag_015", ["mag_015", "mag_025"]),
    # task_178: Optimization for Multi-Agent Systems (medium)
    "task_178": ("mag_006", ["mag_006", "mag_018", "mag_028"]),
    # task_179: Scalable Multi-Agent Coordination (medium)
    "task_179": ("mag_028", ["mag_028", "mag_024", "mag_020"]),
    # task_180: Testing and Debugging Multi-Agent Systems (medium)
    "task_180": ("mag_032", ["mag_032", "operational_008", "mag_019"]),
    # task_181: Large-Scale Coordination Protocol Design (medium)
    "task_181": ("mag_008", ["mag_008", "mag_031", "mag_024"]),
    # task_182: Emergent Strategy Formation (medium)
    "task_182": ("mag_019", ["mag_019", "mag_027", "creative_009"]),
    # task_183: Agent Heterogeneity and Diversity (medium)
    "task_183": ("mag_034", ["mag_034", "mag_007", "mag_025"]),
    # task_184: Byzantine-Tolerant Multi-Agent Systems (medium)
    "task_184": ("mag_006", ["mag_006", "mag_027", "mag_015"]),
    # task_185: Optimal Agent Debate Format (medium)
    "task_185": ("mag_003", ["mag_003", "mag_011", "research_006"]),
    # task_186: Multi-Agent System Dynamics Modeling (medium)
    "task_186": ("mag_019", ["mag_019", "research_006", "mag_025"]),
    # task_187: Information Aggregation in Prediction Markets (medium)
    "task_187": ("mag_006", ["mag_006", "mag_003", "synthesis_002"]),
    # task_188: Multi-Agent Inverse Reinforcement Learning (medium)
    "task_188": ("mag_025", ["mag_025", "mag_027", "evm_028"]),
    # task_189: Learning Communication Protocols (medium)
    "task_189": ("mag_020", ["mag_020", "mag_005", "mag_028"]),
    # task_190: Adaptive Organization Structure (medium)
    "task_190": ("mag_016", ["mag_016", "mag_010", "mag_019"]),
    # task_191: Fairness in Multi-Agent Outcomes (hard)
    "task_191": ("evm_020", ["evm_020", "mag_011", "mag_015"]),
    # task_192: Multi-Agent System Verification (hard)
    "task_192": ("mag_032", ["mag_032", "mag_031", "operational_008"]),
    # task_193: Scalable Knowledge Sharing (hard)
    "task_193": ("mag_012", ["mag_012", "mag_028", "mag_024"]),
    # task_194: Multi-Agent Planning with Uncertainty (hard)
    "task_194": ("mag_018", ["mag_018", "mag_009", "creative_009"]),
    # task_195: Agent Meta-Learning for Collaboration (hard)
    "task_195": ("mag_030", ["mag_030", "mag_007", "mlf_030"]),
    # task_196: Conflict-Free Multi-Agent Consensus (hard)
    "task_196": ("mag_006", ["mag_006", "mag_002", "research_006"]),
    # task_197: Measuring Collective Intelligence (hard)
    "task_197": ("mag_025", ["mag_025", "mag_034", "evm_027"]),
    # task_198: Incentive Compatibility in Multi-Agent Design (hard)
    "task_198": ("mag_027", ["mag_027", "mag_011", "research_006"]),
    # task_199: Temporal Coordination in Multi-Agent Systems (hard)
    "task_199": ("mag_031", ["mag_031", "mag_026", "mag_008"]),
    # task_200: Long-Horizon Multi-Agent Task Execution (hard)
    "task_200": ("mag_008", ["mag_008", "mag_004", "mag_009"]),
    # ========================================================================
    # EVALUATION METHODS (tasks 201-250, 291-300)
    # ========================================================================
    # task_201: LLM-as-Judge for Text Quality (easy)
    "task_201": ("evm_004", ["evm_004"]),
    # task_202: Standard Benchmarks Overview (easy)
    "task_202": ("evm_009", ["evm_009", "research_011"]),
    # task_203: Human Evaluation Protocols (easy)
    "task_203": ("evaluation_008", ["evaluation_008"]),
    # task_204: Inter-Annotator Agreement (easy)
    "task_204": ("evm_007", ["evm_007", "evm_008"]),
    # task_205: Automatic Metrics for NLP (easy)
    "task_205": ("evm_015", ["evm_015", "evm_017"]),
    # task_206: Bias Detection in ML Systems (hard)
    "task_206": ("evm_020", ["evm_020", "evm_033", "evm_029"]),
    # task_207: Robustness Testing (easy)
    "task_207": ("evm_021", ["evm_021", "evm_030"]),
    # task_208: Calibration and Uncertainty (easy)
    "task_208": ("evm_010", ["evm_010", "evm_011"]),
    # task_209: A/B Testing for ML Models (hard)
    "task_209": ("evm_012", ["evm_012", "evaluation_003", "evm_031"]),
    # task_210: Fairness Metrics and Trade-offs (hard)
    "task_210": ("evm_020", ["evm_020", "evm_033", "synthesis_002"]),
    # task_211: Interpretability and Explainability (easy)
    "task_211": ("evm_025", ["evm_025"]),
    # task_212: Reward Hacking Prevention (easy)
    "task_212": ("evm_019", ["evm_019"]),
    # task_213: Cross-Validation Strategies (hard)
    "task_213": ("evm_024", ["evm_024", "evm_032", "evaluation_003"]),
    # task_214: Statistical Significance Testing (easy)
    "task_214": ("evm_031", ["evm_031", "evm_014"]),
    # task_215: Advanced LLM Judging Design (hard)
    "task_215": ("evm_004", ["evm_004", "evm_005", "evm_006"]),
    # task_216: Custom Benchmark Design (hard)
    "task_216": ("evaluation_005", ["evaluation_005", "evm_032", "evm_034"]),
    # task_217: Scale Human Evaluation (hard)
    "task_217": ("evaluation_008", ["evaluation_008", "evm_007", "evm_008"]),
    # task_218: Measuring Hallucination in LLMs (hard)
    "task_218": ("evm_016", ["evm_016", "evm_006", "evm_025"]),
    # task_219: Sensitivity Analysis for Metrics (hard)
    "task_219": ("evm_015", ["evm_015", "evm_017", "evm_027"]),
    # task_220: Temporal Drift Detection (hard)
    "task_220": ("evm_030", ["evm_030", "operational_006", "evm_023"]),
    # task_221: Adversarial Robustness Benchmarking (medium)
    "task_221": ("evm_021", ["evm_021", "evaluation_005", "evm_009"]),
    # task_222: Fairness in Multi-Class Classification (medium)
    "task_222": ("evm_020", ["evm_020", "evm_033"]),
    # task_223: Error Analysis Methodology (medium)
    "task_223": ("synthesis_009", ["synthesis_009", "research_009", "evm_022"]),
    # task_224: Measuring Agent Efficiency (medium)
    "task_224": ("evm_028", ["evm_028", "evm_027", "aga_029"]),
    # task_225: Confounding Variable Analysis (medium)
    "task_225": ("evm_031", ["evm_031", "evaluation_009", "evm_013"]),
    # task_226: Few-Shot Learning Evaluation (medium)
    "task_226": ("evm_032", ["evm_032", "evm_014", "mlf_030"]),
    # task_227: Long-Form Generation Evaluation (medium)
    "task_227": ("evm_018", ["evm_018", "evm_016", "evaluation_002"]),
    # task_228: Negative Results and Publication Bias (medium)
    "task_228": ("evm_035", ["evm_035", "evaluation_010", "evm_031"]),
    # task_229: Multi-Objective Optimization Evaluation (medium)
    "task_229": ("evm_027", ["evm_027", "synthesis_002"]),
    # task_230: Evaluation in Domain Shift (medium)
    "task_230": ("evm_030", ["evm_030", "mlf_035", "evm_022"]),
    # task_231: Comprehensive LLM Evaluation Framework (medium)
    "task_231": ("evm_025", ["evm_025", "evaluation_005", "evm_004"]),
    # task_232: Designing Unbiased Evaluation Benchmarks (medium)
    "task_232": ("evm_034", ["evm_034", "evm_009", "evaluation_005"]),
    # task_233: Human-LLM Evaluation Agreement (medium)
    "task_233": ("evm_005", ["evm_005", "evm_007", "evm_004"]),
    # task_234: Scalable Fairness Auditing (medium)
    "task_234": ("evm_020", ["evm_020", "evm_033", "evm_032"]),
    # task_235: Causal Evaluation of Model Changes (medium)
    "task_235": ("evm_012", ["evm_012", "evaluation_003", "evm_035"]),
    # task_236: Detecting Shortcuts and Spurious Correlations (medium)
    "task_236": ("evm_024", ["evm_024", "evm_021", "synthesis_009"]),
    # task_237: Metric Aggregation and Ranking (medium)
    "task_237": ("evm_027", ["evm_027", "synthesis_002", "evm_035"]),
    # task_238: Generalization Assessment (medium)
    "task_238": ("evm_030", ["evm_030", "evm_032", "mlf_029"]),
    # task_239: Cost-Benefit Analysis of Evaluation (medium)
    "task_239": ("evm_027", ["evm_027", "evm_023", "operational_005"]),
    # task_240: Competitive Benchmarking Analysis (medium)
    "task_240": ("evm_009", ["evm_009", "research_005", "evm_035"]),
    # task_241: Interpretability-Performance Trade-off (hard)
    "task_241": ("evm_025", ["evm_025", "synthesis_002", "evaluation_009"]),
    # task_242: Stress Testing ML Systems (hard)
    "task_242": ("evm_021", ["evm_021", "operational_008", "evm_030"]),
    # task_243: Meta-Analysis of Evaluation Methods (hard)
    "task_243": ("evm_005", ["evm_005", "evm_004", "research_005"]),
    # task_244: Temporal Evaluation for Continual Learning (hard)
    "task_244": ("evm_030", ["evm_030", "mlf_030", "evm_028"]),
    # task_245: Reproducibility and Replicability (hard)
    "task_245": ("evaluation_010", ["evaluation_010", "evm_014", "evm_035"]),
    # task_246: Backdoor Detection and Evaluation (hard)
    "task_246": ("evm_021", ["evm_021", "evm_009", "evaluation_005"]),
    # task_247: Privacy Preservation in Evaluation (hard)
    "task_247": ("evm_012", ["evm_012", "evm_032"]),
    # task_248: Sample Efficiency Evaluation (hard)
    "task_248": ("evm_032", ["evm_032", "evm_014", "evm_012"]),
    # task_249: Axiom-Based Evaluation (hard)
    "task_249": ("evm_010", ["evm_010", "research_006", "evm_025"]),
    # task_250: Comprehensive Research System Evaluation (hard)
    "task_250": ("evaluation_005", ["evaluation_005", "evaluation_009", "evm_032"]),
    # ========================================================================
    # GENERATED_V3 ML FUNDAMENTALS (tasks 251-260)
    # ========================================================================
    # task_251: Diagnosing Training Instability in Deep Residual Networks (hard)
    "task_251": ("mlf_016", ["mlf_016", "mlf_012", "mlf_003"]),
    # task_252: Debugging Mode Collapse in Conditional GANs (hard)
    "task_252": ("mlf_021", ["mlf_021", "mlf_022", "synthesis_009"]),
    # task_253: Optimal Tokenization Strategy for Multilingual Scientific Models (hard)
    "task_253": ("mlf_023", ["mlf_023", "mlf_006", "synthesis_010"]),
    # task_254: Memory-Efficient Training Under Heterogeneous Hardware Constraints (hard)
    "task_254": ("mlf_027", ["mlf_027", "mlf_009", "operational_005"]),
    # task_255: Catastrophic Interference in Multi-Domain Continual Pre-Training (hard)
    "task_255": ("mlf_030", ["mlf_030", "mlf_025", "mlf_019"]),
    # task_256: Theoretical Analysis of LoRA Rank Selection (hard)
    "task_256": ("mlf_030", ["mlf_030", "mlf_018", "research_006"]),
    # task_257: Diagnosing Numerical Instability in Mixture-of-Experts Models (hard)
    "task_257": ("mlf_032", ["mlf_032", "mlf_003", "mlf_027"]),
    # task_258: Designing Self-Supervised Pretext Task for Tabular Data (hard)
    "task_258": ("mlf_014", ["mlf_014", "mlf_020", "research_010"]),
    # task_259: Optimizing Inference Throughput Under Strict Latency and Accuracy SLAs (hard)
    "task_259": ("mlf_027", ["mlf_027", "mlf_026", "operational_005"]),
    # task_260: Failure Analysis of Contrastive Learning with Distribution Shift (hard)
    "task_260": ("mlf_014", ["mlf_014", "mlf_035", "mlf_024"]),
    # ========================================================================
    # GENERATED_V3 AGENT ARCHITECTURES (tasks 261-270)
    # ========================================================================
    # task_261: Designing a Self-Healing Agent with Formal Verification Constraints (hard)
    "task_261": ("aga_009", ["aga_009", "aga_033", "aga_008"]),
    # task_262: Context Window Management Under Adversarial Information Density (hard)
    "task_262": ("aga_010", ["aga_010", "aga_032", "aga_003"]),
    # task_263: Multi-Model Agent Routing with Capability-Aware Orchestration (hard)
    "task_263": ("aga_034", ["aga_034", "aga_029", "mag_013"]),
    # task_264: Designing Agents with Persistent World Models (hard)
    "task_264": ("aga_022", ["aga_022", "aga_003", "aga_013"]),
    # task_265: Bounded Rationality Agent Design Under Compute Budgets (hard)
    "task_265": ("aga_029", ["aga_029", "aga_016", "creative_009"]),
    # task_266: Agent Memory Consolidation Inspired by Cognitive Architecture (hard)
    "task_266": ("aga_026", ["aga_026", "aga_022", "mag_030"]),
    # task_267: Designing Interpretable Agent Decision Traces for Regulatory Compliance (hard)
    "task_267": ("aga_025", ["aga_025", "aga_017", "aga_027"]),
    # task_268: Hierarchical Planning with Temporal Abstraction for Long-Horizon Tasks (hard)
    "task_268": ("aga_020", ["aga_020", "aga_011", "aga_004"]),
    # task_269: Designing Agents Robust to Prompt Injection and Indirect Manipulation (hard)
    "task_269": ("aga_012", ["aga_012", "aga_033", "aga_018"]),
    # task_270: Co-Evolutionary Agent and Environment Design (hard)
    "task_270": ("aga_015", ["aga_015", "mlf_024", "mag_027"]),
    # ========================================================================
    # GENERATED_V3 RETRIEVAL SYSTEMS (tasks 271-280)
    # ========================================================================
    # task_271: Debugging Embedding Drift in Production RAG Systems (hard)
    "task_271": ("ret_019", ["ret_019", "ret_004", "ret_001"]),
    # task_272: Multi-Hop Retrieval with Evidence Chain Verification (hard)
    "task_272": ("ret_014", ["ret_014", "ret_028", "ret_029"]),
    # task_273: Retrieval System Design for Heterogeneous Enterprise Knowledge (hard)
    "task_273": ("ret_034", ["ret_034", "ret_027", "ret_015"]),
    # task_274: Adversarial Document Injection Defense for RAG Systems (hard)
    "task_274": ("ret_012", ["ret_012", "evm_021", "ret_029"]),
    # task_275: Designing Retrieval for Temporal Reasoning Over Evolving Knowledge (hard)
    "task_275": ("ret_015", ["ret_015", "ret_034", "ret_016"]),
    # task_276: Optimizing Retrieval Latency for Real-Time Conversational AI (hard)
    "task_276": ("ret_020", ["ret_020", "ret_005", "ret_026"]),
    # task_277: Cross-Modal Retrieval with Alignment Uncertainty (hard)
    "task_277": ("ret_023", ["ret_023", "ret_017", "ret_003"]),
    # task_278: Retrieval-Augmented Reasoning with Structured and Unstructured Knowledge (hard)
    "task_278": ("ret_024", ["ret_024", "ret_018", "ret_012"]),
    # task_279: Self-Improving Retrieval Through Implicit Feedback Mining (hard)
    "task_279": ("ret_032", ["ret_032", "ret_022", "evm_023"]),
    # task_280: Retrieval System Architecture for Regulatory Compliance Document Analysis (hard)
    "task_280": ("ret_031", ["ret_031", "ret_015", "ret_002"]),
    # ========================================================================
    # GENERATED_V3 MULTI-AGENT SYSTEMS (tasks 281-290)
    # ========================================================================
    # task_281: Designing Fault-Tolerant Consensus Under Partial Network Partitions (hard)
    "task_281": ("mag_006", ["mag_006", "mag_002", "mag_035"]),
    # task_282: Multi-Agent Negotiation for Resource Allocation Under Incomplete Information (hard)
    "task_282": ("mag_022", ["mag_022", "mag_027", "mag_011"]),
    # task_283: Emergent Division of Labor in Self-Organizing Agent Swarms (hard)
    "task_283": ("mag_019", ["mag_019", "mag_007", "mag_010"]),
    # task_284: Designing Trust Propagation in Hierarchical Multi-Agent Systems (hard)
    "task_284": ("mag_015", ["mag_015", "mag_016", "mag_004"]),
    # task_285: Multi-Agent Debate with Structured Argumentation and Epistemic Logic (hard)
    "task_285": ("mag_003", ["mag_003", "mag_011", "mag_021"]),
    # task_286: Scalable Multi-Agent Simulation for Policy Testing (hard)
    "task_286": ("mag_032", ["mag_032", "mag_024", "operational_008"]),
    # task_287: Cross-Organizational Multi-Agent Collaboration with Competing Interests (hard)
    "task_287": ("mag_033", ["mag_033", "mag_017", "mag_027"]),
    # task_288: Multi-Agent Learning Under Non-Stationary Reward Distributions (hard)
    "task_288": ("mag_027", ["mag_027", "mag_030", "evm_026"]),
    # task_289: Designing Governance Mechanisms for Autonomous Agent Collectives (hard)
    "task_289": ("mag_006", ["mag_006", "mag_011", "mag_033"]),
    # task_290: Hybrid Human-Agent Teams with Asymmetric Capabilities (hard)
    "task_290": ("mag_014", ["mag_014", "mag_007", "mag_015"]),
    # ========================================================================
    # GENERATED_V3 EVALUATION METHODS (tasks 291-300)
    # ========================================================================
    # task_291: Designing Evaluation for Stochastic Systems with High Variance Outputs (hard)
    "task_291": ("evm_014", ["evm_014", "evm_012", "evm_031"]),
    # task_292: Evaluating Faithfulness of Chain-of-Thought Reasoning (hard)
    "task_292": ("evm_028", ["evm_028", "aga_005", "evm_025"]),
    # task_293: Multi-Dimensional Evaluation with Conflicting Stakeholder Requirements (hard)
    "task_293": ("evm_027", ["evm_027", "evm_025", "synthesis_002"]),
    # task_294: Detecting and Measuring Data Contamination in LLM Benchmarks (hard)
    "task_294": ("evm_009", ["evm_009", "evm_034", "evm_032"]),
    # task_295: Evaluating Long-Horizon Agent Performance with Sparse Rewards (hard)
    "task_295": ("evm_028", ["evm_028", "evm_023", "aga_029"]),
    # task_296: Designing Evaluation for Multi-Agent System Emergent Properties (hard)
    "task_296": ("mag_032", ["mag_032", "mag_019", "operational_008"]),
    # task_297: Comparative Evaluation Under Non-Identical Test Conditions (hard)
    "task_297": ("evm_031", ["evm_031", "evm_035", "evm_014"]),
    # task_298: Evaluating Model Safety Across Cultural and Linguistic Contexts (hard)
    "task_298": ("evm_025", ["evm_025", "evaluation_008", "evm_020"]),
    # task_299: Dynamic Benchmark Generation to Prevent Overfitting to Static Tests (hard)
    "task_299": ("evm_034", ["evm_034", "evaluation_005", "evm_032"]),
    # task_300: End-to-End Evaluation of Retrieval-Augmented Generation with Attribution (hard)
    "task_300": ("ret_012", ["ret_012", "evm_022", "evm_006"]),
}


# ============================================================================
# BUILD OUTPUT AND VALIDATE
# ============================================================================

# Build the ground truth array
ground_truth = []
for task in tasks:
    tid = task["task_id"]
    primary, relevant = annotations[tid]
    ground_truth.append(
        {"task_id": tid, "relevant_skill_ids": relevant, "primary_skill_id": primary}
    )

# Validate: count skill usage
all_skill_ids = set(s["skill_id"] for s in skills)
excluded_domains = [
    "research",
    "creative",
    "synthesis",
    "evaluation",
    "operational",
]
domain_skill_ids = set(s["skill_id"] for s in skills if s["domain"] not in excluded_domains)
used_skill_ids = set()
for entry in ground_truth:
    for sid in entry["relevant_skill_ids"]:
        used_skill_ids.add(sid)
        if sid not in all_skill_ids:
            print(f"WARNING: {entry['task_id']} references non-existent skill {sid}")

# Check domain skill coverage
unused_domain_skills = domain_skill_ids - used_skill_ids
if unused_domain_skills:
    print(f"WARNING: {len(unused_domain_skills)} domain-specific skills never used:")
    for sid in sorted(unused_domain_skills):
        s = skill_by_id[sid]
        print(f"  {sid}: {s['title']}")

# Stats
total_skills_assigned = sum(len(e["relevant_skill_ids"]) for e in ground_truth)
avg_skills = total_skills_assigned / len(ground_truth)
print("\nStats:")
print(f"  Total tasks: {len(ground_truth)}")
print(f"  Total skills assigned: {total_skills_assigned}")
print(f"  Average skills per task: {avg_skills:.2f}")
print(f"  Domain skills used: {len(domain_skill_ids & used_skill_ids)}/{len(domain_skill_ids)}")
print(f"  Total unique skills used: {len(used_skill_ids)}/{len(all_skill_ids)}")

# Skill usage counts
skill_usage = Counter()
for entry in ground_truth:
    for sid in entry["relevant_skill_ids"]:
        skill_usage[sid] += 1

print(f"\n  Min usage of any used skill: {min(skill_usage.values())}")
print(f"  Max usage of any used skill: {max(skill_usage.values())}")

# Check which skills are used most/least
print("\n  Most used skills:")
for sid, count in skill_usage.most_common(10):
    print(f"    {sid}: {count} ({skill_by_id[sid]['title']})")

print("\n  Least used skills:")
for sid, count in skill_usage.most_common()[-10:]:
    print(f"    {sid}: {count} ({skill_by_id[sid]['title']})")

# Write output
output_path = str(_DIR / "ground_truth_v3.json")
with open(output_path, "w") as f:
    json.dump(ground_truth, f, indent=2)

print(f"\nGround truth written to {output_path}")
