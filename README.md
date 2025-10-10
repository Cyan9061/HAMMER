# ⚒️ HAMMER: **Hierarchical Memory-guided Monte Carlo Tree Search for RAG System Optimization**


## 🌟 Highlights

✨ **HAMMER (HierArchical Memory-guided Monte Carlo TreE Search)** is a next-generation framework for **Retrieval-Augmented Generation (RAG)** optimization.
It brings **human-like reasoning** and **intelligent memory** into hyperparameter tuning — achieving **SOTA performance** with **drastically lower cost**.

### 🧠 Core Idea

HAMMER mimics **human learning**:

1. **Remember** past experiments via a **hierarchical graph memory**
2. **Reason** with stored insights during Monte Carlo Tree Search
3. **Refine** evaluations using selective query sampling and LLM-guided estimation

---

## 🚀 Why HAMMER?

| 🔍 Metric            | 📊 Improvement           |
| -------------------- | ------------------------ |
| **Exact Match (EM)** | ⬆️ +20.0%                |
| **F1-score**         | ⬆️ +15.2%                |
| **Tuning Time**      | ⬇️ 9× faster             |
| **Token Cost**       | ⬇️ 9× fewer              |
| **Interpretability** | 🧩 Memory-based insights |

HAMMER doesn’t just tune faster — it **thinks smarter** 🧠.

---

## 🏗️ Architecture Overview

```text
HAMMER/
├── hammer/
│   ├── mcts/                # Monte Carlo Tree Search core
│   │   ├── optimization_engine.py
│   │   ├── hierarchical_search.py
│   │   └── kb_manager/      # 🧠 Knowledge-based memory
│   │       ├── graph_memory.py
│   │       ├── insight_agent.py
│   │       └── enhanced_evaluator.py
│   ├── tuner/               # ⚙️ Tuning modules (MCTS)
│   ├── utils/               # 🧩 Helpers & API evaluators
│   ├── Rerank/              # 🔁 Advanced reranking system
│   └── configuration.py     # 🧾 Config management
└── Experiment/              # 📈 Experimental scripts & results
```

---

## 🔬 Algorithmic Innovations

### 🧮 1. Hierarchical Memory-Guided MCTS

* **Monte Carlo Tree Search + Memory**
* **Experience Reuse:** Every tuning iteration updates a graph-structured memory
* **Memory-Guided Simulation:** Avoids random rollouts by recalling similar configurations
* **Memory-Guided Evaluation:** Estimates scores efficiently from representative subsets

### 🧩 2. Submodular Query Selection

* Selects diverse & representative queries
* Proven **½-approximation guarantee**
* Reduces evaluation cost by up to **90%**

### 🧠 3. Search Experience Bank

Three interconnected knowledge graphs:

1. **Query Graph** — preserves fine-grained execution traces
2. **Configuration Graph** — stores evaluated parameter sets
3. **Insight Graph** — distills human-like, generalizable lessons

---

## 🧪 Performance Summary

HAMMER achieves **state-of-the-art** across 8 RAG benchmarks:

| Dataset                                     | F1 ↑            | EM ↑   | Tokens ↓ | Time ↓    |
| ------------------------------------------- | --------------- | ------ | -------- | --------- |
| 2WikiMultiHopQA                             | **+15.2%**      | +20.0% | 9× fewer | 9× faster |
| HotpotQA                                    | +13.7%          | +14.8% | 8× fewer | 7× faster |
| MedQA                                       | +6.1%           | +17.0% | 5× fewer | 6× faster |
| FiQA / ELI5 / QuaRTZ / PopQA / WebQuestions | Consistent SOTA |        |          |           |

📈 Even with **fewer queries**, HAMMER matches or exceeds full-scale evaluations!

---

## 💡 Key Features

| 💎 Feature                 | Description                                             |
| -------------------------- | ------------------------------------------------------- |
| 🧠 **Hierarchical Memory** | Human-like experience accumulation and reuse            |
| 🧮 **MCTS Optimization**   | Sequential, dependency-aware hyperparameter exploration |
| ⚡ **Query Selection**      | Submodular optimization for cost-efficient coverage     |
| 🧩 **Explainability**      | Transparent, interpretable tuning process               |
| 🔬 **Multi-Domain**        | Tested on medicine, finance, science, and QA datasets   |

---

## 🧭 Example Command

Run memory-guided optimization on 2WikiMultiHopQA:

```bash
python -m hammer.tuner.main_tuner_mcts \
  --dataset 2wikimultihopqa \
  --iterations 50 \
  --optimization-target train_answer_f1 \
  --train-size 210
```

Resume from previous knowledge bank:

```bash
python -m hammer.tuner.main_tuner_mcts \
  --dataset MedQA \
  --kb-id medical_experiment_01 \
  --iterations 30
```

---

## 📊 Visualization Tools

```bash
python Experiment/query_selection/visualizations/scatter_time.py
python Experiment/query_selection/visualizations/scatter_token.py
```

Generate insights directly in Python:

```python
from hammer.mcts.kb_manager.graph_memory import GraphMemoryRAGMCTS
memory = GraphMemoryRAGMCTS()
insights = memory.get_insights_for_dataset("HotpotQA")
```

---

## 🧩 Research Significance

HAMMER is **the first** framework to:

* Combine **Hierarchical Memory** and **Monte Carlo Tree Search** for RAG tuning
* Offer **provable theoretical guarantees** in query selection
* Deliver **human-like, interpretable optimization** for LLM-based systems

🧾 **Accepted at SIGMOD 2026**, HAMMER represents a milestone in automated RAG system design.




---

## 🧭 Final Words

💥 **HAMMER = Intelligence × Memory × Efficiency**
It doesn’t just search — it learns, remembers, and reasons.

> *"From brute-force tuning to intelligent self-optimization — HAMMER is how RAG systems learn to think."* 🧠⚒️
