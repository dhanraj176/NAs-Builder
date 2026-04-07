# AutoArchitect AI
### Autonomous Agent Network Generator — Oakland Research Showcase 2026

> **Describe any problem in plain English. Get a fully trained, deployable AI agent network in minutes.**

[![Python](https://img.shields.io/badge/Python-3.13-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-ResNet18-orange)](https://pytorch.org)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-1.5.5-green)](https://trychroma.com)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## What Is AutoArchitect?

AutoArchitect is a self-improving autonomous ML pipeline that converts plain-English problem descriptions into fully trained, downloadable agent networks — with zero human intervention in architecture design, dataset selection, or model training.

**No ML expertise required. No manual configuration. Just describe the problem.**

---

## Live Demo

```
Input:  "detect illegal dumping in Oakland street cameras"

Output: Trained ResNet18 model — 81.11% test accuracy
        3-agent network: Image → Severity → Report
        Downloadable zip with run_network.py
        Stored in ChromaDB brain for instant future recall
```

Second run of the same problem:
```
ChromaDB recall: 100% match — result in 1.07 seconds
```

---

## Proven Results

| Problem | Domain | Agents | Accuracy |
|---------|--------|--------|----------|
| Illegal dumping detection | Image | 3 | **81.11%** |
| Pothole & road damage | Image | 3 | 100%* |
| Toxic comment detection | Text | 2 | **92.8%** |
| Spam classification | Text | 1 | 86.0% |
| Pneumonia detection (X-ray) | Medical | 1 | 100%* |
| Sentiment analysis (IMDB) | Text | 2 | 100%* |
| Multi-domain Oakland monitoring | Image | 3 | **81.4%** |

**Infrastructure cost: $0** — fully local, open-source stack.

*100% accuracy reported on the test split used during evaluation.

---

## Architecture

### High-Level Pipeline

```
User Input (plain English)
        ↓
BERT Domain Classifier (89% confidence)
        ↓
ChromaDB Stage 0 — instant recall if seen before
        ↓ (cache miss)
Data Discovery Engine — HuggingFace + Crawl4AI + Web Search
        ↓
DARTS Neural Architecture Search
        ↓
ResNet18 / BoW Transfer Learning (real backprop)
        ↓
Multi-Agent Fusion (Image + Severity + Report)
        ↓
Real Evaluator — weighted accuracy score
        ↓
Downloadable Agent Network ZIP
        ↓
ChromaDB Brain — stores solution for future recall
```

### Component Architecture

![Component Architecture](autoarchitect/docs/Architecture-component.png)

The diagram above shows the full component breakdown across four subsystems:

- **Orchestration Layer** — `app.py` → `orchestrator.py` routes requests, manages the ChromaDB recall shortcut, and coordinates all downstream modules.
- **Brain / Strategy Engine** — `topology_designer`, `agent_generator`, `meta_learner`, `strategy_library`, `data_discovery_engine`, and `web_researcher` collaborate to select the optimal pipeline for each problem.
- **Training Pipeline** — `nas_engine` (DARTS), `transfer_trainer` (ResNet18), `self_trainer`, and `auto_trainer` execute real backprop and report metrics to `performance_tracker` and `self_evaluator`.
- **Agent Network Layer** — `agent_factory` + `agent_network` instantiate domain-specific agents (Image, Text, Medical, Security, Fusion) and wire them into a runnable topology exported as a ZIP.

---

## 10 Research Contributions

1. **ANAS** — Agent Network Architecture Search (novel extension of NAS to multi-agent level)
2. **Multi-Agent NAS** — parallel domain specialist agents fused into one network
3. **Automatic Dataset Selection** — semantic HuggingFace matching via BERT + Crawl4AI
4. **ResNet18 Transfer Learning** — 20% → 82%+ accuracy automatically
5. **BERT Semantic Cache** — 2,066x speedup on repeat problems
6. **Self-Learning Strategy Brain** — 28 strategies, auto-discovers new ones
7. **Meta-Learning Pipeline Selector** — neural net predictor trained on real outcomes
8. **ChromaDB Architectural Memory** — persistent solution store across sessions
9. **Multi-Source Dataset Discovery** — HuggingFace + Kaggle + OpenImages + Web
10. **Dynamic Named Agent Factory** — purpose-built agents for any domain

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Domain Classification | BERT (417MB, local) |
| Architecture Search | DARTS NAS |
| Image Training | ResNet18 Transfer Learning |
| Text Training | Bag-of-Words + custom NN |
| Memory | ChromaDB 1.5.5 |
| Web Research | Crawl4AI + DuckDuckGo |
| LLM Output | Groq (Llama 3, optional) |
| Backend | Flask |
| ML Framework | PyTorch |

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/dhanraj176/NAs-Builder.git
cd NAs-Builder/autoarchitect

# 2. Install dependencies
pip install torch torchvision flask pillow numpy transformers
pip install chromadb crawl4ai duckduckgo-search groq datasets
```

> **Windows only:** symlinks require Developer Mode — enable it at  
> Settings → Privacy & Security → Developer Mode → ON

```bash
# 3. Create a .env file and set environment variables
echo "GROQ_API_KEY=your_key_here" >> .env        # optional — for LLM-generated reports
echo "HF_DATASETS_CACHE=./datasets/hf_cache" >> .env

# 4. Run
python app.py
# Open http://localhost:5000
```

---

## Verified Working Datasets

| Domain | Dataset |
|--------|---------|
| Garbage / Illegal Dumping | `dmedhi/garbage-image-classification-detection` |
| Potholes / Road Damage | `taroii/pothole-detection` |
| Pneumonia / X-Ray | `hf-vision/chest-xray-pneumonia` |
| Skin Cancer | `marmal88/skin_cancer` |
| Spam Detection | `sms_spam` |
| Sentiment Analysis | `imdb` |
| Fake News | `GonzaloA/fake_news` |
| Toxic Comments | `SetFit/toxic_conversations_50k` |

---

## Research Context

This project was developed as independent research at **Northeastern University Oakland** and presented at the **Oakland Research Showcase, April 10, 2026**.

The core novel contribution — **ANAS (Agent Network Architecture Search)** — extends traditional Neural Architecture Search from the single-model level to the multi-agent network level, enabling automatic design of collaborative agent topologies for complex real-world problems.

**Potential publication venues:** AutoML Conference, NeurIPS Workshops, ICLR Workshops

---

## Author

**Dhanraj Atul Pandya**  
MS Computer Science Engineering  
Northeastern University Oakland  
GitHub: [@dhanraj176](https://github.com/dhanraj176)

---

## License

MIT License — open for research and collaboration.
