# LLM.RAG.Python 🤖📚

A complete Retrieval-Augmented Generation (RAG) implementation in Python, demonstrating multiple RAG strategies using modular, production-ready components.

This project is designed for:
- Learning RAG architectures
- Comparing multiple query-routing strategies
- Building real-world AI search assistants
- LLM + Vector DB based applications

---

## 🚀 Features

- Multiple RAG strategies (Hybrid, Router, Multi-Query, Conditional)
- Unified Agent interface
- Document ingestion pipeline
- Re-ranking support
- Config-driven setup via `.env`
- Jupyter demo notebooks
- Modular and extensible architecture

---

## 📂 Project Structure

LLM.RAG.Python/
```
├── test_data/
├── .env.example
├── README.md
├── UnifiedAgent.py
├── UnifiedAgent_demo.ipynb
├── approach_b_conditional.py
├── approach_c_hybrid.py
├── approach_d_router.py
├── approach_e_multiquery.py
├── data_loader.py
├── demo.ipynb
├── ingest.py
├── query.py
├── rag.py
├── rerank.py
└── requirements.txt
```

---

## 🛠️ Installation

```bash
git clone https://github.com/nikhilpateldev/LLM.RAG.Python.git
cd LLM.RAG.Python
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

---

## ▶️ How to Run

```bash
python ingest.py
python query.py
python UnifiedAgent.py
```

---

## 📜 License

MIT License
