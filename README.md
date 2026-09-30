# EquityMind

AI-powered equity research assistant built with **Python, FastAPI, LangGraph, Streamlit, OpenAI, RAG, ChromaDB, and guardrails**.

## Features

* AI-powered equity research
* LangGraph-based routing and workflow
* RAG with ChromaDB
* Input, execution and output guardrails
* Human-in-the-Loop
* Streaming responses using SSE
* Streamlit frontend
* FastAPI backend

## Project Structure

```text
EquityMind/
├── backend/       # FastAPI + LangGraph
├── streamlit/     # Streamlit frontend
├── README.md
└── .gitignore
```

## Requirements

* Python 3.11+
* OpenAI API key

## Setup

### 1. Backend

Open Terminal 1:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create `.env` from `.env.example` and add your OpenAI API key.

Then start the backend:

```powershell
python -m uvicorn app.main:app --reload --port 8000
```

Backend:

```text
http://localhost:8000
```

---

### 2. Frontend

Open Terminal 2:

```powershell
cd streamlit
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Start Streamlit:

```powershell
python -m streamlit run app.py
```

Open:

```text
http://localhost:8501
```

---

## Run the Project

Every time you want to run EquityMind:

**Terminal 1**

```powershell
cd EquityMind\backend
.\.venv\Scripts\Activate.ps1
python -m uvicorn app.main:app --reload --port 8000
```

**Terminal 2**

```powershell
cd EquityMind\streamlit
.\.venv\Scripts\Activate.ps1
python -m streamlit run app.py
```

Then open:

**http://localhost:8501**

---

## Environment Variables

Create:

```text
backend/.env
```

Example:

```text
OPENAI_API_KEY=your_api_key
OPENAI_MODEL=gpt-4.1-mini
RERANK_ENABLED=true
FRONTEND_ORIGIN=http://localhost:8501
```

**Never commit `.env` or your API key to GitHub.**

## Tech Stack

**Frontend:** Streamlit
**Backend:** FastAPI
**AI Workflow:** LangGraph
**LLM:** OpenAI
**RAG:** ChromaDB
**Language:** Python
**Streaming:** SSE

## Disclaimer

EquityMind is an educational project and is **not financial advice**.
