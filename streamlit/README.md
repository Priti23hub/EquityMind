# EquityMind Streamlit frontend

This folder contains the Streamlit UI only. All AI processing remains in the
existing FastAPI backend, including LangGraph routing, checkpoints, streaming,
RAG, tools, and human-in-the-loop interrupts.

## Run the application on Windows PowerShell

Open two PowerShell terminals from the project root.

### 1. Create the Streamlit environment

```powershell
cd streamlit
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run this once in that terminal and activate
again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 2. Install Streamlit dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Start FastAPI

In the second terminal, from the project root:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m uvicorn app.main:app --reload --port 8000
```

The backend must have its existing `.env` configured with the required OpenAI
settings. Do not put API keys in Streamlit code.

### 4. Start Streamlit

In the first terminal, from the project root:

```powershell
cd streamlit
.\.venv\Scripts\Activate.ps1
python -m streamlit run app.py
```

### 5. Open the application

Open [http://localhost:8501](http://localhost:8501) in a browser. Keep the
FastAPI terminal running at `http://localhost:8000` while using the app.

The app sends normal messages to `/api/chat`. When LangGraph pauses for a
human answer, it displays the question and sends the answer to
`/api/chat/resume` using the same checkpoint thread ID.