# LangGraph Sequential Prompt Chain — README

## Overview

This repository contains a small Streamlit app (`app.py`) that builds a dynamic sequential prompt chain using LangGraph and a Chat LLM wrapper. The app loads an `OPENAI_API_KEY` from a `.env` file and runs prompts sequentially, accumulating step outputs.

Key features:
- Define an ordered list of prompts that reference the running cumulative text using `{{text}}`.
- Run the prompts through a Chat LLM (`ChatOpenAI`) to produce step outputs and a final result.
- Simple Streamlit UI to add, reorder, delete prompts and view intermediate outputs.

> Note: This README assumes you are on Windows using PowerShell (your environment). Adjust commands if you use a different shell.

---

## Prerequisites

- Python 3.10+ recommended.
- (Optional) A virtual environment (the project uses `.venv` by default).
 # LangGraph Prompt Chain — README

## Overview

This repository contains a small Streamlit front-end (`app.py`) and a backend workflow module (`workflow.py`).

- `app.py` is a lightweight Streamlit UI that collects a dynamic list of prompts and delegates execution to the backend workflow.
- `workflow.py` builds and runs a LangGraph-based sequential workflow that calls a Chat LLM for each prompt and accumulates outputs.

The app reads the OpenAI API key from a `.env` file and relies on LangChain + a provider integration (OpenAI) to create the chat model used by the workflow.

> This README assumes you're on Windows using PowerShell. Adjust commands if you use another shell.

---

## Key files

- `app.py` — Streamlit UI; imports and calls `workflow.run_prompt_workflow(prompts)`.
- `workflow.py` — Backend: initializes an LLM and builds a LangGraph graph to execute prompts sequentially.
- `.env` — Place `OPENAI_API_KEY` here (not committed).
- `requirements.txt` — Project dependencies (see setup below).

---

## Required packages

The project depends on the following packages (already placed in `requirements.txt`):

- `python-dotenv` — load `.env` variables
- `streamlit` — UI
- `langchain` — LangChain core
- `langchain-openai` — LangChain OpenAI provider integration (provides concrete ChatOpenAI)
- `langgraph` — graph runtime used by the workflow
- `openai` — OpenAI Python client (provider dependency)
- `pydantic` — state models used in `workflow.py`

If you need reproducible installs, pin exact versions in `requirements.txt` after confirming compatibility.

---

## Setup (PowerShell)

1. Open PowerShell and cd into the project:

```powershell
cd C:\prompt\prompt-new
```

2. Create and activate a virtual environment if you haven't already:

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
```

If activation is blocked due to execution policy, allow scripts for the current user:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\\.venv\\Scripts\\Activate.ps1
```

3. Install dependencies (with the venv active):

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

If you prefer not to activate the venv, use the explicit python/pip from `.venv`:

```powershell
.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt
```

4. Add your OpenAI key to `.env` (create the file if missing):

```
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

`workflow.py` uses LangChain's `ChatOpenAI` (via provider); make sure `langchain-openai` is installed so the provider is available to LangChain.

---

## Running the app

Start Streamlit from the project root (venv activated):

```powershell
streamlit run app.py
```

Or (explicit venv executable):

```powershell
.\\.venv\\Scripts\\streamlit.exe run app.py
```

Streamlit will print a local URL (e.g. `http://localhost:8501`). The UI allows you to add prompts and run the workflow; `app.py` will call `workflow.run_prompt_workflow(prompts)` and show the final output.

---

## How the workflow works (brief)

- `workflow.py` loads `OPENAI_API_KEY` via `dotenv`.
- It creates a `ChatOpenAI` (LangChain provider) and defines a `PromptState` model with `pydantic`.
- For each prompt the workflow adds a `FunctionNode` (from `langgraph`) that executes the prompt against the LLM and appends the result to the running context.
- Finally the graph is run and the cumulative context is returned to the UI.

Because LangChain providers and versions vary, `langchain-openai` must be present so the chat model implementation is available.

---

## Troubleshooting

- `OPENAI_API_KEY not found`: ensure `.env` exists in the project root and contains `OPENAI_API_KEY=...`. Start Streamlit from the project root.
- `ModuleNotFoundError` for `langchain` / `langchain_openai` / `langgraph`: install the packages listed in `requirements.txt`. If the package names differ in your environment, adjust `requirements.txt` accordingly.
- If LangChain's public API changes, you may see errors like `cannot import name 'ChatOpenAI'`; installing the provider package (`langchain-openai`) or using LangChain's `init_chat_model` factory resolves this.

---

## Security

- `.env` and `.venv` are in `.gitignore` to avoid committing secrets and the virtual environment.

---

## Next steps I can help with

- Run `pip install -r requirements.txt` inside `.venv` and report issues.
- Start the Streamlit server and capture logs if you hit runtime errors.
- Add a mock-based unit test for `workflow.run_prompt_workflow` so you can test locally without external API calls.

---

If you want me to perform any of those actions, tell me which one and I'll proceed.

---

## Docker (Build & Run)

This project includes a multi-stage `Dockerfile`, a `.dockerignore`, and a `docker-compose.yml` for local containerized runs. Below are exact PowerShell commands to build and run the app locally on Windows. Do NOT commit your `.env` file or API keys.

- **Create `.env` (PowerShell)**: replace the key with your own OpenAI key

```powershell
cd C:\prompt\prompt-new
Set-Content -Path .env -Value "OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

- **Build image**:

```powershell
cd C:\prompt\prompt-new
docker build -t prompt-chain:local .
```

- **Run container (recommended: use `.env`)**:

```powershell
docker run --rm --env-file .env -p 8501:8501 prompt-chain:local
```

- **Run using Docker Compose** (build + run):

```powershell
cd C:\prompt\prompt-new
docker compose up --build
# Or detached:
docker compose up --build -d
```

- **Open the app**: http://localhost:8501

- **View logs / debug**:

```powershell
docker ps
docker logs -f <container-id-or-name>
# For compose
docker compose logs -f
```

Notes & tips:
- The `Dockerfile` is multi-stage and builds wheels in a `builder` stage so the final image is smaller and contains only runtime packages.
- `.dockerignore` excludes `.env`, `.venv`, and git metadata so secrets and dev artifacts are not copied into the image.
- If a package in `requirements.txt` needs system libraries, add them to the `apt-get install` lines in the `Dockerfile` (builder stage).
- Monitor your OpenAI usage — any user using the public app will consume your API quota and may incur costs. To reduce risk:
	- Require users to enter their own `OPENAI_API_KEY` in the UI (shifts cost to them).
	- Add authentication or a server-side proxy with rate limits.
	- Use the OpenAI dashboard to set billing alerts.

