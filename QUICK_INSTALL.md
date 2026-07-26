# Quick Install — one script, everything running

The fastest way to run Atlas AI: a single script installs all dependencies
(first run only) and starts both the backend and the web UI.

## Prerequisites

- **Python 3.12** — https://www.python.org/downloads/
- **Node.js 18+** (includes npm) — https://nodejs.org/
- A **`.env` file in the project root** with the API keys.
  If you don't have one, copy the template and fill it in:

  ```bash
  cp .env.example .env
  ```

## Run it

**macOS / Linux:**

```bash
./quick_install.sh
```

(If you get a permission error: `bash quick_install.sh`.)

**Windows (PowerShell):**

```powershell
powershell -ExecutionPolicy Bypass -File .\quick_install.ps1
```

## What the script does

1. Checks that `.env`, Python, and npm are present (with a clear error if not).
2. Creates a virtualenv at `.venv/` and installs `requirements-server.txt`
   — **first run only**; later runs skip straight to starting the servers.
   pip may print a `protobuf` dependency-conflict warning — it is harmless.
3. Runs `npm install` inside `atlas-web/` — first run only.
4. Starts the **backend** (LangGraph dev server) at `http://127.0.0.1:2024`
   and waits until it responds.
5. Starts the **web UI** at **http://localhost:3000** — open that in your
   browser and start chatting (e.g. *"Plan a 4-day trip from Tel Aviv to Rome
   in October for 2 adults"*).

Press **Ctrl+C** in the terminal to stop both servers.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `no .env file found` | Copy `.env.example` to `.env` in the project root and fill in the keys. |
| Backend fails to start / port error | Something else is using port 2024 or 3000 — stop it and re-run. |
| Web UI shows errors when sending a message | Check the backend terminal output; usually a missing/invalid API key in `.env`. |
| Want a clean reinstall | Delete `.venv/` and `atlas-web/node_modules/`, then re-run the script. |

For manual step-by-step setup, the CLI version, and running the tests, see [README.md](README.md).
