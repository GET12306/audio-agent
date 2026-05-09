# audio-agent

## Setup & dependencies

- **macOS-only** (Apple Silicon). Uses MLX for GPU acceleration via `mlx-whisper` and `demucs-mlx`.
- **Python 3.12** (`.python-version`). Package manager is **`uv`** (not pip/poetry).
  - `uv sync` to install, `uv run python <script.py>` to execute.
- **Ollama must be running locally** (`ollama serve`) with `mxbai-embed-large` (used by `OllamaEmbeddings`). No Ollama model needed for the LLM — default is `ChatDeepSeek` (`deepseek-v4-flash`).
- `.env` loaded via `python-dotenv` at module import time. Requires `DEEPSEEK_API_KEY`. See `.env.example`.

## Entrypoints & architecture

| File | Role |
|---|---|
| `agent.py` | **Interactive ReAct agent.** Entry point for the system. Runs via `uv run python agent.py` |
| `engine.py` | Core logic (transcribe, calibrate, summarize, QA chain) — pure functions, no agent coupling |
| `build_kb.py` | Standalone script to build FAISS KB from `data/episodes/` |

All pipeline steps are exposed as `@tool` functions in `agent.py`. The agent decides which to call based on user input.

## Key conventions & quirks

- **Japanese audio focus**: `initial_prompt` in mlx-whisper calls includes name mappings (e.g., はやしここ → 林鼓子). Transcription JSON format: `{"full_text": str, "segments": [{"start": float, "end": float, "text": str}]}`.
- Default LLM is `ChatDeepSeek(model="deepseek-v4-flash")`. Alternative `ChatOllama` with `gemma4:26b` is commented out in `engine.py`.
- Embeddings use **`mxbai-embed-large`** via Ollama (not the LLM model).
- **Prompts are external YAML files** in `prompts/` (one per pipeline step). Select variant via `PROMPT_SELECT` env var (default: `prompt_1`).
- **FAISS knowledge base is persisted to disk** via `build_kb.py`. Scans `data/episodes/` for `*_calibrated.json`, chunks with episode metadata, saves index to `data/kb/`. `engine.py` loads from this KB for QA instead of building in-memory from a single file.
- No tests, no formatter, no linter, no CI configured. `pyproject.toml` has no `[tool.ruff]`, `[tool.pytest]`, etc.
- `audio_transcribe.py` and `test_demucs.py` are personal utility scripts, gitignored — not part of the agent.
- Input JSON is expected from an external transcription step (e.g., `mlx_whisper`); the agent does not call the transcriber itself.
