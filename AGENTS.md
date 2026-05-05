# audio-agent

## Setup & dependencies

- **macOS-only** (Apple Silicon). Uses MLX for GPU acceleration via `mlx-whisper` and `demucs-mlx`.
- **Python 3.12** (`.python-version`). Package manager is **`uv`** (not pip/poetry).
  - `uv sync` to install, `uv run python <script.py>` to execute.
- **Ollama must be running locally** (`ollama serve`) with `mxbai-embed-large` (used by `OllamaEmbeddings`). No Ollama model needed for the LLM — default is `ChatDeepSeek` (`deepseek-v4-flash`).
- `.env` loaded via `python-dotenv` at module import time. Requires `DEEPSEEK_API_KEY` and `TRANSCRIPTION_JSON_PATH`. See `.env.example`.

## Entrypoints & architecture

| File | Role |
|---|---|
| `agent_core.py` | Main pipeline: calibrate → summarize → QA. Runs via `uv run python agent_core.py` |
| `audio_tools.py` | LangChain `@tool` wrapping `mlx-whisper` transcription |

Pipeline flow:
1. Read transcription JSON (path from `TRANSCRIPTION_JSON_PATH` env var)
2. LLM calibrates (fixes Japanese homophone errors, deduplicates, timestamps preserved)
3. LLM summarizes (outputs Chinese)
4. FAISS vector store + LLM for timestamp-aware QA

## Key conventions & quirks

- **Japanese audio focus**: `initial_prompt` in mlx-whisper calls includes name mappings (e.g., はやしここ → 林鼓子). Transcription JSON format: `{"full_text": str, "segments": [{"start": float, "end": float, "text": str}]}`.
- Default LLM is `ChatDeepSeek(model="deepseek-v4-flash")`. Alternative `ChatOllama` with `gemma4:26b` is commented out in `agent_core.py`.
- Embeddings use **`mxbai-embed-large`** via Ollama (not the LLM model).
- **Prompts are external YAML files** in `prompts/` (one per pipeline step). Select variant via `PROMPT_SELECT` env var (default: `prompt_1`).
- No tests, no formatter, no linter, no CI configured. `pyproject.toml` has no `[tool.ruff]`, `[tool.pytest]`, etc.
- `audio_transcribe.py` and `test_demucs.py` are personal utility scripts, gitignored — not part of the agent.
- Input JSON is expected from an external transcription step (e.g., `mlx_whisper`); the agent does not call the transcriber itself.
