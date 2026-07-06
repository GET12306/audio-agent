# audio-agent

## Setup & dependencies

- **macOS-only** (Apple Silicon). Uses MLX for GPU-accelerated transcription via `mlx-whisper`.
- **Python 3.12** (`.python-version`). Package manager is **`uv`** (not pip/poetry).
  - `uv sync` to install dependencies.
  - `uv run python <script.py>` or `uv run <script.py>` to execute project scripts.
- **Ollama must be running locally** (`ollama serve`) with `mxbai-embed-large` for `OllamaEmbeddings`.
- `.env` is loaded with `python-dotenv`. Required: `DEEPSEEK_API_KEY`. Optional: `KB_PATH` (default: `data/kb`). See `.env.example`.
- No Hugging Face token or diarization dependency is required in the current version.

## Entrypoints & architecture

| File | Role |
|---|---|
| `agent.py` | **Interactive ReAct agent.** Entry point for the system. Runs via `uv run python agent.py` |
| `engine.py` | Core logic (transcribe, calibrate, summarize, QA chain) — pure functions, no agent coupling |
| `deepseek_model.py` | Custom `BaseChatModel` subclass. Captures DeepSeek `reasoning_content` for display, but does not send it back in follow-up API requests |
| `build_kb.py` | Standalone script to rebuild the FAISS KB from `data/episodes/` |
| `audio_tools.py` | Legacy standalone transcription helper; not used by the main agent |

All pipeline steps are exposed as `@tool` functions in `agent.py`. The agent decides which to call based on user input.

## Current scope

- This is a **single-speaker Japanese radio** workflow.
- No speaker diarization, no `HF_TOKEN`, and no `pyannote-audio`.
- No incremental KB manifest. `build_kb.py` rebuilds the FAISS index from all `data/episodes/*_calibrated.json` files.

## Key conventions & quirks

- **Japanese audio focus**: `initial_prompt` in `mlx-whisper` calls includes name mappings (for example, はやしここ → 林鼓子).
- Transcription JSON format: `{"full_text": str, "segments": [{"start": float, "end": float, "text": str}]}`.
- `calibrate_transcription` writes calibrated files to `data/episodes/` and returns the actual saved path.
- `summarize_episode` returns the summary content. It also caches the summary as `*_summary.md` beside the calibrated JSON.
- Default LLM is `DeepSeekReasoning` in `deepseek_model.py`. Alternative local `ChatOllama` usage is commented out in `engine.py`.
- Embeddings use **`mxbai-embed-large`** via Ollama; this is separate from the LLM.
- Prompts are external YAML files in `prompts/`. Select variants via `PROMPT_SELECT` env var (default: `prompt_1`).
- `data/` and `transcribe_result/` are gitignored generated data directories.
- No tests, formatter, linter, or CI are configured yet.

## Tools exposed by `agent.py`

- `transcribe_audio(audio_path, model_path="mlx-community/whisper-large-v3-turbo")`
- `calibrate_transcription(raw_json_path)`
- `summarize_episode(calibrated_json_path)`
- `search_knowledge_base(query, episode="")`
- `rebuild_knowledge_base()`
