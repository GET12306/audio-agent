# Audio Agent

An AI-powered transcription, calibration, summarization, and Q&A system for single-speaker Japanese radio content. Built on LangChain with Apple Silicon acceleration.

## Pipeline

```
Audio → Transcription → Calibration → Summarization → Timestamp-aware RAG QA
```

| Step | What it does |
|------|-------------|
| **Transcription** | `mlx-whisper` converts speech to timestamped text |
| **Calibration** | LLM fixes homophone errors, deduplicates, preserves timestamps |
| **Summarization** | LLM produces structured bullet-point summary |
| **RAG QA** | FAISS vector store + LLM answers questions with source timestamps |

This version intentionally keeps the scope small: no speaker diarization, no Hugging Face token requirement, and no incremental indexing. The knowledge base is rebuilt from the calibrated episode files when requested.

## Prerequisites

- macOS with Apple Silicon (MLX acceleration)
- Python 3.12
- [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.ai) with `mxbai-embed-large` for local embeddings:
  ```bash
  ollama pull mxbai-embed-large
  ```

## Setup

```bash
# Install dependencies
uv sync

# Set up environment
cp .env.example .env
# Edit .env:
#   DEEPSEEK_API_KEY — your DeepSeek API key
```

## Usage

### Interactive agent (recommended)

Start the interactive ReAct agent:

```bash
uv run agent.py
```

The agent decides which tool to use based on your request. Example flow:

```
> Transcribe episode 17 and add it to the knowledge base
→ Agent calls transcribe_audio → calibrate_transcription → rebuild_knowledge_base

> What did the host say about cats?
→ Agent calls search_knowledge_base(query="cat")
→ Returns answer with episode and timestamp references

> Summarize episode 17
→ Agent calls summarize_episode
```

### Individual tools (for scripting)

You can also run each tool directly:

```bash
# Transcribe an audio file
uv run python -c "from agent import transcribe_audio; print(transcribe_audio.invoke({'audio_path': '/path/to/audio.mp4'}))"

# Build knowledge base from data/episodes/
uv run build_kb.py

# Calibrate a raw transcription
uv run python -c "from agent import calibrate_transcription; print(calibrate_transcription.invoke({'raw_json_path': '/path/to/file.json'}))"
```

Prompt profiles are selectable via `PROMPT_SELECT` env var (default: `prompt_1`).

Generated file conventions:

- Raw transcriptions are saved next to the audio file as `.json`.
- Calibrated transcripts are saved under `data/episodes/*_calibrated.json`.
- Summaries are saved next to the calibrated JSON as `*_summary.md`.
- The FAISS index is saved under `data/kb/`.

## Project Structure

| File / Dir | Purpose |
|------------|---------|
| `agent.py` | **Entry point.** ReAct agent with all tools + interactive loop |
| `engine.py` | Core logic (transcribe, calibrate, summarize, QA chain) — pure functions |
| `deepseek_model.py` | Custom `BaseChatModel` subclass. Captures DeepSeek `reasoning_content` for display, but does not send it back in follow-up API requests |
| `build_kb.py` | Build persistent FAISS knowledge base from `data/episodes/` |
| `prompts/` | YAML files for LLM prompts (one per pipeline step) |
| `data/episodes/` | Calibrated JSON files (input for knowledge base) |
| `data/kb/` | Persisted FAISS index (loaded for QA) |


## Tech Stack

- **LLM**: DeepSeek v4 Flash via custom `DeepSeekReasoning(BaseChatModel)`; swap to `ChatOllama` for local inference in `engine.py` if desired
- **Embeddings**: `mxbai-embed-large` via Ollama
- **Speech-to-text**: `mlx-whisper` (Apple Silicon)
- **Vector store**: FAISS (persisted to disk via `build_kb.py`)
- **Orchestration**: LangChain `create_agent` (ReAct loop)
- **Prompt management**: YAML files in `prompts/`, selectable via `PROMPT_SELECT` env var
