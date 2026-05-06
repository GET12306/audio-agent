# Audio Agent

An AI-powered audio processing and Q&A system for Japanese radio content. Built on LangChain with Apple Silicon acceleration.

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
#   TRANSCRIPTION_JSON_PATH — path to a Whisper-transcribed JSON file
```

## Usage

### Step 1: Process a radio episode

Provide a transcription JSON (format: `{"full_text": str, "segments": [{"start": float, "end": float, "text": str}]}`) and run:

```bash
uv run agent_core.py
```

This calibrates the transcription, generates a summary, and saves a copy to `data/episodes/` for the knowledge base.

To switch prompt profiles (e.g., for different radio formats), set `PROMPT_SELECT`:

```bash
PROMPT_SELECT=prompt_2 uv run agent_core.py
```

### Step 2: Build the knowledge base

After processing one or more episodes, build the persistent FAISS index:

```bash
uv run build_kb.py
```

This creates `data/kb/` with a searchable index of all episodes.

### Step 3: Ask questions

The QA step in `agent_core.py` loads from the persistent knowledge base, so you can ask about any processed episode without rebuilding the index every time.

## Project Structure

| File / Dir | Purpose |
|------------|---------|
| `agent_core.py` | Main pipeline (calibrate, summarize, Q&A) |
| `build_kb.py` | Build persistent FAISS knowledge base from `data/episodes/` |
| `audio_tools.py` | LangChain tool for mlx-whisper transcription |
| `prompts/` | YAML files for LLM prompts (one per pipeline step) |
| `data/episodes/` | Calibrated JSON files (input for knowledge base) |
| `data/kb/` | Persisted FAISS index (loaded for QA) |


## Tech Stack

- **LLM**: DeepSeek v4 Flash via `langchain-deepseek` (or swap to Ollama locally)
- **Embeddings**: `mxbai-embed-large` via Ollama
- **Speech-to-text**: `mlx-whisper` (Apple Silicon)
- **Vector store**: FAISS (persisted to disk via `build_kb.py`)
- **Prompt management**: YAML files in `prompts/`, selectable via `PROMPT_SELECT` env var
