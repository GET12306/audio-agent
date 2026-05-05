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

Provide a transcription JSON (format: `{"full_text": str, "segments": [{"start": float, "end": float, "text": str}]}`) and run:

```bash
uv run python agent_core.py
```

This executes the full pipeline: calibrate → summarize → QA with a sample query.

To switch prompt profiles (e.g., for different radio formats), set `PROMPT_SELECT`:

```bash
PROMPT_SELECT=prompt_1 uv run python agent_core.py
```

## Project Structure

| File / Dir | Purpose |
|------------|---------|
| `agent_core.py` | Main pipeline (calibrate, summarize, Q&A) |
| `audio_tools.py` | LangChain tool for mlx-whisper transcription |
| `prompts/` | YAML files for LLM prompts (one per pipeline step) |


## Tech Stack

- **LLM**: DeepSeek v4 Flash via `langchain-deepseek` (or swap to Ollama locally)
- **Embeddings**: `mxbai-embed-large` via Ollama
- **Speech-to-text**: `mlx-whisper` (Apple Silicon)
- **Vector store**: FAISS (in-memory)
- **Prompt management**: YAML files in `prompts/`, selectable via `PROMPT_SELECT` env var
