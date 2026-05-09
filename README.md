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

## Project Structure

| File / Dir | Purpose |
|------------|---------|
| `agent.py` | **Entry point.** ReAct agent with all tools + interactive loop |
| `engine.py` | Core logic (transcribe, calibrate, summarize, QA chain) — pure functions |
| `build_kb.py` | Build persistent FAISS knowledge base from `data/episodes/` |
| `prompts/` | YAML files for LLM prompts (one per pipeline step) |
| `data/episodes/` | Calibrated JSON files (input for knowledge base) |
| `data/kb/` | Persisted FAISS index (loaded for QA) |


## Tech Stack

- **LLM**: DeepSeek v4 Flash via `langchain-deepseek` (thinking disabled via `extra_body`; swap to `ChatOllama` for local inference)
- **Embeddings**: `mxbai-embed-large` via Ollama
- **Speech-to-text**: `mlx-whisper` (Apple Silicon)
- **Vector store**: FAISS (persisted to disk via `build_kb.py`)
- **Orchestration**: LangGraph ReAct agent (`create_react_agent`)
- **Prompt management**: YAML files in `prompts/`, selectable via `PROMPT_SELECT` env var
