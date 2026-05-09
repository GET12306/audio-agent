import os
import json
from pathlib import Path
from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
import yaml
import mlx_whisper


load_dotenv()

# use deepseek-v4-flash, API key is in .env file
llm = ChatDeepSeek(
    model="deepseek-v4-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    extra_body={
        "thinking": {"type": "disabled"},
    }
)

# or local model
# llm = ChatOllama(model="gemma4:26b",)

# local model for embedding
embeddings = OllamaEmbeddings(model="mxbai-embed-large:latest")

PROMPT_DIR = Path(__file__).parent / "prompts"
def load_prompt(name: str) -> dict:
    prompt_temp = os.getenv("PROMPT_SELECT", "prompt_1")
    with open(PROMPT_DIR / f"{name}.yaml") as f:
        prompts = yaml.safe_load(f)
    if prompt_temp not in prompts:
        print(f"Warning: prompt '{prompt_temp}' not found in {name}.yaml, falling back to 'prompt_1'")
        prompt_temp = "prompt_1"
    return prompts[prompt_temp]


# ==========================================
# 0. Transcription
# ==========================================
def transcribe_audio(audio_path: str, model_path: str) -> dict:
    """Transcribe a local audio file into timestamped text using mlx-whisper."""
    print(f"Transcribing audio... {audio_path}")
    try:
        result = mlx_whisper.transcribe(
            audio_path,
            path_or_hf_repo=model_path,
            initial_prompt='はやしここ对应的汉字是林鼓子'
        )
        segments = [
            {"start": seg["start"], "end": seg["end"], "text": seg["text"]}
            for seg in result.get("segments", [])
        ]
        return {
            "full_text": result.get("text", "").strip(),
            "segments": segments,
        }
    except Exception as e:
        error_msg = f"Error transcribing audio: {str(e)}"
        print(error_msg)
        return {
            "full_text": error_msg,
            "segments": [{"start": 0.0, "end": 0.0, "text": error_msg}],
        }


# ==========================================
# 1. Calibration
# ==========================================
def calibrate_text(raw_data):
    cal = load_prompt("calibration")
    calibration_prompt = ChatPromptTemplate.from_messages([
        ("system", cal["system"]),
        ("human", cal["human"]),
    ])
    calibration_chain = calibration_prompt | llm | StrOutputParser()

    print("Starting global text calibration phase with timestamps...")
    input_text = json.dumps(raw_data, ensure_ascii=False, indent=2)
    corrected_text_raw = calibration_chain.invoke({"text": input_text})
    print("Calibration complete")

    try:
        cleaned_output = corrected_text_raw.strip()
        if cleaned_output.startswith("```json"):
            cleaned_output = cleaned_output[7:]
        elif cleaned_output.startswith("```"):
            cleaned_output = cleaned_output[3:]
        if cleaned_output.endswith("```"):
            cleaned_output = cleaned_output[:-3]
        calibrated_segments = json.loads(cleaned_output.strip())
        calibrated_full_text = " ".join([seg["text"] for seg in calibrated_segments])
        return {"full_text": calibrated_full_text, "segments": calibrated_segments}
    except Exception as e:
        print(f"Failed to parse JSON output: {e}")
        print(f"Raw output from model:\n{corrected_text_raw}")
        start_time = raw_data.get("segments", [{"start": 0.0}])[0]["start"]
        end_time = raw_data.get("segments", [{"end": 0.0}])[-1]["end"]
        return {
            "full_text": corrected_text_raw,
            "segments": [{"start": start_time, "end": end_time, "text": corrected_text_raw}],
        }


# ==========================================
# 2. Summarization
# ==========================================
def generate_summary(calibrated_data):
    full_text = calibrated_data.get("full_text", "")
    summary_data = load_prompt("summarization")
    summary_prompt = ChatPromptTemplate.from_messages([
        ("system", summary_data["system"]),
        ("human", summary_data["human"]),
    ])
    summary_chain = summary_prompt | llm | StrOutputParser()
    print("Generating summary...")
    return summary_chain.invoke({"text": full_text})


# ==========================================
# 3. Timestamp-Aware RAG
# ==========================================
def _create_qa_chain(retriever):
    qa = load_prompt("qa")
    qa_prompt = ChatPromptTemplate.from_template(qa["template"])

    def format_docs(retrieved_docs):
        return "\n\n".join(
            f"[{doc.metadata.get('episode', '?')} | {doc.metadata['start_time']}s - {doc.metadata['end_time']}s] Content: {doc.page_content}"
            for doc in retrieved_docs
        )

    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | qa_prompt
        | llm
        | StrOutputParser()
    )


def build_qa_engine(calibrated_data):
    segments = calibrated_data["segments"]
    docs = []
    chunk_size = 5
    overlap = 2
    step = chunk_size - overlap

    for i in range(0, len(segments), step):
        chunk = segments[i : i + chunk_size]
        combined_text = " ".join([seg["text"] for seg in chunk])
        docs.append(
            Document(
                page_content=combined_text,
                metadata={
                    "start_time": chunk[0]["start"],
                    "end_time": chunk[-1]["end"],
                }
            )
        )

    print("\nBuilding in-memory vector store for QA...")
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    return _create_qa_chain(retriever)


def load_qa_engine_from_kb(kb_path="data/kb"):
    if not Path(kb_path).exists():
        print(f"Knowledge base not found at {kb_path}/")
        print("Run `uv run build_kb.py` to create one.")
        return None

    print(f"\nLoading knowledge base from {kb_path}/...")
    vectorstore = FAISS.load_local(kb_path, embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    return _create_qa_chain(retriever)
