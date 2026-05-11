import os
import json
from pathlib import Path
from langchain.tools import tool
from langchain_community.vectorstores import FAISS
from langchain.agents import create_agent

from engine import (
    llm,
    embeddings,
    transcribe_audio as _transcribe_audio,
    calibrate_text,
    generate_summary,
    _create_qa_chain,
    load_prompt,
)
from build_kb import main as _rebuild_kb, EPISODES_DIR


# ---- Tool: Transcribe ----

@tool
def transcribe_audio(audio_path: str, model_path: str = "mlx-community/whisper-large-v3-turbo") -> str:
    """Transcribe an audio file into timestamped text. Saves result as JSON alongside the audio file. Returns the path to the saved JSON."""
    result = _transcribe_audio(audio_path, model_path)
    out_path = Path(audio_path).with_suffix(".json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return str(out_path)


# ---- Tool: Calibrate ----

@tool
def calibrate_transcription(raw_json_path: str) -> str:
    """Calibrate a raw transcription JSON: fix homophone errors, remove duplicates, preserve timestamps. Saves the calibrated JSON to data/episodes/. Returns the path to the calibrated JSON."""
    with open(raw_json_path, encoding="utf-8") as f:
        raw_data = json.load(f)
    corrected = calibrate_text(raw_data)

    out_path = raw_json_path.replace(".json", "_calibrated.json")
    ep_path = EPISODES_DIR / Path(out_path).name
    ep_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ep_path, "w", encoding="utf-8") as f:
        json.dump(corrected, f, ensure_ascii=False, indent=2)

    return out_path


# ---- Tool: Summarize ----

@tool
def summarize_episode(calibrated_json_path: str) -> str:
    """Generate or retrieve a Chinese bullet-point summary from a calibrated transcription. Returns the summary content directly."""
    out_path = calibrated_json_path.replace("_calibrated.json", "_summary.md")
    if Path(out_path).exists():
        with open(out_path, encoding="utf-8") as f:
            return f.read()

    with open(calibrated_json_path, encoding="utf-8") as f:
        data = json.load(f)
    summary = generate_summary(data)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(summary)
    return summary


# ---- Tool: Search KB ----

@tool
def search_knowledge_base(query: str, episode: str = "") -> str:
    """Search the persisted knowledge base for information. Returns an answer with timestamps and episode references. Optionally filter by episode name."""
    kb_path = os.getenv("KB_PATH", "data/kb")
    if not Path(kb_path).exists():
        return "Knowledge base not found. Run rebuild_knowledge_base first."

    vectorstore = FAISS.load_local(kb_path, embeddings, allow_dangerous_deserialization=True)
    search_kwargs = {"k": 3}
    if episode:
        search_kwargs["filter"] = {"episode": episode}
    retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)

    qa_chain = _create_qa_chain(retriever)
    return qa_chain.invoke(query)


# ---- Tool: Rebuild KB ----

@tool
def rebuild_knowledge_base() -> str:
    """Rebuild the FAISS knowledge base index from all calibrated files in data/episodes/."""
    _rebuild_kb()
    return "Knowledge base rebuilt."


# ---- Agent + Main ----

if __name__ == "__main__":
    tools = [
        transcribe_audio,
        calibrate_transcription,
        summarize_episode,
        search_knowledge_base,
        rebuild_knowledge_base,
    ]

    prompt_data = load_prompt("agent")
    agent = create_agent(llm, tools, system_prompt=prompt_data["system"])

    print("Audio Agent ready. Ask me to transcribe, calibrate, summarize, search, or rebuild kb.")
    while True:
        user_input = input("\n> ")
        if user_input.lower() in ("quit", "exit", "q"):
            break

        result = agent.invoke({"messages": [("human", user_input)]})
        for msg in result["messages"]:
            # 1. 检查并打印深度思考过程 (Reasoning Content)
            if hasattr(msg, 'additional_kwargs'):
                reasoning = msg.additional_kwargs.get("reasoning_content")
                if reasoning:
                    print("\n💭 [Thinking...]")
                    print(f"{reasoning}")
                    print("-" * 20)
            # 2. 检查并打印工具调用情况
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    print(f"\n🛠️ [using tool: {tool_call['name']}]")
            # 3. 打印最终回复内容
            if msg.content:
                # 如果有内容，直接打印
                print(f"\n{msg.content}")
            
            print("\n" + "=" * 40)
