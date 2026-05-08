import os
import json
from pathlib import Path
from langchain.tools import tool
from langchain_community.vectorstores import FAISS
# from langgraph.prebuilt import create_react_agent
from langchain.agents import create_agent
from langchain_core.messages import AIMessage

from agent_core import (
    llm,
    embeddings,
    calibrate_text,
    generate_summary,
    _create_qa_chain,
    load_prompt,
)
from audio_tools import transcribe_audio_tool as _transcribe_tool
from build_kb import main as _rebuild_kb, KB_PATH, EPISODES_DIR


# ---- Tool: Transcribe ----

@tool
def transcribe_audio(audio_path: str, model_path: str = "mlx-community/whisper-large-v3-turbo") -> str:
    """Transcribe an audio file into timestamped text. Saves result as JSON alongside the audio file. Returns the path to the saved JSON."""
    result = _transcribe_tool.invoke({"audio_path": audio_path, "model_path": model_path})
    out_path = Path(audio_path).with_suffix(".json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return str(out_path)


# ---- Tool: Calibrate ----

@tool
def calibrate_transcription(raw_json_path: str) -> str:
    """Calibrate a raw transcription JSON: fix homophone errors, remove duplicates, preserve timestamps. Saves the calibrated JSON alongside the original and copies it to data/episodes/. Returns the path to the calibrated JSON."""
    with open(raw_json_path, encoding="utf-8") as f:
        raw_data = json.load(f)
    corrected = calibrate_text(raw_data)

    out_path = raw_json_path.replace(".json", "_calibrated.json")
    # with open(out_path, "w", encoding="utf-8") as f:
    #     json.dump(corrected, f, ensure_ascii=False, indent=2)

    # save to data/episodes/ for the knowledge base
    ep_path = EPISODES_DIR / Path(out_path).name
    ep_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ep_path, "w", encoding="utf-8") as f:
        json.dump(corrected, f, ensure_ascii=False, indent=2)

    return out_path


# ---- Tool: Summarize ----

@tool
def summarize_episode(calibrated_json_path: str) -> str:
    """Generate a Chinese bullet-point summary from a calibrated transcription. Returns the path to the saved summary markdown."""
    with open(calibrated_json_path, encoding="utf-8") as f:
        data = json.load(f)
    summary = generate_summary(data)

    out_path = calibrated_json_path.replace("_calibrated.json", "_summary.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(summary)
    return out_path


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
        # for msg in result["messages"]:
        #     if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
        #         print(msg.content)

        for msg in result["messages"]:
            # message type: HumanMessage, AIMessage, ToolMessage
            print(f"message type: [{msg.__class__.__name__}]")
            
            # if this message contains tool calls, print them
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tool in msg.tool_calls:
                    print(f"prepare to use tool: {tool['name']}")
                    print(f"args: {tool['args']}")
                    
            # print message content
            if msg.content:
                # to avoid  overflow
                preview = msg.content
                print(f"message content: {preview}")
            
            print("-" * 40)
