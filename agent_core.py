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


load_dotenv()

# use deepseek-v4-flash, API key is in .env file
llm = ChatDeepSeek(
    model="deepseek-v4-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# or local model
# llm = ChatOllama(model="gemma4:26b",)

# local model for embedding
embeddings = OllamaEmbeddings(model="mxbai-embed-large:latest")

PROMPT_DIR = Path(__file__).parent / "prompts"
def load_prompt(name: str) -> dict:
    prompt_temp = os.getenv("PROMPT_SELECT", "prompt_1")  # default to prompt_1
    with open(PROMPT_DIR / f"{name}.yaml") as f:
        prompts = yaml.safe_load(f)
    if prompt_temp not in prompts:
        print(f"Warning: prompt '{prompt_temp}' not found in {name}.yaml, falling back to 'prompt_1'")
        prompt_temp = "prompt_1"
    return prompts[prompt_temp]

# ==========================================
# 1. Calibration
# ==========================================
def calibrate_text(raw_data):
    # calibration_prompt = ChatPromptTemplate.from_messages([
    #     ("system", "你是一个专业的音频转录文本校对助手。你的任务是修正文本中的同音字错误、专有名词错误，并使语句通顺。\
    #         请保持原意，不要进行扩写。由于转录模型的能力有限，转录结果可能会出现无意义的重复语句和空白语句或符号，请去除这些语句和符号。\
    #         对于一些无法理解的句子，首先根据日语中具有相似发音的词进行猜测，同时注意上下文曾经提到的事物和逻辑关系，并尝试得出可能的原句。\
    #         请注意这些转录后的结果是日语，你需要结合日语汉字的读音和相关知识进行修正。\
    #         在这里提供参考的专有名词，はやしここ/林ココ/ハヤシココ对应的汉字是林鼓子。ココ在具有人名的含义的上下文环境里请修改为鼓子。\n\n\
    #         【输出格式要求】\n\
    #         输入数据包含完整文本(full_text)和带时间戳的片段列表(segments)。请结合 full_text 的全局上下文，对各个 segments 中的文字进行校对。\n\
    #         你必须返回一个严格的JSON数组，数组元素包含 start, end, 和 text（校对后的文本）。请不要输出任何 Markdown 标记（例如 ```json），只输出合法的 JSON 纯文本！\n\
    #         示例格式：\n\
    #         [\n  {{\"start\": 0.0, \"end\": 5.5, \"text\": \"校对后的句子1\"}},\n  {{\"start\": 5.5, \"end\": 12.0, \"text\": \"校对后的句子2\"}}\n]"),
    #     ("human", "请校对下面的转录文本：\n{text}")
    # ])

    cal = load_prompt("calibration")
    calibration_prompt = ChatPromptTemplate.from_messages([
        ("system", cal["system"]),
        ("human", cal["human"]),
    ])
    
    calibration_chain = calibration_prompt | llm | StrOutputParser()
    
    print("Starting global text calibration phase with timestamps...")
    
    # Convert the entire dictionary (including `full_text` and `segments`) into a JSON string 
    # so that the model can see both the global context and the timestamp-based segmentation
    input_text = json.dumps(raw_data, ensure_ascii=False, indent=2)
    
    corrected_text_raw = calibration_chain.invoke({"text": input_text})
    print("Calibration complete")
    
    # Parse the JSON string returned by the model
    try:
        cleaned_output = corrected_text_raw.strip()
        # Remove Markdown tags
        if cleaned_output.startswith("```json"):
            cleaned_output = cleaned_output[7:]
        elif cleaned_output.startswith("```"):
            cleaned_output = cleaned_output[3:]
        if cleaned_output.endswith("```"):
            cleaned_output = cleaned_output[:-3]
            
        calibrated_segments = json.loads(cleaned_output.strip())
        
        # Concatenate the corrected segments back into a full text
        calibrated_full_text = " ".join([seg["text"] for seg in calibrated_segments])
        
        return {
            "full_text": calibrated_full_text,
            "segments": calibrated_segments
        }
    except Exception as e:
        print(f"Failed to parse JSON output: {e}")
        print(f"Raw output from model:\n{corrected_text_raw}")
        # If parsing fails, return a fallback segment
        start_time = raw_data.get("segments", [{"start": 0.0}])[0]["start"]
        end_time = raw_data.get("segments", [{"end": 0.0}])[-1]["end"]
        fallback_segments = [{"start": start_time, "end": end_time, "text": corrected_text_raw}]
        return {
            "full_text": corrected_text_raw,
            "segments": fallback_segments
        }

# ==========================================
# 2. Summarization
# ==========================================
def generate_summary(calibrated_data):
    # 由于 calibrated_data 现在是一个包含 full_text 和 segments 的字典
    # 我们直接提取校对后的 full_text 即可，无需再手动遍历 segments
    full_text = calibrated_data.get("full_text", "")
    
    # summary_prompt = ChatPromptTemplate.from_messages([
    #     ("system", "你是一个善于归纳核心要点的助手。请将用户提供的长文本总结为结构清晰的要点。\
    #         请输出提供给你的文档中所有涉及到的内容，不要有任何遗漏。如果有相同内容出现过多次，可以并成一条。\
    #         请注意：提供给你的原文是日语，但请你统一整理为中文并输出。\
    #         但是对于一些词源是英语或者别的语言的日语，可以在输出结果中保留日语。"),
    #     ("human", "文本内容：\n{text}")
    # ])

    summary_data = load_prompt("summarization")
    summary_prompt = ChatPromptTemplate.from_messages([
        ("system", summary_data["system"]),
        ("human", summary_data["human"]),
    ])
    
    summary_chain = summary_prompt | llm | StrOutputParser()
    
    print("Generating summary...")

    summary = summary_chain.invoke({"text": full_text})
    return summary

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

if __name__ == "__main__":
    text_file_path = os.getenv('TRANSCRIPTION_JSON_PATH')
    
    if not text_file_path or not os.path.exists(text_file_path):
        print(f"Error: TRANSCRIPTION_JSON_PATH is not set or file does not exist: {text_file_path}")
        print("Please set TRANSCRIPTION_JSON_PATH in your .env file.")
        exit(1)
    
    # 1. calibrate the transcription
    corrected_transcription_path = text_file_path.replace('.json', '_calibrated.json')
    with open(text_file_path, "r", encoding="utf-8") as f:
        whisper_output = json.load(f)
    corrected_transcription = calibrate_text(whisper_output)

    with open(corrected_transcription_path, "w", encoding="utf-8") as f:
        json.dump(corrected_transcription, f, ensure_ascii=False, indent=2)

    # also save a copy to data/episodes/ for the knowledge base
    episodes_dir = Path("data/episodes")
    episodes_dir.mkdir(parents=True, exist_ok=True)
    episode_copy = episodes_dir / Path(corrected_transcription_path).name
    with open(episode_copy, "w", encoding="utf-8") as f:
        json.dump(corrected_transcription, f, ensure_ascii=False, indent=2)
    print(f"Copied to {episode_copy}")

    # 2. generate summary
    # with open(corrected_transcription_path, "r", encoding="utf-8") as f:
    #     corrected_transcription = json.load(f)
    final_summary = generate_summary(corrected_transcription)
    # write summary to md file
    summary_path = text_file_path.replace('.json', '_summary.md')
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(final_summary)
    
    # 3. load qa engine from knowledge base
    kb_path = os.getenv("KB_PATH", "data/kb")
    qa_engine = load_qa_engine_from_kb(kb_path)
    if qa_engine:
        query = "我想了解鼓子家的宠物应该从哪开始听?"
        print(f"User Query: {query}")
        answer = qa_engine.invoke(query)
        print("--- Agent Response: ---")
        print(answer)
    else:
        print("Skipping QA step.")
