import os
import json
from pathlib import Path
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

load_dotenv()

KB_PATH = Path(os.getenv("KB_PATH", "data/kb"))
EPISODES_DIR = Path(os.getenv("EPISODES_DIR", "data/episodes"))
CHUNK_SIZE = 5
OVERLAP = 2
STEP = CHUNK_SIZE - OVERLAP

embeddings = OllamaEmbeddings(model="mxbai-embed-large:latest")


def chunk_episode(episode_path: Path) -> list[Document]:
    with open(episode_path, encoding="utf-8") as f:
        data = json.load(f)

    segments = data["segments"]
    episode_name = episode_path.stem.replace("_calibrated", "")
    docs = []

    for i in range(0, len(segments), STEP):
        chunk = segments[i : i + CHUNK_SIZE]
        combined_text = " ".join([seg["text"] for seg in chunk])
        docs.append(
            Document(
                page_content=combined_text,
                metadata={
                    "start_time": chunk[0]["start"],
                    "end_time": chunk[-1]["end"],
                    "episode": episode_name,
                    "source_file": episode_path.name,
                },
            )
        )
    return docs


def main():
    KB_PATH.mkdir(parents=True, exist_ok=True)

    # find all calibrated files
    episode_files = sorted(EPISODES_DIR.glob("*_calibrated.json"))
    if not episode_files:
        print(f"No calibrated files found in {EPISODES_DIR}/")
        print("Run agent_core.py first, then copy the _calibrated.json files here.")
        return

    all_docs = []
    for f in episode_files:
        print(f"  Processing {f.name}...")
        all_docs.extend(chunk_episode(f))

    print(f"Total chunks: {len(all_docs)}")
    print("Building FAISS index...")
    vectorstore = FAISS.from_documents(all_docs, embeddings)
    vectorstore.save_local(str(KB_PATH))
    print(f"Saved to {KB_PATH}/")


if __name__ == "__main__":
    main()
