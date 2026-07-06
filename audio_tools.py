from langchain_core.tools import tool
import mlx_whisper

@tool
def transcribe_audio_tool(audio_path: str, model_path: str) -> dict:
    """
    A tool for transcribing local audio files into timestamped text.
    Accelerated by mlx-whisper.
    
    Args:
        audio_path (str): Path to the local audio file.
        model_path (str): HuggingFace model name to use.

    Returns:
        dict: A dictionary containing the overall text and segmented text.
        {
            "full_text": "The complete transcribed text...",
            "segments": [
                {"start": 0.0, "end": 5.0, "text": "The first sentence"},
                ...
            ]
        }
    """
    print(f"[Tool] Transcribing audio... Preparing to transcribe file: {audio_path}")

    try:
        result = mlx_whisper.transcribe(
            audio_path,
            path_or_hf_repo=model_path,
            initial_prompt='はやしここ对应的汉字是林鼓子'
        )
        
        segments = []
        for segment in result.get("segments", []):
            start = segment["start"]
            end = segment["end"]
            text = segment["text"]
            
            segments.append({
                "start": start,
                "end": end,
                "text": text
            })
        
        return {
            "full_text": result.get("text", "").strip(),
            "segments": segments
        }

    except Exception as e:
        error_msg = f"Error transcribing audio: {str(e)}"
        print(f"[Tool] {error_msg}")
        return {
            "full_text": error_msg,
            "segments": [{"start": 0.0, "end": 0.0, "text": error_msg}]
        }


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    test_file = './test.mp4'
    print(transcribe_audio_tool.invoke({"audio_path": test_file, "model_path": 'mlx-community/whisper-large-v3-turbo'}))
