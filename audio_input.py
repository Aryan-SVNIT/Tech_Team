from pathlib import Path
import whisper
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".wma", ".mp4", ".mpeg", ".mpga", ".webm"}
def transcribe_audio(audio_path=None, language_choice=None, model=None):
    if audio_path is None: audio_path = input("Enter audio file path: ")
    path = Path(str(audio_path).strip().strip('"'))
    if not path.is_file() or path.suffix.lower() not in AUDIO_EXTENSIONS: print("Please provide an existing supported audio file."); return None
    if language_choice is None:
        print("\nChoose audio language:\n1. Hindi song / Hindi speech\n2. English\n3. Auto-detect")
        language_choice = input("Enter choice (1/2/3): ").strip()
    languages = {"1": "hi", "2": "en", "3": None}
    if language_choice not in languages: print("Invalid language choice."); return None
    try:
        if model is None: print("Loading Whisper base model..."); model = whisper.load_model("base")
        kwargs = {"task": "transcribe", "fp16": False, "verbose": False, "condition_on_previous_text": False}
        language = languages[language_choice]
        if language: kwargs["language"] = language
        if language == "hi": kwargs["initial_prompt"] = "यह एक हिंदी गीत या हिंदी भाषण है। गीत के बोल हिंदी या अंग्रेज़ी में हो सकते हैं।"
        print("Transcribing audio... please wait.")
        result = model.transcribe(str(path), **kwargs); text = result.get("text", "").strip()
        if not text: print("No speech or lyrics could be recognized."); return None
        out = path.with_name(path.stem + "_lyrics.txt")
        with out.open("w", encoding="utf-8") as f:
            f.write("--- TRANSCRIPTION ---\n" + text + "\n\n--- TIMESTAMPS ---\n")
            for seg in result.get("segments", []): f.write(f"[{seg['start']:.2f}s - {seg['end']:.2f}s] {seg['text'].strip()}\n")
        print("Transcription saved to:", out); return text
    except Exception as error: print(f"Audio transcription failed: {error}"); return None
if __name__ == "__main__": transcribe_audio()
