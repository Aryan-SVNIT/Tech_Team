from pathlib import Path
from datetime import datetime
import json
from text_input import take_text_input
from image_input import extract_image_text
from pdf_input import extract_pdf_text
from url_input import extract_article
from audio_input import transcribe_audio
from fact_checker import fact_check, save_report
from text_normalizer import normalize_extracted_text

BASE_DIR = Path(__file__).resolve().parent
REPORT_DIR = BASE_DIR / "reports"

def acquire_input():
    print("\n========== ForwardTrue AI ==========")
    print("1. Paste text\n2. Extract text from image (OCR)\n3. Extract text from PDF\n4. Extract article from URL\n5. Transcribe audio / song\n0. Exit")
    choice = input("\nChoose input type: ").strip()
    meta = {"input_type": None, "source": None, "title": None}
    text = None
    if choice == "1":
        meta["input_type"] = "text"; text = take_text_input()
    elif choice == "2":
        meta["input_type"] = "image"; path = input("Enter image path: ").strip().strip('"'); meta["source"] = path; text = extract_image_text(path)
    elif choice == "3":
        meta["input_type"] = "pdf"; path = input("Enter PDF file path: ").strip().strip('"'); meta["source"] = path; text = extract_pdf_text(path)
    elif choice == "4":
        meta["input_type"] = "url"; url = input("Enter article or webpage URL: ").strip(); meta["source"] = url
        title, text, error = extract_article(url); meta["title"] = title
        if error:
            print("URL extraction error:", error); return None, None
        print("Extracted webpage:", title)
    elif choice == "5":
        meta["input_type"] = "audio"; path = input("Enter audio file path: ").strip().strip('"'); meta["source"] = path
        language = input("Audio language (1=Hindi, 2=English, 3=Auto): ").strip(); text = transcribe_audio(path, language)
    elif choice == "0": return "EXIT", None
    else:
        print("Invalid choice."); return None, None
    if not text or not text.strip():
        print("No text was extracted. Try another input."); return None, None
    return text.strip(), meta

def save_extraction(text, metadata):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = REPORT_DIR / f"extraction_{stamp}.json"
    payload = {"project": "ForwardTrue AI", "created_at": datetime.now().astimezone().isoformat(),
               "status": "extracted_not_fact_checked", "metadata": metadata, "extracted_text": text}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path

def main():
    print("ForwardTrue AI — Evidence-first fact-checking pipeline")
    while True:
        text, metadata = acquire_input()
        if text == "EXIT": print("Goodbye!"); break
        if text is None:
            continue

        text = normalize_extracted_text(text)

        if not text:
            print("No text remains after normalization.")
            continue

        print("\n========== EXTRACTED TEXT PREVIEW ==========")
        print("Input type:", metadata["input_type"], "| Characters:", len(text))
        print(text[:1200] + ("\n...[preview truncated]" if len(text) > 1200 else ""))
        print("\nExtraction saved:", save_extraction(text, metadata))
        if input("\nRun AI fact-check now? (y/n): ").strip().lower() not in {"y", "yes"}:
            print("Saved extracted text only."); continue
        try:
            print("\nExtracting claims and searching for evidence...")
            report = fact_check(text, metadata)

            # Preserve where the original text came from.
            report["input_metadata"] = metadata
            report["provenance"] = {
                "input_type": metadata.get("input_type"),
                "original_source": metadata.get("source"),
                "original_title": metadata.get("title"),
                "extracted_text_characters": len(text),
                "note": (
                    "This identifies the original input source. "
                    "It is not independent corroboration by itself."
                ),
            }

            if metadata.get("input_type") == "url":
                report["provenance"]["source_role"] = (
                    "Original webpage from which the text was extracted"
                )
            else:
                report["provenance"]["source_role"] = (
                    "Original user-provided input"
                )

            jp, mp = save_report(report, REPORT_DIR)
            try:
                from supabase_store import save_report_to_supabase

                db_report_id = save_report_to_supabase(
                    report,
                    text,
                    metadata,
                )
                print("\nSupabase report ID:", db_report_id)

            except Exception as db_exc:
                print(
                    "\nSupabase save failed; "
                    "local reports are still saved."
                )
                print(
                    f"Database error: {type(db_exc).__name__}: {db_exc}"
                )
            print("\n========== FACT-CHECK SUMMARY ==========")
            if not report["claims"]: print("No checkable factual claims found.")
            for i, item in enumerate(report["claims"], 1):
                print(f"\n{i}. {item['claim']}\n   Verdict: {item['verdict']} | Confidence: {item['confidence']}\n   {item['explanation']}")
                for source in item.get("sources", []): print(f"   Source: {source['title']} — {source['url']}")
            print("\nJSON report:", jp, "\nReadable report:", mp)
        except Exception as exc:
            print(f"\nFact-check could not be completed: {type(exc).__name__}: {exc}")
            print("Extracted text remains saved. Check API quota, model access, and .env.")

if __name__ == "__main__": main()
