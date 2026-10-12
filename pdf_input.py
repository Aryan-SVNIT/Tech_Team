from pathlib import Path
import pymupdf
def extract_pdf_text(file_path=None):
    if file_path is None: file_path = input("Enter PDF file path: ").strip().strip('"')
    path = Path(file_path)
    if not path.is_file() or path.suffix.lower() != ".pdf": print("Please provide a valid PDF file."); return None
    try:
        with pymupdf.open(path) as pdf: text = "\n".join(page.get_text() for page in pdf).strip()
        if not text: print("No selectable text found. This PDF may need OCR."); return None
        return text
    except Exception as error: print(f"PDF extraction failed: {error}"); return None
if __name__ == "__main__":
    text = extract_pdf_text()
    if text: print("\n--- Extracted PDF Text ---\n" + text)
