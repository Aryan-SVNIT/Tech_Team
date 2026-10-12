from pathlib import Path
from PIL import Image
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
def extract_image_text(file_path=None):
    if file_path is None: file_path = input("Enter image path: ").strip().strip('"')
    path = Path(file_path)
    if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        print("Please provide an existing supported image file."); return None
    try:
        with Image.open(path) as image: text = pytesseract.image_to_string(image, lang="eng+hin").strip()
        if not text: print("No readable text found in image."); return None
        return text
    except Exception as error: print(f"Image extraction failed: {error}"); return None
if __name__ == "__main__":
    text = extract_image_text()
    if text: print("\n--- Extracted Image Text ---\n" + text)
