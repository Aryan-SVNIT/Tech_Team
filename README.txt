ForwardTrue AI — Common Input + Fact-checking Pipeline

SETUP (use your existing Conda environment)
1. Keep .env in this folder:
   GEMINI_API_KEY=your_actual_key
   TAVILY_API_KEY=your_actual_key
   GEMINI_MODEL=gemini-3.5-flash-lite
2. Install dependencies: python -m pip install -r requirements.txt
3. Run: python main.py

main.py connects text, image OCR, PDF, URL, and Whisper audio input to a common
fact-checking flow. Reports are saved in reports/ as JSON and Markdown.

Image OCR expects Tesseract at C:\Program Files\Tesseract-OCR\tesseract.exe
and Hindi language data for eng+hin. Whisper requires FFmpeg on PATH.
Never share or commit .env.

Limitations: this is an AI-assisted preliminary assessment, not authoritative truth.
Tavily search snippets may be incomplete; open sources to verify them. Only citation
IDs returned by Tavily are accepted, and no valid citation forces Insufficient Evidence.
API calls can consume credits. The URL fetcher uses basic SSRF protections, but public
hosting requires stronger network egress restrictions and DNS/redirect controls.
