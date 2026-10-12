
import os
from dotenv import load_dotenv
from google import genai
from tavily import TavilyClient

load_dotenv()

gemini_key = os.getenv("GEMINI_API_KEY")
tavily_key = os.getenv("TAVILY_API_KEY")

# Test Gemini
try:
    if not gemini_key:
        raise ValueError("GEMINI_API_KEY is missing")

    client = genai.Client(api_key=gemini_key)
    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents="Reply with exactly: Gemini connection successful."
    )
    print("Gemini:", response.text.strip())

except Exception as e:
    print("Gemini connection failed:", type(e).__name__, str(e)[:400])

# Test Tavily
try:
    if not tavily_key:
        raise ValueError("TAVILY_API_KEY is missing")

    client = TavilyClient(api_key=tavily_key)
    result = client.search(
        query="What is the official website of the World Health Organization?",
        max_results=1
    )

    if result.get("results"):
        item = result["results"][0]
        print("Tavily: Connection successful")
        print("Sample result:", item.get("title", "Untitled"))
        print("Source URL:", item.get("url", "Unavailable"))
    else:
        print("Tavily connected, but no search results returned.")

except Exception as e:
    print("Tavily connection failed:", type(e).__name__, str(e)[:400])
