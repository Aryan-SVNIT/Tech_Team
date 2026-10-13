"""Conservative text normalization for ForwardTrue AI."""

import re
import unicodedata


# Only repair spacing errors observed in our PDF test.
# Add more corrections when real extraction tests reveal them.
KNOWN_HINDI_REPAIRS = {
    "सेल्सि यस": "सेल्सियस",
    "मस्ति ष्क": "मस्तिष्क",
}


def normalize_extracted_text(text: str) -> str:
    """Normalize Unicode and whitespace without merging arbitrary words."""
    if not isinstance(text, str):
        return ""

    # Normalize Unicode representations.
    text = unicodedata.normalize("NFC", text)

    # Remove invisible formatting characters that can disrupt extraction.
    text = text.replace("\ufeff", "")
    text = text.replace("\u200b", "")
    text = text.replace("\u00ad", "")

    # Normalize line endings and horizontal whitespace.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"[ \t]+([,.;:!?।])", r"\1", text)

    # Apply only explicitly known Hindi word repairs.
    for broken, corrected in KNOWN_HINDI_REPAIRS.items():
        text = text.replace(broken, corrected)

    # Remove leading/trailing whitespace while preserving paragraphs.
    text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()