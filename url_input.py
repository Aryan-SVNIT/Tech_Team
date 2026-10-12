import ipaddress
import re
import socket
from urllib.parse import urlparse

import requests
import trafilatura
from bs4 import BeautifulSoup


MAX_BYTES = 5 * 1024 * 1024
REQUEST_TIMEOUT = (5, 15)

HEADERS = {
    "User-Agent": "ForwardTrueAI/1.0 (article text extraction)"
}

# These headings commonly mark end matter rather than the article body.
END_MATTER_HEADINGS = {
    "references",
    "external links",
    "further reading",
    "explanatory notes",
    "notes",
}


def validate_url(url):
    """Basic URL validation. Public deployment needs network-level SSRF controls."""
    try:
        parsed = urlparse(url.strip())

        if parsed.scheme.lower() not in ("http", "https"):
            return False

        if not parsed.hostname:
            return False

        if parsed.username or parsed.password:
            return False

        hostname = parsed.hostname.rstrip(".").lower()

        if hostname == "localhost" or hostname.endswith(".localhost"):
            return False

        # Reject IP literals that are not globally routable.
        try:
            ip = ipaddress.ip_address(hostname)
            if not ip.is_global:
                return False
        except ValueError:
            # It is a hostname; resolve it and reject if ANY result is non-public.
            try:
                results = socket.getaddrinfo(
                    hostname,
                    parsed.port or (443 if parsed.scheme.lower() == "https" else 80),
                    type=socket.SOCK_STREAM,
                )
            except (socket.gaierror, OSError, ValueError):
                return False

            if not results:
                return False

            for result in results:
                try:
                    resolved_ip = ipaddress.ip_address(result[4][0])
                except ValueError:
                    return False

                if not resolved_ip.is_global:
                    return False

        # Only standard web ports are allowed.
        if parsed.port is not None and parsed.port not in (80, 443):
            return False

        return True

    except (ValueError, TypeError):
        return False


def _read_limited_response(response, max_bytes=MAX_BYTES):
    """Read at most max_bytes from a response, including when Content-Length is absent."""
    content_length = response.headers.get("Content-Length")

    if content_length:
        try:
            if int(content_length) > max_bytes:
                raise ValueError("Webpage is too large (limit: 5 MB).")
        except ValueError as error:
            if str(error).startswith("Webpage is too large"):
                raise
            # Invalid Content-Length: continue enforcing the streaming limit.

    chunks = []
    total_bytes = 0

    for chunk in response.iter_content(chunk_size=8192):
        if not chunk:
            continue

        total_bytes += len(chunk)
        if total_bytes > max_bytes:
            raise ValueError("Webpage is too large (limit: 5 MB).")

        chunks.append(chunk)

    return b"".join(chunks)


def _clean_article_text(article_text):
    if not article_text:
        return ""

    # Stop before end-matter headings that appear on their own line.
    lines = article_text.splitlines()
    kept_lines = []

    for line in lines:
        heading = line.strip().strip(" =:#*").casefold()

        if heading in {
            "see also",
            "references",
            "external links",
            "further reading",
            "explanatory notes",
            "notes",
        }:
            break

        kept_lines.append(line)

    article_text = "\n".join(kept_lines)

    # Remove inline citation markers such as [1] and [23].
    article_text = re.sub(r"\[(?:\d+|[a-z])\]", "", article_text)

    # Normalize whitespace.
    article_text = re.sub(r"[ \t]+\n", "\n", article_text)
    article_text = re.sub(r"\n{3,}", "\n\n", article_text)

    return article_text.strip()


def extract_article(url):
    """Download a webpage and extract its main article."""
    url = url.strip()

    if not validate_url(url):
        return None, None, "Invalid or unsafe URL."

    parsed = urlparse(url)
    if parsed.port is not None and parsed.port not in (80, 443):
        return None, None, "Only standard HTTP/HTTPS ports are allowed."

    try:
        # Redirects are deliberately disabled. Validate a redirect destination
        # separately before fetching it.
        with requests.get(
            url,
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT,
            allow_redirects=False,
            stream=True,
        ) as response:
            if response.is_redirect:
                location = response.headers.get("Location")
                if location:
                    return None, None, (
                        "This URL redirects. Please open the redirect destination, "
                        "check that it is trusted, and try that URL directly."
                    )
                return None, None, "This URL redirects but provides no destination."

            response.raise_for_status()

            content_type = response.headers.get("Content-Type", "").lower()
            if "text/html" not in content_type and "application/xhtml+xml" not in content_type:
                return None, None, "This URL does not return an HTML webpage."

            raw_html = _read_limited_response(response)

            # Requests may not know the encoding until response headers are read.
            encoding = response.encoding or "utf-8"
            html = raw_html.decode(encoding, errors="replace")

        soup = BeautifulSoup(html, "html.parser")

        title_tag = soup.find("title")
        title = (
            title_tag.get_text(" ", strip=True)
            if title_tag
            else "Untitled webpage"
        )

        # Prefer extracting the main article, not navigation and page chrome.
        article_text = trafilatura.extract(
            html,
            url=url,
            include_comments=False,
            include_tables=False,
            favor_precision=True,
            output_format="txt",
        )

        # Fallback when Trafilatura cannot identify an article.
        if not article_text or not article_text.strip():
            for tag in soup(
                [
                    "script", "style", "nav", "footer", "header",
                    "aside", "form", "noscript", "svg",
                ]
            ):
                tag.decompose()

            main = soup.find("article") or soup.find("main") or soup.body
            if main:
                article_text = main.get_text("\n", strip=True)

        article_text = _clean_article_text(article_text)

        if not article_text:
            return title, None, "No readable article text found on this webpage."

        return title, article_text, None

    except requests.Timeout:
        return None, None, "The webpage took too long to respond. Please try again."
    except requests.RequestException as error:
        return None, None, f"Could not fetch webpage: {error}"
    except ValueError as error:
        return None, None, str(error)
    except Exception as error:
        return None, None, f"Extraction failed: {error}"


def main():
    url = input("Enter article or webpage URL: ").strip()

    title, text, error = extract_article(url)

    if error:
        print(f"\nError: {error}")
        return

    print("\n--- Article Title ---")
    print(title)

    print("\n--- Source URL ---")
    print(url)

    print("\n--- Extracted Article Text ---")
    print(text)


if __name__ == "__main__":
    main()
