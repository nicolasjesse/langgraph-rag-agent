"""Scrape LangChain Python docs into corpus/ as markdown.

Strategy:
1. Fetch llms.txt — a curated index of all docs URLs (community standard
   for AI-readable site maps).
2. Filter to URLs under /oss/python/ (LangChain + LangGraph Python docs).
3. Download each as raw markdown (Mintlify serves .md natively when you
   append .md to a docs URL — no HTML scraping needed).

Run once locally; the corpus/ folder is gitignored.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import httpx

CORPUS_DIR = Path(__file__).parent / "corpus"
LLMS_TXT_URL = "https://python.langchain.com/llms.txt"
URL_PREFIX_FILTER = "https://docs.langchain.com/oss/python/"


def parse_llms_txt(text: str) -> list[str]:
    """Extract all .md URLs from llms.txt and filter to our prefix."""
    urls = re.findall(r"\]\((https://docs\.langchain\.com/[^)]+\.md)\)", text)
    return [u for u in urls if u.startswith(URL_PREFIX_FILTER)]


def slugify(url: str) -> str:
    """Turn a URL into a safe filename."""
    path = url.replace(URL_PREFIX_FILTER, "").removesuffix(".md")
    return re.sub(r"[^a-zA-Z0-9]+", "_", path) + ".md"


def fetch_one(client: httpx.Client, url: str) -> tuple[str, bool, str]:
    filename = slugify(url)
    try:
        resp = client.get(url, follow_redirects=True, timeout=20)
        resp.raise_for_status()
    except httpx.HTTPError as e:
        return filename, False, f"HTTP error: {e}"

    md = resp.text.strip()
    if len(md) < 200:
        return filename, False, "content too short"
    if md.lstrip().startswith(("<!DOCTYPE", "<html")):
        return filename, False, "HTML returned (auto-generated page, no .md source)"

    (CORPUS_DIR / filename).write_text(md, encoding="utf-8")
    return filename, True, f"{len(md):,} chars"


def main() -> int:
    CORPUS_DIR.mkdir(exist_ok=True)

    print(f"Fetching index: {LLMS_TXT_URL}")
    with httpx.Client(headers={"User-Agent": "langgraph-rag-agent/0.1"}) as client:
        index = client.get(LLMS_TXT_URL, follow_redirects=True, timeout=20)
        index.raise_for_status()
        urls = parse_llms_txt(index.text)
        print(f"Found {len(urls)} URLs under {URL_PREFIX_FILTER}\n")

        ok_count = 0
        for url in urls:
            filename, ok, msg = fetch_one(client, url)
            mark = "OK " if ok else "ERR"
            print(f"  [{mark}] {filename:<60} {msg}")
            if ok:
                ok_count += 1

    print(f"\nDone: {ok_count}/{len(urls)} pages saved to {CORPUS_DIR}/")
    return 0 if ok_count > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
