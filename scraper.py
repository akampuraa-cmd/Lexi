"""
scraper.py — Lexi Web Crawling & Scraping Module

Uses *requests* to fetch HTML pages and *BeautifulSoup* to extract
visible text.  The extracted text is then passed through the
DataCleaner to produce a training-ready string.

Design notes
------------
* Only the **visible** text is kept — scripts, styles, and HTML tags
  are discarded by BeautifulSoup's ``get_text()`` method.
* A generous but finite timeout (15 s) prevents the GUI from hanging
  on unresponsive servers.
* The module is intentionally synchronous — the GUI wraps each call
  in a ``QThread`` so the event loop stays responsive.
"""

from __future__ import annotations

import requests
from bs4 import BeautifulSoup

from data_cleaner import DataCleaner

# Default HTTP headers to identify ourselves politely.
_HEADERS = {
    "User-Agent": (
        "Lexi-AI-Scraper/1.0 "
        "(+https://github.com/akampuraa-cmd/Lexi)"
    ),
}

_TIMEOUT = 15  # seconds


def scrape_url(url: str) -> str:
    """Fetch *url*, extract visible text, clean it, and return the result.

    Parameters
    ----------
    url : str
        The full URL (including scheme) to scrape.

    Returns
    -------
    str
        Cleaned text extracted from the page.

    Raises
    ------
    requests.RequestException
        On any network or HTTP error.
    """
    response = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    # Remove <script> and <style> elements before extracting text.
    for tag in soup(["script", "style"]):
        tag.decompose()

    raw_text = soup.get_text(separator=" ")
    return DataCleaner.clean(raw_text)
