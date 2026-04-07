"""
data_cleaner.py — Lexi Data Cleaning System

This module provides aggressive text preprocessing to prepare raw text
for tokenisation and training. The cleaning pipeline:

1. Replaces all carriage-return / line-feed sequences with a single space.
2. Collapses multiple consecutive whitespace characters into one space.
3. Strips leading / trailing whitespace from every logical line.
4. Removes blank lines entirely.
5. Returns a single continuous, clean string of text.
"""

from __future__ import annotations

import re


class DataCleaner:
    """Stateless text-cleaning utility used by both the file-loader and web-scraper."""

    # Pre-compiled patterns (compiled once, reused on every call)
    _LINE_BREAK = re.compile(r"[\r\n]+")
    _MULTI_SPACE = re.compile(r"[ \t]+")

    @classmethod
    def clean(cls, raw_text: str) -> str:
        """Return a cleaned version of *raw_text*.

        The result is a single string with no blank lines, no stray
        line-breaks, and no leading/trailing whitespace.

        Parameters
        ----------
        raw_text : str
            The raw, unprocessed text (e.g. from a file or web page).

        Returns
        -------
        str
            Cleaned text ready for tokenisation.
        """
        if not raw_text:
            return ""

        # Step 1 – Replace every line-break with a space so that
        # paragraphs are merged into a continuous stream.
        text = cls._LINE_BREAK.sub(" ", raw_text)

        # Step 2 – Collapse runs of spaces / tabs into a single space.
        text = cls._MULTI_SPACE.sub(" ", text)

        # Step 3 – Strip leading and trailing whitespace.
        text = text.strip()

        return text
