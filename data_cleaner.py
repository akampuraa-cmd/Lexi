"""
data_cleaner.py — DataCleaner for Thanos

Cleaning pipeline (applied in order):
  1. Remove blank lines
  2. Trim leading/trailing whitespace from each line
  3. Join lines into a single string (removes all line breaks)
  4. Collapse multiple spaces into one
  5. Trim final output
"""

import re


class DataCleaner:
    """Cleans raw text data before tokenization or training."""

    def clean(self, text: str) -> str:
        # Step 1: Split into lines and remove blank lines
        lines = text.splitlines()
        lines = [line for line in lines if line.strip()]

        # Step 2: Trim leading/trailing whitespace from each line
        lines = [line.strip() for line in lines]

        # Step 3: Join all lines into one string (removes line breaks)
        text = " ".join(lines)

        # Step 4: Collapse multiple spaces into a single space
        text = re.sub(r" {2,}", " ", text)

        # Step 5: Trim final output
        text = text.strip()

        return text
