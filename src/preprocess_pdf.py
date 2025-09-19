# preprocess_pdf.py

import pdfplumber
from pathlib import Path
import pytesseract
from pdf2image import convert_from_path
import json
from concurrent.futures import ThreadPoolExecutor
from src.kb_utils import clean_text
import logging
from src.logging_config import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


def preprocess_pdf(file_path, output_path=None):
    """
    Extracts text from a PDF file using pdfplumber, applies OCR on pages
    where text cannot be extracted, cleans the text, and saves the results
    into a JSON file.

    Args:
        file_path (str or Path): Path to the input PDF file.
        output_path (str or Path, optional): Path to the output JSON file.
            Defaults to the same filename as the PDF with `.json` extension.

    Returns:
        Path: Path to the generated JSON file containing:
            {
                "file_name": <original PDF filename>,
                "pages": [<cleaned text per page>]
            }

    Workflow:
        1. Extracts text from each PDF page with pdfplumber.
        2. Runs OCR (Tesseract) on pages that look garbled or contain no text.
        3. Cleans extracted text using `clean_text`.
        4. Saves all page texts into a structured JSON file.
    """
    # Implementation hidden for demo purposes
    raise NotImplementedError("Implementation hidden in public repository")


def looks_garbled(text):
    """
    Heuristic check to detect if extracted text is garbled (likely OCR needed).

    Args:
        text (str): Extracted text from a PDF page.

    Returns:
        bool: True if the text contains a high ratio of non-ASCII characters,
              suggesting it's unreadable or incorrectly extracted.
    """
    if not text:
        return True
    non_ascii = sum(1 for c in text if ord(c) > 127)
    return (non_ascii / max(len(text), 1)) > 0.3


if __name__ == "__main__":
    import sys
    preprocess_pdf(sys.argv[1])
