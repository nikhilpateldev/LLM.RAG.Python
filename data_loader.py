"""
Load documents from a directory. Supports:
 - PDF (.pdf) via pdfplumber
 - HTML (.html, .htm) via BeautifulSoup
 - Plain text (.txt)

Returns a list of dicts: {"id": "<path-based-id>", "text": "...", "meta": {...}}
"""
import os, pathlib, hashlib
from typing import List, Dict
import pdfplumber
from bs4 import BeautifulSoup

def file_id(path: str) -> str:
    # stable id based on path
    return hashlib.sha1(path.encode("utf-8")).hexdigest()[:16]

def load_pdf(path: str) -> str:
    text_parts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            txt = page.extract_text()
            if txt:
                text_parts.append(txt)
    return "\n".join(text_parts)

def load_html(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        html = f.read()
    soup = BeautifulSoup(html, "html.parser")
    # remove scripts/styles
    for s in soup(["script", "style", "noscript"]):
        s.decompose()
    text = soup.get_text(separator="\n")
    # collapse whitespace
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return "\n".join(lines)

def load_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def load_directory(directory: str) -> List[Dict]:
    directory = os.path.abspath(directory)
    results = []
    for root, dirs, files in os.walk(directory):
        for fname in files:
            path = os.path.join(root, fname)
            _, ext = os.path.splitext(fname.lower())
            try:
                if ext == ".pdf":
                    text = load_pdf(path)
                elif ext in (".html", ".htm"):
                    text = load_html(path)
                elif ext == ".txt":
                    text = load_txt(path)
                else:
                    # skip unknown file types
                    continue
            except Exception as e:
                print(f"Failed to read {path}: {e}")
                continue
            if not text or len(text.strip()) == 0:
                continue
            results.append({"id": file_id(path), "path": path, "text": text, "meta": {"filename": fname}})
    return results
