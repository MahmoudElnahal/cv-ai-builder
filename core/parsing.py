from typing import Optional
import fitz  # PyMuPDF
import pdfplumber

def pdf_to_text_pymupdf(data: bytes) -> str:
    text = []
    with fitz.open(stream=data, filetype="pdf") as doc:
        for page in doc:
            text.append(page.get_text("text") or "")
    return "\n".join(text)

def pdf_to_text_pdfplumber(data: bytes) -> str:
    text = []
    with pdfplumber.open(io.BytesIO(data)) as pdf:  # type: ignore
        for page in pdf.pages:
            text.append(page.extract_text() or "")
    return "\n".join(text)

def parse_pdf_text(data: bytes) -> str:
    try:
        return pdf_to_text_pymupdf(data)
    except Exception:
        try:
            return pdf_to_text_pdfplumber(data)
        except Exception:
            return ""
