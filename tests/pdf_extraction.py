#!/usr/bin/env python3
"""Minimal PDF -> images -> OCR test script.

Place `sample.pdf` next to this script and run:
    python tests/pdf_extraction.py

The script will create page images in `tests/tmp-images`, call the OCR API
`http://localhost:8000/api/v1/images/ocr` for each page, and write
results to `tests/output.txt`.
"""
import os
import io
import sys
import requests

PDF = "tests/HR-Policy.pdf"
OCR_URL = os.environ.get("OCR_URL", "http://localhost:8000/api/v1/images/ocr")
OUT = "tests/output.txt"

# configurable retries / timeout
OCR_TIMEOUT = int(os.environ.get("OCR_TIMEOUT", "300"))
OCR_RETRIES = int(os.environ.get("OCR_RETRIES", "3"))
PDF_DPI = int(os.environ.get("PDF_DPI", "100"))
# how many pages to read: default 4. If explicitly set to empty string, read all pages.
_RP = os.environ.get("READ_PAGES")
if _RP is None:
    read_pages = 4
elif _RP.strip() == "":
    read_pages = None
else:
    try:
        read_pages = int(_RP)
    except Exception:
        read_pages = 4

def pdf_to_images(pdf_path, out_dir):
    """Try PyMuPDF (fitz) first, then pdf2image if available."""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(pdf_path)
        paths = []
        for i, page in enumerate(doc, start=1):
            pix = page.get_pixmap(dpi=PDF_DPI)
            out_path = os.path.join(out_dir, f"page-{i}.png")
            pix.save(out_path)
            paths.append(out_path)
        return paths
    except Exception:
        pass

    try:
        from pdf2image import convert_from_path
        imgs = convert_from_path(pdf_path, dpi=200)
        paths = []
        for i, img in enumerate(imgs, start=1):
            out_path = os.path.join(out_dir, f"page-{i}.png")
            img.save(out_path, format="PNG")
            paths.append(out_path)
        return paths
    except Exception as e:
        raise RuntimeError(f"No PDF renderer available: {e}")


def ocr_image_file(path):
    last_exc = None
    for attempt in range(1, OCR_RETRIES + 1):
        with open(path, "rb") as f:
            files = {"file": (os.path.basename(path), f, "image/png")}
            try:
                r = requests.post(OCR_URL, files=files, timeout=OCR_TIMEOUT)
                r.raise_for_status()
                try:
                    j = r.json()
                    return j.get("text") if isinstance(j, dict) else str(j)
                except Exception:
                    return r.text or ""
            except Exception as e:
                last_exc = e
                print(f"OCR request attempt {attempt} failed for {path}: {e}")
                if attempt < OCR_RETRIES:
                    import time
                    backoff = 2 ** attempt
                    print(f"Retrying after {backoff}s...")
                    time.sleep(backoff)
                    continue
    return f"[ERROR] {last_exc}"


TMP_DIR = "tests/tmp-images"

def main():
    if not os.path.exists(PDF):
        print(f"Place sample.pdf next to this script at: {PDF}")
        sys.exit(1)

    os.makedirs(TMP_DIR, exist_ok=True)
    print("Converting PDF to images...")
    try:
        pages = pdf_to_images(PDF, TMP_DIR)
        # limit pages if read_pages is set (None means all)
        if read_pages is not None and read_pages > 0:
            pages = pages[:read_pages]
    except Exception as e:
        print("Failed to render PDF:", e)
        sys.exit(1)

    results = []
    for i, img_path in enumerate(pages, start=1):
        print(f"OCR page {i} -> {img_path}")
        text = ocr_image_file(img_path)
        results.append((i, text))

    # write output
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for i, text in results:
            f.write(f"page-{i}:\n")
            f.write((text or "").strip())
            f.write("\n\n")

    print(f"Wrote OCR output to {OUT}")


if __name__ == "__main__":
    main()
