"""
Extract text from example-works PDFs into readable .txt files.
Each PDF is saved as a separate .txt file with page markers.

Usage:
    python extract_pdfs.py
"""

import sys
import io
import os

# Ensure UTF-8 output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

try:
    import PyPDF2
except ImportError:
    print("PyPDF2 not installed. Installing...")
    os.system(f"{sys.executable} -m pip install PyPDF2")
    import PyPDF2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

PDFS = [
    {
        "input": os.path.join(SCRIPT_DIR, "Tailored structured peptide design with a.pdf"),
        "output": os.path.join(SCRIPT_DIR, "Tailored_structured_peptide_design.txt"),
        "label": "Tailored structured peptide design with a genetic algorithm",
    },
    {
        "input": os.path.join(SCRIPT_DIR, "Reshaping the discovery of self-assembling.pdf"),
        "output": os.path.join(SCRIPT_DIR, "Reshaping_the_discovery_of_self_assembling.txt"),
        "label": "Reshaping the discovery of self-assembling peptides",
    },
]


def extract_pdf(pdf_path: str, txt_path: str, label: str) -> None:
    """Extract all text from a PDF and write to a .txt file with page markers."""
    print(f"Processing: {label}")
    print(f"  Input:  {pdf_path}")
    print(f"  Output: {txt_path}")

    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        num_pages = len(reader.pages)
        print(f"  Pages:  {num_pages}")

        with open(txt_path, "w", encoding="utf-8", errors="replace") as out:
            out.write(f"{'='*80}\n")
            out.write(f"  {label}\n")
            out.write(f"  Extracted from: {os.path.basename(pdf_path)}\n")
            out.write(f"  Total pages: {num_pages}\n")
            out.write(f"{'='*80}\n\n")

            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    out.write(f"{'─'*80}\n")
                    out.write(f"  PAGE {i + 1} of {num_pages}\n")
                    out.write(f"{'─'*80}\n\n")
                    out.write(text.strip())
                    out.write("\n\n")

    print(f"  Done! Written to {os.path.basename(txt_path)}\n")


def main():
    print("=" * 60)
    print("  PDF Text Extractor for Example Works")
    print("=" * 60)
    print()

    for pdf_info in PDFS:
        if not os.path.exists(pdf_info["input"]):
            print(f"  WARNING: File not found: {pdf_info['input']}")
            continue
        extract_pdf(pdf_info["input"], pdf_info["output"], pdf_info["label"])

    print("All done! Text files are ready for reference.")


if __name__ == "__main__":
    main()
