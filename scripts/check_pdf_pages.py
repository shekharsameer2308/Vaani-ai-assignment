import sys
import pypdf

pdf_path = sys.argv[1] if len(sys.argv) > 1 else "submission.pdf"
try:
    reader = pypdf.PdfReader(pdf_path)
    print(f"File: {pdf_path}")
    print(f"Pages: {len(reader.pages)}")
except Exception as e:
    print(f"Error reading {pdf_path}: {e}")
