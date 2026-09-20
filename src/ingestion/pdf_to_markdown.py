import pymupdf4llm
from pathlib import Path

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed/markdown")

def convert_pdf_to_markdown(pdf_path: Path) -> Path:
    md_text = pymupdf4llm.to_markdown(str(pdf_path))
    out_path = OUT_DIR / f"{pdf_path.stem}pdf_markdown.md"
    out_path.write_text(md_text, encoding="utf-8")
    return out_path

if __name__ == "__main__":
    for pdf_file in RAW_DIR.glob("*.pdf"):
        result = convert_pdf_to_markdown(pdf_file)
        print(f"{pdf_file.name} -> {result}")