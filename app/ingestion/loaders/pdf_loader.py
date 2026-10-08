from pathlib import Path

from app.processing.parser import build_converter, parse_pdf, save_parsed_document


def load_pdfs(input_dir: str):
    input_path = Path(input_dir)

    converter = build_converter()

    documents = []

    for pdf_path in sorted(input_path.glob("*.pdf")):
        print(f"Parsing: {pdf_path.name}")

        doc = parse_pdf(converter, pdf_path)
        parsed_path = save_parsed_document(doc, pdf_path)
        print(f"Saved: {parsed_path}")

        documents.append({
            "document_id": pdf_path.stem,
            "source_file": pdf_path.name,
            "document": doc,
            "parsed_path": str(parsed_path),
        })

    return documents