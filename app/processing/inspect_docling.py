from pathlib import Path

from app.processing.parser import build_converter, parse_pdf


PDF_PATH = Path(r"C:\personal-projects\RAG\study_rag\data\raw_pdfs\corpus_pdfs\BLW-002.pdf")


def main():
    converter = build_converter()
    doc = parse_pdf(converter, PDF_PATH)

    print("\n" + "=" * 80)
    print("DOCUMENT")
    print("=" * 80)

    print("Pages:", doc.num_pages())
    print("Tables:", len(doc.tables))
    print("Pictures:", len(doc.pictures))

    # --------------------------------------------------
    # TEXT
    # --------------------------------------------------

    print("\n" + "=" * 80)
    print("TEXT ITEMS")
    print("=" * 80)

    for i, item in enumerate(doc.texts[:20]):
        print(f"\n[{i}]")
        print("Type:", type(item).__name__)
        print("Text:", item.text)

    # --------------------------------------------------
    # TABLES
    # --------------------------------------------------

    print("\n" + "=" * 80)
    print("TABLES")
    print("=" * 80)

    for i, table in enumerate(doc.tables[:5]):
        print(f"\n--- TABLE {i + 1} ---")

        print("Page:", table.prov[0].page_no if table.prov else None)

        df = table.export_to_dataframe(doc)

        print(df.to_string(index=False))

    # --------------------------------------------------
    # MARKDOWN
    # --------------------------------------------------

    print("\n" + "=" * 80)
    print("MARKDOWN PREVIEW")
    print("=" * 80)

    markdown = doc.export_to_markdown()

    print(markdown[:5000])


if __name__ == "__main__":
    main()