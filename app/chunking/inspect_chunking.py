from app.chunking.chunker import chunk_parsed_file
from app.processing.parser import PROCESSED_DIR


PARSED_PATH = PROCESSED_DIR / "BLW-002.json"


def main():
    result = chunk_parsed_file(PARSED_PATH)
    chunks = result["chunks"]

    print("Document:", result["document_id"])
    print("Chunks:", len(chunks))

    picture_chunks = [c for c in chunks if "picture" in c["metadata"]["labels"]]
    table_chunks = [c for c in chunks if "table" in c["metadata"]["labels"]]
    print(f"Picture chunks: {len(picture_chunks)}")
    print(f"Table chunks: {len(table_chunks)}")

    for i, chunk in enumerate(chunks[:10]):
        print("\n" + "=" * 80)
        print(f"CHUNK {i}")
        print("Pages:", chunk["metadata"].get("page_start"), "-", chunk["metadata"].get("page_end"))
        print("Headings:", chunk["metadata"]["headings"])
        print("Labels:", chunk["metadata"]["labels"])
        if chunk["metadata"].get("image_refs"):
            print("Image refs:", chunk["metadata"]["image_refs"])
        print(chunk["text"][:500])

    print("\n\n--- SAMPLE TABLE CHUNK ---")
    for chunk in table_chunks[:2]:
        print("\n" + "=" * 80)
        print(f"CHUNK {chunk['chunk_index']}")
        print("Labels:", chunk["metadata"]["labels"])
        print(chunk["text"][:600])

    print("\n\n--- SAMPLE PICTURE CHUNK ---")
    for chunk in picture_chunks[:3]:
        print("\n" + "=" * 80)
        print(f"CHUNK {chunk['chunk_index']}")
        print("Labels:", chunk["metadata"]["labels"])
        if chunk["metadata"].get("image_refs"):
            print("Image refs:", chunk["metadata"]["image_refs"])
        print(chunk["text"][:500])


if __name__ == "__main__":
    main()
