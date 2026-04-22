import json
from pydantic import BaseModel, ValidationError


class Document(BaseModel):
    id: str
    title: str
    text: str


class Chunk(BaseModel):
    doc_id: str
    title: str
    chunk_id: str
    text: str


def load_documents(file_path: str) -> list[Document] | None:
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            raw_data = json.load(file)

        if not isinstance(raw_data, list):
            print("Expected a list of documents")
            return None

        documents: list[Document] = []

        for item in raw_data:
            try:
                document = Document(**item)
                documents.append(document)
            except ValidationError as e:
                print(f"Invalid document: {item}")
                print(e.errors())

        return documents

    except FileNotFoundError:
        print("File not found")
        return None
    except json.JSONDecodeError:
        print("Invalid JSON")
        return None


def split_text_into_chunks(text: str, chunk_size: int = 80) -> list[str]:
    words = text.split()
    chunks: list[str] = []

    for i in range(0, len(words), chunk_size):
        chunk_words = words[i:i + chunk_size]
        chunks.append(" ".join(chunk_words))

    return chunks


def build_chunks(documents: list[Document], chunk_size: int = 12) -> list[Chunk]:
    result: list[Chunk] = []

    for doc in documents:
        text_chunks = split_text_into_chunks(doc.text, chunk_size=chunk_size)

        for index, chunk_text in enumerate(text_chunks, start=1):
            chunk = Chunk(
                doc_id=doc.id,
                title=doc.title,
                chunk_id=f"{doc.id}_chunk_{index}",
                text=chunk_text,
            )
            result.append(chunk)

    return result


def save_chunks(chunks: list[Chunk], file_path: str) -> None:
    data = [chunk.model_dump() for chunk in chunks]

    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    documents = load_documents("../documents.json")

    if documents is None:
        print("Failed to load documents.")
    else:
        chunks = build_chunks(documents, chunk_size=12)

        for chunk in chunks:
            print(chunk)

        save_chunks(chunks, "chunks.json")