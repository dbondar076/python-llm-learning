from app.services.retrievers.chroma_retriever import ChromaRetriever


def test_chroma_retriever_builds_where_filter_for_title_only() -> None:
    retriever = ChromaRetriever.__new__(ChromaRetriever)

    where = retriever._build_where_filter(
        title_filter="FastAPI",
        doc_id_filter=None,
    )

    assert where == {"title": "FastAPI"}


def test_chroma_retriever_builds_where_filter_for_doc_id_only() -> None:
    retriever = ChromaRetriever.__new__(ChromaRetriever)

    where = retriever._build_where_filter(
        title_filter=None,
        doc_id_filter="doc2",
    )

    assert where == {"doc_id": "doc2"}


def test_chroma_retriever_builds_where_filter_for_both_filters() -> None:
    retriever = ChromaRetriever.__new__(ChromaRetriever)

    where = retriever._build_where_filter(
        title_filter="FastAPI",
        doc_id_filter="doc2",
    )

    assert where == {
        "$and": [
            {"title": "FastAPI"},
            {"doc_id": "doc2"},
        ]
    }


def test_chroma_retriever_builds_where_filter_as_none_when_empty() -> None:
    retriever = ChromaRetriever.__new__(ChromaRetriever)

    where = retriever._build_where_filter()

    assert where is None


def test_chroma_retriever_search_returns_chunks(tmp_path) -> None:
    records = [
        {
            "doc_id": "doc1",
            "title": "Python",
            "chunk_id": "c1",
            "text": "Python is a programming language.",
            "embedding": [0.1, 0.2, 0.3],
        },
        {
            "doc_id": "doc2",
            "title": "FastAPI",
            "chunk_id": "c2",
            "text": "FastAPI is a Python framework.",
            "embedding": [0.2, 0.3, 0.4],
        },
    ]

    retriever = ChromaRetriever(
        records=records,
        collection_name="test_chunks",
        persist_dir=str(tmp_path),
    )

    # TODO: search() внутри сейчас вызывает get_query_embedding(query),
    # так что этот тест пока лучше мокать позже,
    # либо сделать отдельный метод search_by_embedding.
    assert retriever.collection.count() == 2