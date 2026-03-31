from app.services.retrievers.chroma_retriever import ChromaRetriever


def test_chroma_retriever_builds_where_filter_for_title_only():
    retriever = ChromaRetriever.__new__(ChromaRetriever)

    where = retriever._build_where_filter(
        title_filter="Python",
        doc_id_filter=None,
    )

    assert where == {"title": "Python"}


def test_chroma_retriever_builds_where_filter_for_doc_id_only():
    retriever = ChromaRetriever.__new__(ChromaRetriever)

    where = retriever._build_where_filter(
        title_filter=None,
        doc_id_filter="doc-python",
    )

    assert where == {"doc_id": "doc-python"}


def test_chroma_retriever_builds_where_filter_for_both_filters():
    retriever = ChromaRetriever.__new__(ChromaRetriever)

    where = retriever._build_where_filter(
        title_filter="Python",
        doc_id_filter="doc-python",
    )

    assert where == {
        "$and": [
            {"title": "Python"},
            {"doc_id": "doc-python"},
        ]
    }


def test_chroma_retriever_builds_where_filter_as_none_when_empty():
    retriever = ChromaRetriever.__new__(ChromaRetriever)

    where = retriever._build_where_filter()

    assert where is None