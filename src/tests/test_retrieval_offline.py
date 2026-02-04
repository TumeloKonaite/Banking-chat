import re

from src.retrieval.document_retriever import DocumentRetriever


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


class _Doc:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata


class _FakeVectorRetriever:
    def __init__(self, docs, search_kwargs):
        self.docs = docs
        self.search_kwargs = search_kwargs

    def invoke(self, query: str):
        query_tokens = _tokenize(query)
        top_k = self.search_kwargs.get("k", 5)
        doc_type_filter = (self.search_kwargs.get("filter") or {}).get("doc_type")

        scored = []
        for doc in self.docs:
            if doc_type_filter and doc.metadata.get("doc_type") != doc_type_filter:
                continue
            overlap = len(query_tokens & _tokenize(doc.page_content))
            if overlap > 0:
                scored.append((overlap, doc.metadata.get("source", ""), doc))

        scored.sort(key=lambda row: (-row[0], row[1]))
        return [row[2] for row in scored[:top_k]]


class _FakeVectorStore:
    def __init__(self, docs):
        self.docs = docs

    def as_retriever(self, search_kwargs):
        return _FakeVectorRetriever(self.docs, search_kwargs)


def _build_retriever_with_corpus():
    corpus = [
        _Doc(
            page_content="ATM withdrawal fees are charged for out-of-network cash withdrawals.",
            metadata={"source": "fees_atm.pdf", "doc_type": "product_terms"},
        ),
        _Doc(
            page_content="Debit order disputes must be raised within 40 days of the debit date.",
            metadata={"source": "debit_disputes.pdf", "doc_type": "agreements"},
        ),
        _Doc(
            page_content="Monthly account fee includes digital banking access and card maintenance.",
            metadata={"source": "monthly_fees.pdf", "doc_type": "product_terms"},
        ),
    ]

    retriever = DocumentRetriever.__new__(DocumentRetriever)
    retriever._vectorstore = _FakeVectorStore(corpus)
    return retriever


def test_retrieval_top_k_returns_expected_docs():
    retriever = _build_retriever_with_corpus()
    results = retriever.retrieve(query="ATM withdrawal fee", top_k=2)

    assert len(results) == 2
    assert results[0].metadata["source"] == "fees_atm.pdf"
    assert "ATM withdrawal fees" in results[0].page_content


def test_retrieval_doc_type_filter_is_applied():
    retriever = _build_retriever_with_corpus()
    results = retriever.retrieve(
        query="debit disputes",
        top_k=3,
        doc_type="agreements",
    )

    assert len(results) == 1
    assert results[0].metadata["source"] == "debit_disputes.pdf"
    assert results[0].metadata["doc_type"] == "agreements"
