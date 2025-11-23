# src/retrieval/document_retriever.py

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from langchain_chroma import Chroma

from src.exception import CustomException
from src.logger import logging

from dotenv import load_dotenv

# Ensure .env is loaded for embedding provider
load_dotenv(override=True)

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class RetrieverConfig:
    """
    Configuration for retrieving vectors.
    """
    db_dir: str = str(PROJECT_ROOT / "artifacts" / "vector_db")
    provider: str = "openai"
    openai_model: str = "text-embedding-3-large"
    hf_model: str = "sentence-transformers/all-MiniLM-L6-v2"


class DocumentRetriever:
    """
    Wrapper around Chroma’s retriever with optional metadata filtering.
    """

    def __init__(self):
        self.config = RetrieverConfig()
        self._embeddings = self._init_embeddings_model()
        self._vectorstore = self._load_vector_store()

    def _init_embeddings_model(self):
        """
        Initialise the same embedding model used in vector store creation.
        """
        logging.info("Initialising embedding model for Retriever")

        try:
            if self.config.provider.lower() == "openai":
                from langchain_openai import OpenAIEmbeddings
                return OpenAIEmbeddings(model=self.config.openai_model)

            elif self.config.provider.lower() == "huggingface":
                from langchain_huggingface import HuggingFaceEmbeddings
                return HuggingFaceEmbeddings(model_name=self.config.hf_model)

            else:
                raise ValueError(f"Unknown embedding provider: {self.config.provider}")

        except Exception as e:
            logging.error("Error initialising embeddings in DocumentRetriever", exc_info=True)
            raise CustomException(e, sys)

    def _load_vector_store(self) -> Chroma:
        """
        Load the existing Chroma vector store from disk.
        """
        logging.info(f"Loading vector DB from {self.config.db_dir}")

        if not os.path.exists(self.config.db_dir):
            raise FileNotFoundError(
                f"No vector database found at {self.config.db_dir}. "
                f"Please build it first."
            )

        try:
            vectorstore = Chroma(
                persist_directory=self.config.db_dir,
                embedding_function=self._embeddings,
            )
            logging.info("Vector store loaded successfully")
            return vectorstore

        except Exception as e:
            logging.error("Error loading vector store", exc_info=True)
            raise CustomException(e, sys)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        doc_type: Optional[str] = None,
    ) -> List[Document]:
        """
        Retrieve the most relevant chunks from the vector store.

        Args:
            query: User query string.
            top_k: How many results to return.
            doc_type: Optional metadata filter (e.g. 'product_terms')

        Returns:
            List[Document]
        """
        logging.info(
            f"Retrieving with query='{query}', top_k={top_k}, doc_type={doc_type}"
        )

        try:
            search_kwargs = {"k": top_k}
            if doc_type:
                search_kwargs["filter"] = {"doc_type": doc_type}

            retriever = self._vectorstore.as_retriever(search_kwargs=search_kwargs)
            results = retriever.invoke(query)

            logging.info(f"Retrieved {len(results)} chunks")
            return results

        except Exception as e:
            logging.error("Error in DocumentRetriever.retrieve", exc_info=True)
            raise CustomException(e, sys)


if __name__ == "__main__":
    logging.info("Testing the retrieval module")

    retriever = DocumentRetriever()

    query = "What fees apply for ATM withdrawals?"
    results = retriever.retrieve(query, top_k=3, doc_type="product_terms")

    print("\n--- RESULTS ---")
    for idx, r in enumerate(results, start=1):
        print(f"\n[{idx}] {r.metadata.get('doc_type')} :: {r.metadata.get('source')}")
        print(r.page_content[:300], "...")
