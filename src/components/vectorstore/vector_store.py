# src/vectorstore/vector_store.py

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain.schema import Document
from langchain_chroma import Chroma

from src.exception import CustomException
from src.logger import logging

# Load env vars for OpenAI or HF keys
load_dotenv(override=True)

# Project root: .../Banking-rag/
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class VectorStoreConfig:
    """
    Configuration for vector store persistence.
    """
    db_dir: str = str(PROJECT_ROOT / "artifacts" / "vector_db")
    provider: str = "openai"
    openai_model: str = "text-embedding-3-large"
    hf_model: str = "sentence-transformers/all-MiniLM-L6-v2"


class VectorStoreBuilder:
    def __init__(self):
        self.config = VectorStoreConfig()
        self._embeddings = self._init_embeddings_model()

    def _init_embeddings_model(self):
        """
        Initialise embeddings model for the vector store.
        """
        logging.info(
            f"Initialising vector store embeddings "
            f"(provider={self.config.provider})"
        )
        try:
            if self.config.provider.lower() == "openai":
                from langchain_openai import OpenAIEmbeddings

                return OpenAIEmbeddings(model=self.config.openai_model)

            elif self.config.provider.lower() == "huggingface":
                from langchain_huggingface import HuggingFaceEmbeddings

                return HuggingFaceEmbeddings(model_name=self.config.hf_model)

            else:
                raise ValueError(f"Unknown embeddings provider: {self.config.provider}")

        except Exception as e:
            logging.error("Error initialising embeddings model in VectorStoreBuilder", exc_info=True)
            raise CustomException(e, sys)

    def build_from_chunks(self, chunks: List[Document]) -> Chroma:
        """
        Build (or rebuild) a Chroma vector store from a list of chunked Documents.

        - Deletes any existing collection at the configured directory.
        - Creates a new Chroma store with fresh embeddings.
        - Logs number of vectors and embedding dimensions.

        Returns:
            Chroma vector store instance.
        """
        logging.info("Entered VectorStoreBuilder.build_from_chunks")

        try:
            if not chunks:
                logging.warning("No chunks provided to VectorStoreBuilder.build_from_chunks")
                # Still construct an empty vector store
                os.makedirs(self.config.db_dir, exist_ok=True)
                vectorstore = Chroma(
                    persist_directory=self.config.db_dir,
                    embedding_function=self._embeddings,
                )
                return vectorstore

            db_dir = self.config.db_dir

            # If there is an existing DB, drop it (like your previous code)
            if os.path.exists(db_dir):
                logging.info(f"Existing vector DB found at {db_dir}; deleting collection")
                Chroma(
                    persist_directory=db_dir,
                    embedding_function=self._embeddings,
                ).delete_collection()

            logging.info(
                f"Creating Chroma vector store from {len(chunks)} chunks "
                f"at {db_dir}"
            )

            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self._embeddings,
                persist_directory=db_dir,
            )

            # Access underlying collection to log stats
            collection = vectorstore._collection  # Chroma internal; OK for inspection
            count = collection.count()

            dims = None
            if count > 0:
                sample_embedding = collection.get(
                    limit=1, include=["embeddings"]
                )["embeddings"][0]
                dims = len(sample_embedding)

            logging.info(
                f"There are {count:,} vectors in the vector store"
                + (f" with {dims:,} dimensions" if dims is not None else "")
            )

            return vectorstore

        except Exception as e:
            logging.error("Error occurred in VectorStoreBuilder.build_from_chunks", exc_info=True)
            raise CustomException(e, sys)


if __name__ == "__main__":
    # Example full pipeline run
    from src.ingestion.fetch_documents import fetch_documents
    from src.chunking.document_chunking import DocumentChunking

    logging.info("Running vector store builder as a script")

    # 1) Load documents from data/
    docs = fetch_documents()

    # 2) Chunk them
    chunker = DocumentChunking()
    _, chunks = chunker.create_chunks(docs)

    # 3) Build vector store
    vs_builder = VectorStoreBuilder()
    vs = vs_builder.build_from_chunks(chunks)

    print("Vector store created and persisted.")
