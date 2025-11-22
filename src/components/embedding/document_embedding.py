# src/embedding/document_embedding.py

import os
import sys
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Any

from langchain.schema import Document
from src.exception import CustomException
from src.logger import logging

from dotenv import load_dotenv

# Load .env for OpenAI / HF keys
load_dotenv(override=True)

# Project root: .../Banking-rag/
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class EmbeddingConfig:
    """
    Configuration for document embeddings & persistence.
    """
    # Where to store embeddings artifacts
    embeddings_dir: str = str(PROJECT_ROOT / "artifacts" / "embeddings")
    embeddings_file: str = str(PROJECT_ROOT / "artifacts" / "embeddings" / "embeddings.pkl")

    # Embedding backend: "openai" or "huggingface"
    provider: str = "openai"

    # Model names
    openai_model: str = "text-embedding-3-large"
    hf_model: str = "sentence-transformers/all-MiniLM-L6-v2"


class DocumentEmbedding:
    def __init__(self):
        self.config = EmbeddingConfig()
        self._embeddings_model = self._init_embeddings_model()

    def _init_embeddings_model(self):
        """
        Initialise the underlying embeddings model based on provider.
        """
        logging.info(f"Initialising embeddings model (provider={self.config.provider})")
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
            logging.error("Error initialising embeddings model", exc_info=True)
            raise CustomException(e, sys)

    def create_embeddings(
        self, chunks: List[Document]
    ) -> Tuple[str, List[List[float]], List[str], List[Dict[str, Any]]]:
        """
        Create embeddings for a list of chunked Documents and persist them.

        Args:
            chunks: List of LangChain Document objects (chunked docs).

        Returns:
            embeddings_file_path (str): Path to the saved pickle file.
            vectors (List[List[float]]): Embedding vectors for each chunk.
            texts (List[str]): Text content of each chunk.
            metadatas (List[dict]): Metadata per chunk.
        """
        logging.info("Entered DocumentEmbedding.create_embeddings")

        try:
            if not chunks:
                logging.warning("No chunks provided to create_embeddings")
                # Ensure directory exists even if we store empty
                os.makedirs(self.config.embeddings_dir, exist_ok=True)
                empty_payload = {
                    "embeddings": [],
                    "texts": [],
                    "metadatas": [],
                }
                with open(self.config.embeddings_file, "wb") as f:
                    pickle.dump(empty_payload, f)
                return self.config.embeddings_file, [], [], []

            logging.info(f"Creating embeddings for {len(chunks)} chunks")

            texts: List[str] = [chunk.page_content for chunk in chunks]
            metadatas: List[Dict[str, Any]] = [chunk.metadata for chunk in chunks]

            # This calls OpenAI / HF embedding API under the hood
            vectors: List[List[float]] = self._embeddings_model.embed_documents(texts)

            logging.info("Embeddings created successfully; persisting to disk")

            os.makedirs(self.config.embeddings_dir, exist_ok=True)

            payload = {
                "embeddings": vectors,
                "texts": texts,
                "metadatas": metadatas,
            }

            with open(self.config.embeddings_file, "wb") as f:
                pickle.dump(payload, f)

            logging.info(
                f"Saved embeddings to {self.config.embeddings_file} "
                f"(total vectors: {len(vectors)})"
            )

            return self.config.embeddings_file, vectors, texts, metadatas

        except Exception as e:
            logging.error("Error occurred in DocumentEmbedding.create_embeddings", exc_info=True)
            raise CustomException(e, sys)


if __name__ == "__main__":
    # Example wiring with your existing pipeline
    from src.ingestion.fetch_documents import fetch_documents
    from src.chunking.document_chunking import DocumentChunking

    logging.info("Running embedding module as a script")

    # 1) Load raw documents from data/
    docs = fetch_documents()

    # 2) Chunk them
    chunker = DocumentChunking()
    _, chunks = chunker.create_chunks(docs)

    # 3) Create embeddings
    embedder = DocumentEmbedding()
    emb_path, vectors, texts, metadatas = embedder.create_embeddings(chunks)

    print(f"Embeddings saved to: {emb_path}")
    print(f"Total vectors: {len(vectors)}")
