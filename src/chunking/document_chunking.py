# src/chunking/document_chunking.py

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.exception import CustomException
from src.logger import logging

# Project root: .../Banking-rag/
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class DocumentChunkingConfig:
    """
    Configuration for document chunking & persistence.
    """
    # <project_root>/artifacts/chunks
    chunks_dir: str = str(PROJECT_ROOT / "artifacts" / "chunks")
    # CSV file to store chunks
    chunks_file: str = str(PROJECT_ROOT / "artifacts" / "chunks" / "chunks.csv")
    # Chunking params
    chunk_size: int = 1000
    chunk_overlap: int = 200


class DocumentChunking:
    def __init__(self):
        self.config = DocumentChunkingConfig()

    def create_chunks(self, documents: List[Document]) -> Tuple[str, List[Document]]:
        """
        Split a list of LangChain Documents into smaller chunks and save them to CSV.

        Args:
            documents: List of LangChain Document objects (from fetch_documents)

        Returns:
            chunks_file_path (str): path to the saved chunks CSV.
            chunks (List[Document]): in-memory list of chunked Documents.
        """
        logging.info("Entered DocumentChunking.create_chunks")

        try:
            if not documents:
                logging.warning("No documents provided to DocumentChunking.create_chunks")
                # Ensure directory exists
                os.makedirs(self.config.chunks_dir, exist_ok=True)
                # Save an empty CSV with correct columns
                empty_df = pd.DataFrame(columns=["text", "source", "doc_type"])
                empty_df.to_csv(self.config.chunks_file, index=False)
                return self.config.chunks_file, []

            logging.info(
                f"Splitting {len(documents)} documents into chunks "
                f"(size={self.config.chunk_size}, overlap={self.config.chunk_overlap})"
            )

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
            )

            chunks: List[Document] = splitter.split_documents(documents)
            logging.info(f"Created {len(chunks)} chunks from {len(documents)} documents")

            # Ensure directory exists
            os.makedirs(self.config.chunks_dir, exist_ok=True)

            # Convert chunks to a DataFrame
            df = pd.DataFrame(
                [
                    {
                        "text": chunk.page_content,
                        "source": chunk.metadata.get("source", ""),
                        "doc_type": chunk.metadata.get("doc_type", ""),
                    }
                    for chunk in chunks
                ]
            )

            df.to_csv(self.config.chunks_file, index=False, header=True)
            logging.info(
                f"Saved {len(chunks)} chunks to {self.config.chunks_file}"
            )

            return self.config.chunks_file, chunks

        except Exception as e:
            logging.error(
                "Error occurred in DocumentChunking.create_chunks", exc_info=True
            )
            raise CustomException(e, sys)


if __name__ == "__main__":
    # Example wiring with your existing fetch_documents
    from src.ingestion.load_documents import fetch_documents

    logging.info("Running document chunking as a script")
    docs = fetch_documents()  # uses your DATA_DIR and PDF loader

    chunker = DocumentChunking()
    chunks_path, chunks = chunker.create_chunks(docs)

    print(f"Chunks saved to: {chunks_path}")
    print(f"Total chunks: {len(chunks)}")
