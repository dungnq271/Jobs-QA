from abc import ABC, abstractmethod
from typing import List

from llama_index.core import VectorStoreIndex


class BaseDatabase(ABC):
    """Base database builder."""

    @abstractmethod
    def __init__(self):
        """Initial database."""

    @property
    @abstractmethod
    def index(self) -> VectorStoreIndex:
        """Get index of database"""

    @abstractmethod
    def insert_documents(self, documents: List):
        """Add document to the database."""

    @abstractmethod
    def delete_documents(self, file_paths: List):
        """Delete document from the database."""
