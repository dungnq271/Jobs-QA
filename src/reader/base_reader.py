from abc import ABC, abstractmethod
from typing import Dict


class BaseReader(ABC):
    """Base file reader."""

    @abstractmethod
    def load_data(self, file_path: str, metadata: Dict, **kwargs):
        """Get list of documents or nodes"""
        pass
