from abc import ABC, abstractmethod

import pandas as pd


class BaseReader(ABC):
    """Base file reader."""

    @abstractmethod
    def load_data(
        self,
        file_path: str,
        metadata: dict,
        table: pd.DataFrame | None = None,
        **kwargs,
    ):
        """Get list of documents or nodes"""
