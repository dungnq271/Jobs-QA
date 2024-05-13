from abc import ABC, abstractmethod
from typing import Dict, Optional

import pandas as pd


class BaseReader(ABC):
    """Base file reader."""

    @abstractmethod
    def load_data(
        self,
        filepath: str,
        metadata: Dict,
        table: Optional[pd.DataFrame] = None,
        **kwargs,
    ):
        """Get list of documents or nodes"""
