from abc import ABC, abstractmethod
from typing import Any


class BaseAgent(ABC):
    """Base agent factory."""

    @abstractmethod
    def get_agent(self):
        """Get the agent."""

    @abstractmethod
    def chat(self, query: str, history: Any = None, **kwargs):
        """Response to user's query."""
