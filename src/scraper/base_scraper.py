from abc import ABC, abstractmethod


class BaseScraper(ABC):
    def __init__(
        self,
        output_dpath: str,
        top_recent: int,
    ):
        self.output_dpath = output_dpath
        self.top_recent = top_recent

    @abstractmethod
    def scrape(self, url: str, name: str):
        """Start scraping process."""
