"""Google Search tool spec."""

import json
import os
import urllib.parse
from concurrent.futures import ThreadPoolExecutor

import requests  # type: ignore
from bs4 import BeautifulSoup
from llama_index.core.schema import Document
from llama_index.core.tools.tool_spec.base import BaseToolSpec

from src.utils.text import remove_redundant_whitespaces

QUERY_URL_TMPL = (
    "https://www.googleapis.com/customsearch/v1?key={key}&cx={engine}&q={query}"
)


class CustomGoogleSearchToolSpec(BaseToolSpec):
    """Custom Google Search tool spec."""

    spec_functions = ["google_search"]

    def __init__(
        self,
        key: str,
        engine: str,
        num: int | None = None,
        log_dir: str = "tool_results",
    ) -> None:
        """Initialize with parameters."""
        self.key = key
        self.engine = engine
        self.num = num

        os.makedirs(log_dir, exist_ok=True)
        self.save_result_path = os.path.join(
            log_dir, self.spec_functions[0] + "_result.json"
        )

    @staticmethod
    def scrape_url(url: str):
        content = requests.get(url).content
        soup = BeautifulSoup(content, "html.parser")
        return soup.get_text()

    def google_search(self, query: str):
        """
        Make a query to the Google search engine to receive a list of results.
        Then scrape the text from link of each result.

        Args:
            query (str): The query to be passed to Google search.
            num (int, optional): The number of search results to return.
            Defaults to None.

        Raises:
            ValueError: If the 'num' is not an integer between 1 and 10.
        """
        # Query Google Search
        url = QUERY_URL_TMPL.format(
            key=self.key, engine=self.engine, query=urllib.parse.quote_plus(query)
        )

        if self.num is not None:
            if not 1 <= self.num <= 10:
                raise ValueError("num should be an integer between 1 and 10, inclusive")
            url += f"&num={self.num}"

        response = requests.get(url).json()

        # Save the result
        with open(self.save_result_path, "w") as f:
            json.dump(response, f)

        # Scrape content of each url
        urls = [item["link"] for item in response["items"]]

        with ThreadPoolExecutor(max_workers=os.cpu_count()) as exc:
            url_texts = [
                Document(text=remove_redundant_whitespaces(text))
                for text in exc.map(self.scrape_url, urls)
            ]

        return url_texts
