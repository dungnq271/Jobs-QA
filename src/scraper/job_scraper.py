"""
Adapted from: https://stackoverflow.com/questions/75473465/scrape-and-extract-job-data-from-google-jobs-using-selenium-and-store-in-pandas
"""

import contextlib
import os.path as osp
import time

import pandas as pd
from selenium import webdriver
from selenium.common.exceptions import (
    ElementNotInteractableException,
    NoSuchElementException,
)
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

from src.utils import create_dir

from .base_scraper import BaseScraper

xpaths = {
    "Logo": "./div[1]//img",
    "Role": "./div[2]",
    "Company": "./div[4]/div/div[1]",
    "Location": "./div[4]/div/div[2]",
    "Source": "./div[4]/div/div[3]",
    "Posted": ".//*[name()='path'][contains(@d,'M11.99')]/ancestor::div[1]",
    "Full / Part Time": ".//*[name()='path'][contains(@d,'M20 6')]/" "ancestor::div[1]",
    "Salary": ".//*[name()='path'][@fill-rule='evenodd']/ancestor::div[1]",
    "Description": ".//div[contains(@class, 'YgLbBe')]//" "span[@class='HBvzbc']",
    "Link": ".//div[contains(@class, 'B8oxKe')]//" "a[contains(@class, 'pMhGee')]",
}


class JobScraper(BaseScraper):
    def __init__(self, output_dpath: str, top_recent: int):
        create_dir(output_dpath)

        # Initialize selenium web driver
        self.options = Options()
        self.options.add_experimental_option("detach", True)  # keep the window open

        super().__init__(output_dpath, top_recent)

    def scrape(self, url: str, name: str):
        self._driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=self.options,
        )
        self._driver.get(url)
        self._driver.maximize_window()
        self._driver.find_elements("xpath", ".//div[@id='search']//a[1]")[0].click()

        data: dict[str, list] = {key: [] for key in xpaths}
        items_done: int = 0

        while items_done < self.top_recent:
            lis = self._driver.find_elements(
                "xpath", "//li[@data-ved]//div[@role='treeitem']/div/div"
            )

            if len(lis[items_done:]) == 0:
                break

            # Get all job infos of each li tag
            for li in lis[items_done:]:
                li.click()
                self._driver.execute_script(
                    """
                    arguments[0].scrollIntoView
                    ({block: "center", behavior: "smooth"});
                    """,
                    li,
                )
                # job_block = self._driver.find_elements("xpath", ".//
                # div[contains(@class, 'pE8vnd')]")

                # Expand the job description first
                expand_buttons = self._driver.find_elements(
                    "xpath",
                    ".//div[contains(@class, 'YgLbBe')]//div[@role='button']",
                )
                if len(expand_buttons) > 0:
                    with contextlib.suppress(
                        NoSuchElementException, ElementNotInteractableException
                    ):
                        expand_buttons[-1].click()

                for key in xpaths:
                    try:
                        if key == "Description":
                            t = self._driver.find_elements("xpath", xpaths[key])[
                                -1
                            ].text
                        elif key == "Link":
                            t = self._driver.find_elements("xpath", xpaths[key])[
                                -1
                            ].get_attribute("href")
                        else:
                            t = li.find_element("xpath", xpaths[key]).get_attribute(
                                "src" if key == "Logo" else "innerText"
                            )
                    except NoSuchElementException:
                        t = "*missing data*"
                    data[key].append(t)

                items_done += 1
                print(f"{items_done=}", end="\r")
                time.sleep(0.2)

        output_fpath = osp.join(self.output_dpath, name + ".csv")
        df = pd.DataFrame(data)
        df.to_csv(output_fpath, index=False)

        return df, output_fpath
