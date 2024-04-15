import time
from typing import Dict

import pandas as pd
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

options = Options()
options.add_experimental_option("detach", True)  # keep the window open

driver = webdriver.Chrome(
    service=Service(ChromeDriverManager().install()), options=options
)
url = """
https://www.google.com/search?channel=fs&client=ubuntu-sn&q=ai+engineer+tuy%E1%BB%83n+d%E1%BB%A5ng&ibp=htl;jobs&sa=X&ved=2ahUKEwizqofEs_qEAxVcc_UHHcCHDEQQudcGKAF6BAgbECw&sxsrf=ACQVn0-XIEogMp2cd_SshdN_dlCCdh23sg:1710647767330#htivrt=jobs&htidocid=3dyy4ZJFWqY6X_LZAAAAAA%3D%3D&fpstate=tldetail
"""
driver.get(url)
driver.maximize_window()


xpaths = {
    "Logo": "./div[1]//img",
    "Role": "./div[2]",
    "Company": "./div[4]/div/div[1]",
    "Location": "./div[4]/div/div[2]",
    "Source": "./div[4]/div/div[3]",
    "Posted": ".//*[name()='path'][contains(@d,'M11.99')]/ancestor::div[1]",
    "Full / Part Time": ".//*[name()='path'][contains(@d,'M20 6')]/"
    "ancestor::div[1]",
    "Salary": ".//*[name()='path'][@fill-rule='evenodd']/ancestor::div[1]",
    "Description": ".//div[contains(@class, 'YgLbBe')]//"
    "span[@class='HBvzbc']",
    "Link": ".//div[contains(@class, 'B8oxKe')]//"
    "a[contains(@class, 'pMhGee')]",
}

data: Dict = {key: [] for key in xpaths}
jobs_to_do = 100
jobs_done = 0


while jobs_done < jobs_to_do:
    lis = driver.find_elements(
        "xpath", "//li[@data-ved]//div[@role='treeitem']/div/div"
    )

    if len(lis[jobs_done:]) == 0:
        break

    # Get all job infos of each li tag
    for li in lis[jobs_done:]:
        li.click()
        driver.execute_script(
            """
            arguments[0].scrollIntoView
            ({block: "center", behavior: "smooth"});
            """,
            li,
        )
        # job_block = driver.find_elements("xpath", ".//
        # div[contains(@class, 'pE8vnd')]")

        # Expand the job description first
        expand_buttons = driver.find_elements(
            "xpath", ".//div[contains(@class, 'YgLbBe')]//div[@role='button']"
        )
        if len(expand_buttons) > 0:
            try:
                expand_buttons[-1].click()
            except NoSuchElementException:
                pass

        for key in xpaths:
            try:
                if key == "Description":

                    t = driver.find_elements("xpath", xpaths[key])[-1].text
                elif key == "Link":
                    t = driver.find_elements("xpath", xpaths[key])[
                        -1
                    ].get_attribute("href")
                else:
                    t = li.find_element("xpath", xpaths[key]).get_attribute(
                        "src" if key == "Logo" else "innerText"
                    )
            except NoSuchElementException:
                t = "*missing data*"
            data[key].append(t)

        jobs_done += 1
        print(f"{jobs_done=}", end="\r")
        time.sleep(0.2)


df = pd.DataFrame(data)
df.to_csv("job_vn_posted_full_recent_v2.csv", index=False)
