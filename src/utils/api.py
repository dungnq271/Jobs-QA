import time
from typing import Dict, List

import requests  # type: ignore

from .wrapper import calculate_time


def update_llm(model_name: str):
    requests.put(
        "http://127.0.0.1:8000/agent/update_llm",
        json={"text": model_name},
    )


@calculate_time("Request List API Tools")
def get_list_api_tools():
    response = requests.post(url="http://127.0.0.1:8000/tool/get_api_tools_name")
    assert response.status_code == 200, response.status_code
    response_json = response.json()

    if isinstance(response_json, List):
        return response_json
    else:
        raise TypeError("Response must be of type List")


def get_response(text_input: str):
    response = requests.post(
        url="http://127.0.0.1:8000/agent/chat",
        json={"text": text_input},
    )
    assert response.status_code == 200, response.status_code
    response_json = response.json()
    if isinstance(response_json, Dict):
        response_str = response_json["response"]
    elif isinstance(response_json, str):
        response_str = response_json
    else:
        raise TypeError("Response must be of type str or Dict")
    return response_str


def response_generator(response: str):
    for word in response.split():
        yield word + " "
        time.sleep(0.05)


async def add_document(filepath: str):
    requests.put(
        "http://127.0.0.1:8000/document/add_document",
        json={"text": filepath},
    )


async def add_api_tools(tool_list: List[str]):
    requests.put(
        "http://127.0.0.1:8000/tool/add_api_tools_use",
        json={"text_list": tool_list},
    )


async def remove_api_tools(tool_list: List[str]):
    requests.put(
        "http://127.0.0.1:8000/tool/remove_tools_use",
        json={"text_list": tool_list},
    )


def get_tool_call():
    response = requests.post("http://127.0.0.1:8000/tool/get_latest_tool_call")
    response_json = response.json()
    return response_json
