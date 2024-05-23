import time

import requests  # type: ignore


def get_response(input: str):
    response = requests.post(
        url="http://127.0.0.1:8000/agent/chat",
        json={"text": input},
    )
    assert response.status_code == 200, response.status_code
    json_response = response.json()
    if isinstance(json_response, dict):
        response_str = json_response["response"]
    elif isinstance(json_response, str):
        response_str = json_response
    else:
        raise TypeError("Response must be of type str or Dict")
    return response_str


def response_generator(response: str):
    for word in response.split():
        yield word + " "
        time.sleep(0.05)


def update_llm(model_name: str):
    requests.put(
        "http://127.0.0.1:8000/agent/update_llm",
        json={"text": model_name},
    )


def update_table(file_path: str):
    requests.put(
        "http://127.0.0.1:8000/document/update_table",
        json={"text": file_path},
    )


def get_path():
    response = requests.post("http://127.0.0.1:8000/document/get_path")
    assert response.status_code == 200, response.status_code
    return response.json()
