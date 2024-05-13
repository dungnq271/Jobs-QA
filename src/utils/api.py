import time

import requests  # type: ignore


def get_response(input):
    response = requests.post(
        url="http://127.0.0.1:8000/chat",
        json={"text": input},
    )
    assert response.status_code == 200, response.status_code
    json_response = response.json()
    if isinstance(json_response, dict):
        response = json_response["response"]
    elif isinstance(json_response, str):
        response = json_response
    else:
        raise TypeError("Response must be of type str or Dict")
    return response


def response_generator(response):
    for word in response.split():
        yield word + " "
        time.sleep(0.05)
