import os
import time

import requests  # type: ignore
from astrapy.db import AstraDB
from sqlalchemy import text


def save_file(filebytes, filepath):
    with open(filepath, "wb") as f:
        f.write(filebytes)
    f.close()


def calculate_time(func):
    def timing(*args, **kwargs):
        t1 = time.time()
        outputs = func(*args, **kwargs)
        t2 = time.time()
        print(f"Time: {(t2-t1):.3f}s")
        return outputs

    return timing


def get_response(input):
    response = requests.post(
        url="http://127.0.0.1:8000/query",
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


def delete_astradb(table_name):
    # Drop the table created for this session
    db = AstraDB(
        token=os.getenv("ASTRA_TOKEN"),
        api_endpoint=os.getenv("ASTRA_API_ENDPOINT"),
        namespace=os.getenv("ASTRA_NAMESPACE"),
    )
    db.delete_collection(collection_name=table_name)
    print("----------------------APP EXITED----------------------")


def debug_qa(query, response, engine):
    print("\n***********Query***********")
    print(query)
    print("\n***********Response***********")
    print(response)

    print("\n***********Source Nodes***********")
    for node in response.source_nodes:
        # display_source_node(node, source_length=2000)
        print(node.text)

    if len(response.metadata) > 0:
        print("\n***********SQL Query***********")
        if "sql_query" in response.metadata:
            sql_query = response.metadata["sql_query"]
            print("Command:", sql_query)
            try:
                with engine.connect() as conn:
                    cursor = conn.execute(text(sql_query))
                    result = cursor.fetchall()
                print("Result:", result)
            except RuntimeError:
                print("SQL Command invalid!")


def modify_days_to_3digits(day=str):
    words = day.split()
    try:
        nday = int(words[0])
        return " ".join([f"{nday:03}"] + words[1:])
    except ValueError:
        return "9999"
