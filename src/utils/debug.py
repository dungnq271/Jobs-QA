# from llama_index.core import Settings
# from llama_index.core.callbacks import (
#     CallbackManager,
#     CBEventType,
#     EventPayload,
#     LlamaDebugHandler,
# )
import sqlalchemy
from sqlalchemy import text

# Using the LlamaDebugHandler to print the trace of the sub questions
# captured by the SUB_QUESTION callback event type
# llama_debug = LlamaDebugHandler(print_trace_on_end=True)
# callback_manager = CallbackManager([llama_debug])

# Settings.callback_manager = callback_manager


def debug_table_qa(query, response, db_engine, **kwargs):
    print("\n***********Query***********")
    print(query)
    print("\n***********Response***********")
    print(response)

    print("\n***********Source Nodes***********")
    for i, source_node in enumerate(response.source_nodes):
        # display_source_node(node, source_length=2000)
        print(f"Node {i+1}:", source_node.text)
        print("\n")

    if len(response.metadata) > 0:
        print("\n***********SQL Query***********")
        if "sql_query" in response.metadata:
            sql_query = response.metadata["sql_query"]
            print("Command:", sql_query)
            try:
                with db_engine.connect() as conn:
                    cursor = conn.execute(text(sql_query))
                    result = cursor.fetchall()
                print("Result:", result)
            except sqlalchemy.exc.OperationalError:
                print("SQL Command invalid!")
        print("\n*******************************")
    else:
        print("\n**********************************")


def debug_subquestion():
    pass


#     # iterate through sub_question items captured in SUB_QUESTION event

#     for i, (start_event, end_event) in enumerate(
#         llama_debug.get_event_pairs(CBEventType.SUB_QUESTION)
#     ):
#         qa_pair = end_event.payload[EventPayload.SUB_QUESTION]
#         print(
#             "Sub Question "
#             + str(i)
#             + ": "
#             + qa_pair.sub_q.sub_question.strip()
#         )
#         print("Answer: " + qa_pair.answer.strip())
#         print("====================================")
