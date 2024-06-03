DEFAULT_FUNTION_QUERY_ARGS = """\
    You are required to pass the natural language query argument \
when calling this endpoint

Args:
    query (str): The natural language query used to retreieve information from the index
"""


DEFAULT_QUERY_TEXT_DESCRIPTION_TMPL = """\
    This tool is useful for answering semantic questions \
from the uploaded file(s)
"""


DEFAULT_FUNCTION_QUERY_TEXT_DESCRIPTION_TMPL = f"""\
{DEFAULT_QUERY_TEXT_DESCRIPTION_TMPL}
{DEFAULT_FUNTION_QUERY_ARGS}
"""
