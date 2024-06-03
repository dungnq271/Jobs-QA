DEFAULT_TABLE_DESCRIPTION_TMPL = """\
Table name: {table_name}
Table description: {table_description}
Table columns:\n{column_descriptions}
"""


DEFAULT_FUNCTION_SQL_ARGS = """\
    You are then required to pass the SQL query argument when calling this endpoint

    Args:
        query (str): The SQL query used to retreieve information from the tables
"""


DEFAULT_FUNCTION_NLSQL_ARGS = """\
    You are required to pass the natural language query argument \
when calling this endpoint

    Args:
        query (str): The natural language query used \
to be transformed into an SQL query to retreieve information from the tables
"""


DEFAULT_QUERY_SQL_DESCRIPTION_TMPL = """\
    This tool is useful for execute an SQL query over \
one of the following tables:\n{table_description}
    Given an input question, first create a syntactically correct SQL query to run.
    Pay attention to use only the column names that you can see \
in the schema description.
"""


DETAILED_SQL_DESCRIPTION = """
    Never query for all the columns from a specific table, \
only ask for a few relevant columns given the question.
    Pay attention to use only the column names that \
you can see in the schema description.
    Be careful to not query for columns that do not exist.
    Pay attention to which column is in which table.
    Also, qualify column names with the table name when needed.
"""


DEFAULT_QUERY_NLSQL_DESCRIPTION_TMPL = """\
    This tool is useful for translating a natural language query \
into an SQL query over uploaded table(s) about {file_description} \
having columns:\n{column_descriptions}
"""


DEFAULT_FUNCTION_QUERY_SQL_DESCRIPTION_TMPL = f"""\
{DEFAULT_QUERY_SQL_DESCRIPTION_TMPL}
{DEFAULT_FUNCTION_SQL_ARGS}
"""


DEFAULT_FUNCTION_QUERY_NLSQL_DESCRIPTION_TMPL = f"""\
{DEFAULT_QUERY_NLSQL_DESCRIPTION_TMPL}
{DEFAULT_FUNCTION_NLSQL_ARGS}
"""
