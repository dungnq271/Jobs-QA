from llama_index.core import PromptTemplate


file_description = {
    "bootstrap": "",
    "uber_10q_march_2022": "containing financial information about UBER TECHNOLOGIES, INC.",
    "allPlayersLookup": "containing biographical information on NHL hockey players",
    "all_teams": "containing stats related to NHL hockey teams",
}


custom_prompt = PromptTemplate(
    """\
Given a conversation (between Human and Assistant) and a follow up message from Human, \
rewrite the message to be a standalone question that captures all relevant context \
from the conversation.

<Chat History>
{chat_history}

<Follow Up Message>
{question}

<Standalone question>
"""
)


system_prompt = """ 
You are an agent designed to answer queries from user.
Please ALWAYS use the tools provided to answer a question. Do not rely on prior knowledge.
If there is no information please answer you don't have that information.
"""


query_text_description = """
Useful for querying for information
from text documents about {file_description}
"""


query_sql_description = """
Useful for translating a natural language query 
into an SQL query over table {description}
"""


clarifying_template = """
Q: {question}
A: {answer}
"""


rewrite_query = """
Given a query and a set of clarifying questions, please rewrite the query to be more clear.
Example:
Q: What trajectory is the monthly earning from the three months: April, May and June?
Clarifying Questions:
   Q: What year are you referring to?
   A: In 2022
   Q: What company are you referring to?
   A: Uber
Rewrite: What was the trajectory of Uber's monthly earnings for the months of April, May, and June in 2022?

Q:{orig_question}
Clarifying Questions: {clarifying_texts}
Rewrite: 
"""
