from llama_index.core import PromptTemplate

from utils import modify_days_to_3digits

# Prompt Template
custom_prompt = PromptTemplate(
    """
Given a conversation (between Human and Assistant)
and a follow up message from Human,
rewrite the message to be a standalone question
that captures all relevant context from the conversation.

<Chat History>
{chat_history}

<Follow Up Message>
{question}

<Standalone question>
"""
)


system_prompt = """
You are an agent designed to answer queries from user.
Please ALWAYS use the tools provided to answer a question.
Do not rely on prior knowledge.
If there is no information please answer you don't have that information.
"""


tool_description = """
Useful for answering questions about {file_description}
"""


query_text_description = """
Useful for answering semantic questions about {file_description}
"""


query_sql_description = """
Useful for translating a natural language query
into an SQL query over table having columns: {columns_list}
"""


clarifying_template = """
Q: {question}
A: {answer}
"""


rewrite_query = """
Given a query and a set of clarifying questions,
please rewrite the query to be more clear.
Example:
Q: What trajectory is the monthly earning
from the three months: April, May and June?
Clarifying Questions:
   Q: What year are you referring to?
   A: In 2022
   Q: What company are you referring to?
   A: Uber
Rewrite: What was the trajectory of
Uber's monthly earnings for the months of
April, May, and June in 2022?

Q:{orig_question}
Clarifying Questions: {clarifying_texts}
Rewrite:
"""


# File Metadata
files_metadata = {
    "./documents/job_vn_posted_full_recent_v2.csv": {
        "table_name": "jobPosted",
        "table_desc": "different AI jobs information at different companies",
        "renamed_cols": {"Posted": "Number_of_days_posted_ago"},
        "apply_col_funcs": {
            "Number_of_days_posted_ago": modify_days_to_3digits
        },
        "chosen_cols_descs": [
            {
                "name": "Number_of_days_posted_ago",
                "type": "str",
                "description": "The number of days ago the job was posted",
            },
            {
                "name": "Full / Part Time",
                "type": "str",
                "description": "Working time for the job",
            },
            {
                "name": "Salary",
                "type": "str",
                "description": "Job's pay range",
            },
            {
                "name": "Link",
                "type": "str",
                "description": "Link to the posted job",
            },
        ],
        "exc_cols": ["Links"],
    },
    "uber_10q_march_2022": {
        "table_desc": "containing financial information"
        " about UBER TECHNOLOGIES, INC.",
    },
    "allPlayersLookup": {
        "table_desc": "containing biographical information"
        " on NHL hockey players",
    },
    "all_teams": {
        "table_desc": "containing stats related to NHL hockey teams",
    },
}
