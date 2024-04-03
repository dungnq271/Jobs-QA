# %%
import sys

sys.path.append("../agent_api.py")

from agent_api import *

# %%
agent = Agent(
    mode=mode,
    collection_name=table_name,
    node_parser=MarkdownElementNodeParser(
        llm=llm,
        num_workers=4,
    ),
    reranker=SimilarityPostprocessor(similarity_cutoff=0.5),
)

# %%
filepaths = [
    "../documents/all_teams.csv",
    "../documents/allPlayersLookup.csv",
    "../documents/uber_10q_march_2022.pdf",
]
for path in filepaths:
    agent.parse_file(path)

# %%
agent.chat("Who is the oldest player?")

# %%
agent.chat("Which team has the most points?")

# %%
agent.chat(
    "what is the net loss value attributable to Uber compared to last year?"
)

# %%
