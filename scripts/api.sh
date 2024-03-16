#!/bin/zsh
uvicorn src.llama_index_agent:app --loop asyncio --reload
