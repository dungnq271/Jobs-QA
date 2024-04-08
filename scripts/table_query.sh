#!/bin/zsh
uvicorn src.table_query:app --loop asyncio
