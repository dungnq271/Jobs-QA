#!/bin/zsh
uvicorn src.table_factory:app --loop asyncio $1
