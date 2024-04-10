#!/bin/zsh
uvicorn src.react_agent:app --loop asyncio $1
