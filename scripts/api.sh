#!/bin/zsh
uvicorn api:app --loop asyncio $1
