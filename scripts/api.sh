#!/bin/zsh
uvicorn agent_api:app --loop asyncio --reload
