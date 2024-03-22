#!/bin/zsh
uvicorn src.agent:app --loop asyncio --reload
