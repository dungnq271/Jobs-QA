#!/bin/zsh
uvicorn src.chat:app --loop asyncio --reload
