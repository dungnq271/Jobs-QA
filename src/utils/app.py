import asyncio
import os
from typing import Any, List

from . import api
from .io import save_file


async def add_file(
    document_dir_path: str, uploaded_file: Any, existing_files: List[str]
):
    new_files = []
    filename = uploaded_file.name
    if filename not in existing_files:
        filepath = os.path.join(document_dir_path, os.path.basename(filename))
        save_file(uploaded_file.read(), filepath)
        existing_files.append(filename)
        await api.add_document(filepath)
        new_files.append(filename)
    return new_files


async def add_documents_process(
    document_dir_path: str, uploaded_files: List[Any], documents: List[str]
):
    new_files = await asyncio.gather(
        *[add_file(document_dir_path, uf, documents) for uf in uploaded_files]
    )
    documents.extend(new_files)


async def add_tools_process(tools: List[str]):
    await api.add_api_tools(tools)


async def remove_tools_process(tools: List[str]):
    await api.remove_api_tools(tools)
