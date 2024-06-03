from typing import Dict, List

import pandas as pd
from fsspec.implementations.local import LocalFileSystem
from llama_index.core import Document
from llama_index.core.readers.file.base import default_file_metadata_func
from pandas import DataFrame
from sqlalchemy import text

from src.reader.base_reader import BaseReader

fs = LocalFileSystem()


class TableReader(BaseReader):
    def __init__(self, db_engine, use_tool_types: List[str] = ["sql", "recursive"]):
        self.db_engine = db_engine
        self.use_tool_types = use_tool_types

    def preprocess_dataframe(self, df: DataFrame, metadata: Dict, **kwargs):
        if metadata.get("renamed_column"):
            df = df.rename(columns=metadata["renamed_column"])
        if metadata.get("column_map_function"):
            for col, func in metadata["column_map_function"].items():
                df[col] = df[col].apply(lambda x, func=func: func(x))
        return df

    def preprocess_metadata(self, df: DataFrame, metadata: Dict, **kwargs):
        metadata["all_columns"] = list(metadata["column_description"].keys())
        return metadata

    def add_table_to_db_engine(self, df: DataFrame, metadata: Dict):
        table_name = metadata["table_name"]
        df.to_sql(table_name, self.db_engine)  # add into database

    def get_documents(self, metadata: Dict, file_metadata: Dict, **kwargs):
        with self.db_engine.connect() as conn:
            cursor = conn.execute(text(f'SELECT * FROM "{metadata["table_name"]}"'))
            result = cursor.fetchall()

        documents = []
        for row in result:
            idx, infos = row[0], row[1:]
            # infos = row[1:]

            info_strs = [f"Row: {idx}"]

            column_descriptions = metadata["column_description"]
            column_names = column_descriptions.keys()

            for name, value in zip(column_names, infos, strict=False):
                column_description = column_descriptions[name]["description"] or name
                info_strs.append(f"{column_description}: {value}")

            all_info_str = "\n".join(info_strs)
            document = Document(text=all_info_str)

            # add file metadata
            document.metadata.update(file_metadata)
            document.metadata.update({"use_tool_types": self.use_tool_types})
            document.excluded_embed_metadata_keys.extend(
                [
                    "file_path",
                    "file_name",
                    "file_type",
                    "file_size",
                    "creation_date",
                    "last_modified_date",
                    "last_accessed_date",
                    "use_tool_types",
                ]
            )
            document.excluded_llm_metadata_keys.extend(
                [
                    "file_path",
                    "file_name",
                    "file_type",
                    "file_size",
                    "creation_date",
                    "last_modified_date",
                    "last_accessed_date",
                    "use_tool_types",
                ]
            )

            # add to documents
            documents.append(document)

        return documents

    def load_data(self, file_path: str, metadata: Dict, **kwargs):
        df = pd.read_csv(file_path)
        df = self.preprocess_dataframe(df, metadata)

        self.add_table_to_db_engine(df=df, metadata=metadata)
        metadata = self.preprocess_metadata(df=df, metadata=metadata)

        file_metadata = default_file_metadata_func(file_path=file_path, fs=fs)

        documents = self.get_documents(
            metadata=metadata, file_metadata=file_metadata, **kwargs
        )
        return documents
