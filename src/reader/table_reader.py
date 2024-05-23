import pandas as pd
from fsspec.implementations.local import LocalFileSystem
from llama_index.core import Document
from llama_index.core.readers.file.base import default_file_metadata_func
from pandas import DataFrame
from sqlalchemy import text

from .base_reader import BaseReader

fs = LocalFileSystem()


class TableReader(BaseReader):
    def __init__(self, db_engine):
        self.db_engine = db_engine

    def preprocess_dataframe(self, df: DataFrame, metadata: dict, **kwargs):
        if metadata.get("renamed_column"):
            df = df.rename(columns=metadata["renamed_column"])
        if metadata.get("column_map_function"):
            for col, func in metadata["column_map_function"].items():
                df[col] = df[col].apply(lambda x: func(x))
        return df

    def preprocess_metadata(self, df: DataFrame, metadata: dict, **kwargs):
        metadata["all_columns"] = list(metadata["column_description"].keys())
        return metadata

    def add_table_to_db_engine(self, df: DataFrame, metadata: dict):
        table_name = metadata["table_name"]
        df.to_sql(table_name, self.db_engine)  # add into database

    def get_documents(self, metadata: dict, file_metadata: dict):
        with self.db_engine.connect() as conn:
            cursor = conn.execute(text(f'SELECT * FROM "{metadata["table_name"]}"'))
            result = cursor.fetchall()

        documents = []
        for row in result:
            # idx, infos = row[0], row[1:]
            infos = row[1:]

            info_strs = []

            column_descriptions = metadata["column_description"]
            column_names = column_descriptions.keys()

            for name, value in zip(column_names, infos, strict=False):
                column_description = column_descriptions[name]["description"] or name
                info_strs.append(f"{column_description}: {value}")

            all_info_str = "\n".join(info_strs)
            document = Document(text=all_info_str)

            # add file metadata
            document.metadata.update(file_metadata)
            document.excluded_embed_metadata_keys.extend(
                [
                    "file_path",
                    "file_name",
                    "file_type",
                    "file_size",
                    "creation_date",
                    "last_modified_date",
                    "last_accessed_date",
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
                ]
            )

            # add to documents
            documents.append(document)

        return documents

    def load_data(
        self,
        file_path: str,
        metadata: dict,
        table: pd.DataFrame | None = None,
        **kwargs,
    ):
        if table is None:
            table = pd.read_csv(file_path)

        table = self.preprocess_dataframe(table, metadata)
        self.add_table_to_db_engine(df=table, metadata=metadata)
        metadata = self.preprocess_metadata(df=table, metadata=metadata)

        file_metadata = default_file_metadata_func(file_path=file_path, fs=fs)

        documents = self.get_documents(metadata=metadata, file_metadata=file_metadata)
        return documents
