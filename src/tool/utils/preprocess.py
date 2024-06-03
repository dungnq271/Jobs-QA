from typing import Any, Dict

from src.prompt import DEFAULT_TABLE_DESCRIPTION_TMPL


def preprocess_tool_description(
    desciption_template: str,
    tool_type: str,
    metadata: Dict[str, Any],
):
    if tool_type == "sql":
        return preprocess_sql_tool_description(desciption_template, metadata)
    else:
        return preprocess_retriever_tool_description(desciption_template, metadata)


def preprocess_sql_tool_description(
    description_template: str,
    metadata: Dict[str, Any],
):
    column_descs = metadata["column_description"]
    all_column_descs_str = "\n".join(
        [
            f"{col}: {(meta['description'] if meta['description'] else '')}"
            for col, meta in column_descs.items()
        ]
    )
    table_description = DEFAULT_TABLE_DESCRIPTION_TMPL.format(
        table_name=metadata["table_name"],
        table_description=metadata["file_description"],
        column_descriptions=all_column_descs_str,
    )

    tool_description = description_template.format(table_description=table_description)
    return tool_description


def preprocess_retriever_tool_description(
    description_template: str,
    metadata: Dict[str, Any],
):
    # tool_description = description_template.format(
    #     file_description=metadata["file_description"]
    # )
    tool_description = description_template
    return tool_description
