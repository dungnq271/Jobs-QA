from typing import Dict, List

from llama_index.core import SimpleDirectoryReader
from llama_parse import LlamaParse

from .base_reader import BaseReader

# Currently, Llamaparser only the following file types are supported:
all_supported_formats = [
    ".pdf",
    ".doc",
    ".docx",
    ".docm",
    ".dot",
    ".dotx",
    ".dotm",
    ".rtf",
    ".wps",
    ".wpd",
    ".sxw",
    ".stw",
    ".sxg",
    ".pages",
    ".mw",
    ".mcw",
    ".uot",
    ".uof",
    ".uos",
    ".uop",
    ".ppt",
    ".pptx",
    ".pot",
    ".pptm",
    ".potx",
    ".potm",
    ".key",
    ".odp",
    ".odg",
    ".otp",
    ".fopd",
    ".sxi",
    ".sti",
    ".epub",
    ".html",
    ".htm",
]


supported_formats = [
    ".pdf",
    ".pptx",
    ".ppt",
    ".pptm",
    ".docx",
    ".doc",
    ".docm",
    ".dot",
    ".dotx",
    ".dotm",
    ".epub",
    ".html",
    ".htm",
]


parser = LlamaParse(
    # can also be set in your env as LLAMA_CLOUD_API_KEY
    api_key="llx-BURYg70IGnuz1GD1wOCBVYt64rS3f5bASLKKxbkywVX92KRG",
    result_type="text",  # "markdown" and "text" are available,
    num_workers=8,
    max_timeout=600,
)


class DocumentReader(BaseReader):
    def __init__(
        self,
        supported_formats: List[str] = supported_formats,
        use_tool_types: List[str] = ["normal"],
    ):
        self.supported_formats = supported_formats
        self.use_tool_types = use_tool_types

        # if file_paths is not None and len(file_paths) > 0:
        #     for i in range(len(file_paths)):
        #         self.add_single_file(file_path=file_paths[i])

    def load_data(self, file_path: str, metadata: Dict, **kwargs):
        # if not os.path.exists(file_path):
        #     sys.stderr.write("File not found\n")
        #     exit(1)

        documents = SimpleDirectoryReader(
            input_files=[file_path],
            file_extractor={fmt: parser for fmt in self.supported_formats},
            # required_exts=self.supported_formats,
        ).load_data()

        for document in documents:
            document.metadata.update({"use_tool_types": self.use_tool_types})
            document.excluded_embed_metadata_keys.append("use_tool_types")
            document.excluded_llm_metadata_keys.append("use_tool_types")

        return documents
