from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Literal

from docling.document_converter import DocumentConverter
from docling_core.transforms.id_generator import BaseIDGenerator, DocHashIDGenerator
from docling_core.transforms.metadata_extractor import (
    BaseMetadataExtractor,
    SimpleMetadataExtractor,
)
from llama_index.core.readers.base import BasePydanticReader
from llama_index.core.schema import Document as LIDocument

from llama_index.core import Document as LIDocument


class DoclingPDFReader(BasePydanticReader):
    export_type: Literal["markdown", "json"] = "markdown"
    metadata_extractor: BaseMetadataExtractor | None = SimpleMetadataExtractor()
    doc_id_generator: BaseIDGenerator = DocHashIDGenerator()
    chunk_docs: bool = True

    class _Keys(str, Enum):
        EXCL_EMBED_META_KEYS = "excluded_embed_metadata_keys"
        EXCL_LLM_META_KEYS = "excluded_llm_metadata_keys"

    def lazy_load_data(
        self,
        file_path: str | Path | Iterable[str] | Iterable[Path],
        *args: Any,
        **load_kwargs: Any,
    ) -> Iterable[LIDocument]:
        converter = DocumentConverter()
        file_paths = (
            file_path
            if isinstance(file_path, Iterable) and not isinstance(file_path, str)
            else [file_path]
        )

        for source in file_paths:
            dl_doc = converter.convert_single(source).output
            text = (
                dl_doc.export_to_markdown()
                if self.export_type == "markdown"
                else dl_doc.model_dump_json()
            )

            _source = str(source) if isinstance(source, Path) else source
            doc_id = self.doc_id_generator.generate_id(doc=dl_doc)
            doc_kwargs = (
                {
                    "excluded_embed_metadata_keys": self.metadata_extractor.get_excluded_embed_metadata_keys(),
                    "excluded_llm_metadata_keys": self.metadata_extractor.get_excluded_llm_metadata_keys(),
                }
                if self.metadata_extractor
                else {}
            )
            li_doc = LIDocument(
                doc_id=doc_id,
                text=text,
                **doc_kwargs,
            )
            if self.metadata_extractor:
                li_doc.metadata = self.metadata_extractor.get_metadata(
                    doc=dl_doc,
                    origin=_source,
                )
            yield li_doc
