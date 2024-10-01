from enum import Enum
from pathlib import Path
from typing import Any, Iterable

from docling.document_converter import DocumentConverter
from docling_core.transforms.id_generator import BaseIDGenerator, DocHashIDGenerator
from docling_core.transforms.metadata_extractor import (
    BaseMetadataExtractor,
    SimpleMetadataExtractor,
)
from llama_index.core.readers.base import BasePydanticReader
from llama_index.core.schema import Document as LIDocument


class DoclingPDFReader(BasePydanticReader):
    class ExportType(str, Enum):
        MARKDOWN = "markdown"
        JSON = "json"

    class _Keys(str, Enum):
        EXCL_EMBED_META_KEYS = "excluded_embed_metadata_keys"
        EXCL_LLM_META_KEYS = "excluded_llm_metadata_keys"

    export_type: ExportType = ExportType.MARKDOWN
    metadata_extractor: BaseMetadataExtractor | None = SimpleMetadataExtractor()
    doc_id_generator: BaseIDGenerator = DocHashIDGenerator()

    def lazy_load_data(
        self,
        file_path: str | Path | Iterable[str] | Iterable[Path],
        *args: Any,
        **load_kwargs: Any,
    ) -> Iterable[LIDocument]:
        file_paths = (
            file_path
            if isinstance(file_path, Iterable) and not isinstance(file_path, str)
            else [file_path]
        )
        converter = DocumentConverter()
        for source in file_paths:
            dl_doc = converter.convert_single(source).output
            match self.export_type:
                case self.ExportType.MARKDOWN:
                    text = dl_doc.export_to_markdown()
                case self.ExportType.JSON:
                    text = dl_doc.model_dump_json()
                case _:
                    raise RuntimeError(
                        f"Unexpected export type encountered: {self.export_type}"
                    )
            doc_kwargs = {}
            _source = str(source) if isinstance(source, Path) else source
            doc_id = self.doc_id_generator.generate_id(doc=dl_doc)
            if self.metadata_extractor:
                doc_kwargs[
                    self._Keys.EXCL_EMBED_META_KEYS
                ] = self.metadata_extractor.get_excluded_embed_metadata_keys()
                doc_kwargs[
                    self._Keys.EXCL_LLM_META_KEYS
                ] = self.metadata_extractor.get_excluded_llm_metadata_keys()
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
