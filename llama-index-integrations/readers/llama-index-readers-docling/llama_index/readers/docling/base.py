from pathlib import Path
from typing import Any, Iterable, Literal

from docling.document_converter import DocumentConverter
from docling_core.transforms.id_generator import BaseIDGenerator, DocHashIDGenerator
from docling_core.transforms.metadata_extractor import (
    BaseMetadataExtractor,
    SimpleMetadataExtractor,
)
from llama_index.core.readers.base import BasePydanticReader
from llama_index.core import Document as LIDocument
from pydantic import Field


class DoclingPDFReader(BasePydanticReader):
    """Docling PDF Reader.

    Extracts PDF into LlamaIndex documents either as Markdown or JSON-serialized Docling native format.

    Args:
        export_type (Literal["markdown", "json"], optional): The type to export to. Defaults to "markdown".
        doc_converter (DocumentConverter, optional): The Docling converter to use. Default factory: `DocumentConverter`.
        doc_id_generator (BaseIDGenerator | None, optional): The document ID generator to use. Setting to `None` falls back to LlamaIndex's default ID generation. Defaults to `DocHashIDGenerator()`.
        metadata_extractor (BaseMetadataExtractor | None, optional): The document metadata extractor to use. Setting to `None` skips doc metadata extraction. Defaults to `SimpleMetadataExtractor()`.
    """

    export_type: Literal["markdown", "json"] = "markdown"
    doc_converter: DocumentConverter = Field(default_factory=DocumentConverter)
    doc_id_generator: BaseIDGenerator | None = DocHashIDGenerator()
    metadata_extractor: BaseMetadataExtractor | None = SimpleMetadataExtractor()

    def lazy_load_data(
        self,
        file_path: str | Path | Iterable[str] | Iterable[Path],
        *args: Any,
        **load_kwargs: Any,
    ) -> Iterable[LIDocument]:
        """Lazily load PDF data from given source.

        Args:
            file_path (str | Path | Iterable[str] | Iterable[Path]): PDF file source as single str (URL or local file) or pathlib.Path — or iterable thereof

        Returns:
            Iterable[LIDocument]: Iterable over the created LlamaIndex documents
        """
        file_paths = (
            file_path
            if isinstance(file_path, Iterable) and not isinstance(file_path, str)
            else [file_path]
        )

        for source in file_paths:
            dl_doc = self.doc_converter.convert_single(source).output
            text = (
                dl_doc.export_to_markdown()
                if self.export_type == "markdown"
                else dl_doc.model_dump_json()
            )

            origin = str(source) if isinstance(source, Path) else source
            doc_kwargs = {}
            if self.doc_id_generator:
                doc_kwargs["doc_id"] = self.doc_id_generator.generate_id(doc=dl_doc)
            if self.metadata_extractor:
                doc_kwargs[
                    "excluded_embed_metadata_keys"
                ] = self.metadata_extractor.get_excluded_embed_metadata_keys()
                doc_kwargs[
                    "excluded_llm_metadata_keys"
                ] = self.metadata_extractor.get_excluded_llm_metadata_keys()
            li_doc = LIDocument(
                text=text,
                **doc_kwargs,
            )
            if self.metadata_extractor:
                li_doc.metadata = self.metadata_extractor.get_metadata(
                    doc=dl_doc,
                    origin=origin,
                )
            yield li_doc
