from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Sequence

from docling.document_converter import DocumentConverter
from docling_core.transforms.id_generator import BaseIDGenerator, DocHashIDGenerator
from docling_core.transforms.metadata_extractor import (
    BaseMetadataExtractor,
    SimpleMetadataExtractor,
)
from llama_index.core.readers.base import BasePydanticReader
from llama_index.core.schema import Document as LIDocument
from llama_index.core.node_parser import NodeParser

from docling_core.transforms.chunker import BaseChunker
from docling_core.transforms.chunker import ChunkWithMetadata, HierarchicalChunker
from docling_core.types import Document as DLDocument
from llama_index.core import Document as LIDocument
from llama_index.core.node_parser import MarkdownNodeParser, NodeParser
from llama_index.core.schema import (
    BaseNode,
    NodeRelationship,
    RelatedNodeType,
    TextNode,
)
from llama_index.core.utils import get_tqdm_iterable


class DocMetaKeys(str, Enum):
    DL_DOC_HASH = "dl_doc_hash"
    ORIGIN = "origin"


class NodeMetaKeys(str, Enum):
    PATH = "path"
    PAGE = "page"
    BBOX = "bbox"
    ORIGIN = "origin"
    HEADING = "heading"


class DoclingJSONNodeParser(NodeParser):
    chunker: BaseChunker = HierarchicalChunker(heading_as_metadata=True)

    def _parse_nodes(
        self,
        nodes: Sequence[BaseNode],
        show_progress: bool = False,
        **kwargs: Any,
    ) -> list[BaseNode]:
        nodes_with_progress: Iterable[BaseNode] = get_tqdm_iterable(
            items=nodes, show_progress=show_progress, desc="Parsing nodes"
        )
        all_nodes: list[BaseNode] = []
        for input_node in nodes_with_progress:
            li_doc = LIDocument.model_validate(input_node)
            dl_doc: DLDocument = DLDocument.model_validate_json(li_doc.get_content())
            chunk_iter = self.chunker.chunk(dl_doc=dl_doc)
            for chunk in chunk_iter:
                rels: dict[NodeRelationship, RelatedNodeType] = {
                    NodeRelationship.SOURCE: li_doc.as_related_node_info(),
                }
                excl_doc_meta_keys = [d.value for d in DocMetaKeys]
                excl_node_meta_keys = [
                    n.value for n in NodeMetaKeys if n not in [NodeMetaKeys.HEADING]
                ]
                excl_meta_keys = excl_doc_meta_keys + excl_node_meta_keys
                node = TextNode(
                    text=chunk.text,
                    excluded_embed_metadata_keys=excl_meta_keys,
                    excluded_llm_metadata_keys=excl_meta_keys,
                    relationships=rels,
                )
                node.metadata = {
                    NodeMetaKeys.PATH: chunk.path,
                    NodeMetaKeys.HEADING: chunk.heading,
                }
                if isinstance(chunk, ChunkWithMetadata):
                    node.metadata[NodeMetaKeys.PAGE] = chunk.page
                    node.metadata[NodeMetaKeys.BBOX] = chunk.bbox
                all_nodes.append(node)
        return all_nodes


class DoclingPDFReader(BasePydanticReader):
    class ExportType(str, Enum):
        MARKDOWN = "markdown"
        JSON = "json"

    export_type: ExportType = ExportType.JSON
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
        node_parser: NodeParser | None = None
        if self.chunk_docs:
            node_parser = (
                MarkdownNodeParser()
                if self.export_type == self.ExportType.MARKDOWN
                else DoclingJSONNodeParser()
            )

        for source in file_paths:
            dl_doc = converter.convert_single(source).output
            text = (
                dl_doc.export_to_markdown()
                if self.export_type == self.ExportType.MARKDOWN
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

            if self.chunk_docs:
                nodes = node_parser.get_nodes_from_documents([li_doc])
                yield from nodes
            else:
                yield li_doc
