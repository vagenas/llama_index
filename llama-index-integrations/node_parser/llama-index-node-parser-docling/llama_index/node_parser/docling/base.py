from enum import Enum
from typing import Any, Iterable, Sequence

from llama_index.core.schema import Document as LIDocument
from llama_index.core.node_parser import NodeParser

from docling_core.transforms.chunker import BaseChunker
from docling_core.transforms.chunker import ChunkWithMetadata, HierarchicalChunker
from docling_core.types import Document as DLDocument
from llama_index.core import Document as LIDocument
from llama_index.core.node_parser import NodeParser
from llama_index.core.schema import (
    BaseNode,
    NodeRelationship,
    RelatedNodeType,
    TextNode,
)
from llama_index.core.utils import get_tqdm_iterable


class _DocMetaKeys(str, Enum):
    DL_DOC_HASH = "dl_doc_hash"
    ORIGIN = "origin"


class _NodeMetaKeys(str, Enum):
    PATH = "path"
    PAGE = "page"
    BBOX = "bbox"
    ORIGIN = "origin"
    HEADING = "heading"


class DoclingNodeParser(NodeParser):
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
                excl_doc_meta_keys = [d.value for d in _DocMetaKeys]
                excl_node_meta_keys = [
                    n.value for n in _NodeMetaKeys if n not in [_NodeMetaKeys.HEADING]
                ]
                excl_meta_keys = excl_doc_meta_keys + excl_node_meta_keys
                node = TextNode(
                    text=chunk.text,
                    excluded_embed_metadata_keys=excl_meta_keys,
                    excluded_llm_metadata_keys=excl_meta_keys,
                    relationships=rels,
                )
                node.metadata = {
                    _NodeMetaKeys.PATH: chunk.path,
                    _NodeMetaKeys.HEADING: chunk.heading,
                }
                if isinstance(chunk, ChunkWithMetadata):
                    node.metadata[_NodeMetaKeys.PAGE] = chunk.page
                    node.metadata[_NodeMetaKeys.BBOX] = chunk.bbox
                all_nodes.append(node)
        return all_nodes
