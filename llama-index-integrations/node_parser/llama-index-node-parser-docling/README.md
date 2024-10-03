# Docling Node Parser

## Overview

Docling Node Parser parses [Docling](https://github.com/DS4SD/docling)'s native format (JSON-serialized) into LlamaIndex nodes with rich metadata for usage in downstream pipelines for RAG / QA etc.

## Installation

```console
pip install llama-index-node-parser-docling
```

## Usage

Docling Node Parser parses LlamaIndex documents containing JSON-serialized Docling format, for example as produced from a Docling PDF Reader.

Basic usage looks like this:

```python
from llama_index.readers.docling import DoclingPDFReader
from llama_index.node_parser.docling import DoclingNodeParser

reader = DoclingPDFReader(export_type="json")
docs = reader.load_data(file_path="https://arxiv.org/pdf/2408.09869")
print(f"{docs[0].text[:50]}...")
# > {"_name":"","type":"pdf-document","description":{"...

node_parser = DoclingNodeParser()
nodes = node_parser.get_nodes_from_documents(documents=docs)
print(f"{nodes[6].text[:70]}...")
# > Docling provides an easy code interface to convert PDF documents from ...

print(nodes[6].metadata)
# > {'dl_doc_hash': '556ad9e23b...',
# >  'path': '#/main-text/22',
# >  'heading': '2 Getting Started',
# >  'page': 2,
# >  'bbox': [107.40, 456.93, 504.20, 499.65]}
```
