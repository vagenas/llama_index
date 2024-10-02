# Docling PDF Reader

## Overview

Docling PDF Reader uses [Docling](https://github.com/DS4SD/docling) to enable fast and easy PDF extraction and export to Markdown or JSON (Docling's native format), for usage in LlamaIndex pipelines for RAG / QA etc.

## Installation

```console
pip install llama-index-readers-docling
```

## Usage

### Markdown export

By default, Docling PDF Reader exports to Markdown. Basic usage looks like this:

```python
from llama_index.readers.docling import DoclingPDFReader

reader = DoclingPDFReader()
docs = reader.load_data(file_path="https://arxiv.org/pdf/2408.09869")
print(f"{docs[0].text[409:462]}...")
# > ## Abstract
# >
# > This technical report introduces Docling...
```

### JSON export

Docling PDF Reader can also export Docling's native format to JSON:

```python
from llama_index.readers.docling import DoclingPDFReader

reader = DoclingPDFReader(export_type="json")
docs = reader.load_data(file_path="https://arxiv.org/pdf/2408.09869")
print(f"{docs[0].text[:50]}...")
# > {"_name":"","type":"pdf-document","description":{"...
```

> [!IMPORTANT]
> To appropriately parse Docling's native format, when using JSON export make sure
> to use a `llama_index.node_parser.docling.DoclingNodeParser` in your pipeline.

<!--
TODO add usage with SimpleDirectoryReader
https://docs.llamaindex.ai/en/stable/module_guides/loading/simpledirectoryreader/
-- >
