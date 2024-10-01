# Docling PDF Reader

This package provides a simple yet powerful integration of [Docling](https://github.com/DS4SD/docling) PDF document conversion to LlamaIndex.

## Installation

```console
pip install llama-index-readers-docling
```

## Usage

Basic usage looks as below:

```console
from llama_index.readers.docling.base import DoclingPDFReader

reader = DoclingPDFReader()
docs = reader.load_data(file_path="https://arxiv.org/pdf/2408.09869")
```
