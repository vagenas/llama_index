import json

from llama_index.core.schema import Document as LIDocument

from llama_index.node_parser.docling import DoclingNodeParser
from llama_index.core.schema import BaseNode

in_json_str = """{
  "name": "foo",
  "description": {
    "logs": []
  },
  "main_text": [
    {
      "text": "Test subtitle",
      "type": "subtitle-level-1",
      "name": "Section-header"
    },
    {
      "text": "This is a test paragraph.",
      "type": "paragraph",
      "name": "Text"
    }
  ],
  "file-info": {
    "filename": "foo.pdf",
    "document-hash": "123"
  }
}
"""

out_get_nodes = {
    "root": [
        {
            "id_": "129210df929c78e70d74e6f141a46d8326905ce58562f2081819c80c3921d5a3_0",
            "embedding": None,
            "metadata": {
                "dl_doc_hash": "129210df929c78e70d74e6f141a46d8326905ce58562f2081819c80c3921d5a3",
                "path": "#/main-text/0",
                "page": 1,
                "bbox": [1.0, 2.0, 3.0, 4.0],
            },
            "excluded_embed_metadata_keys": ["dl_doc_hash", "path", "page", "bbox"],
            "excluded_llm_metadata_keys": ["dl_doc_hash", "path", "page", "bbox"],
            "relationships": {
                "1": {
                    "node_id": "129210df929c78e70d74e6f141a46d8326905ce58562f2081819c80c3921d5a3",
                    "node_type": "4",
                    "metadata": {
                        "dl_doc_hash": "129210df929c78e70d74e6f141a46d8326905ce58562f2081819c80c3921d5a3"
                    },
                    "hash": "4a59a6b249fa5485206e49ee7a10be02d810e3ca1179b14ce23d5bb83ec33e63",
                    "class_name": "RelatedNodeInfo",
                },
                "3": {
                    "node_id": "129210df929c78e70d74e6f141a46d8326905ce58562f2081819c80c3921d5a3_1",
                    "node_type": "1",
                    "metadata": {
                        "path": "#/main-text/1",
                        "page": 1,
                        "bbox": [1.0, 2.0, 3.0, 4.0],
                    },
                    "hash": "fc3edef366333e99ad544bbc0208289b85aacc00c5d8b3b868eae720b11573ef",
                    "class_name": "RelatedNodeInfo",
                },
            },
            "text": "A duckling is a young duck in downy plumage[1] or baby duck,[2] but in the food trade a young domestic duck which has just reached adult size and bulk and its meat is still fully tender, is sometimes labelled as a duckling.",
            "mimetype": "text/plain",
            "start_char_idx": 649,
            "end_char_idx": 872,
            "text_template": "{metadata_str}\n\n{content}",
            "metadata_template": "{key}: {value}",
            "metadata_seperator": "\n",
            "class_name": "TextNode",
        },
        {
            "id_": "129210df929c78e70d74e6f141a46d8326905ce58562f2081819c80c3921d5a3_1",
            "embedding": None,
            "metadata": {
                "dl_doc_hash": "129210df929c78e70d74e6f141a46d8326905ce58562f2081819c80c3921d5a3",
                "path": "#/main-text/1",
                "page": 1,
                "bbox": [1.0, 2.0, 3.0, 4.0],
            },
            "excluded_embed_metadata_keys": ["dl_doc_hash", "path", "page", "bbox"],
            "excluded_llm_metadata_keys": ["dl_doc_hash", "path", "page", "bbox"],
            "relationships": {
                "1": {
                    "node_id": "129210df929c78e70d74e6f141a46d8326905ce58562f2081819c80c3921d5a3",
                    "node_type": "4",
                    "metadata": {
                        "dl_doc_hash": "129210df929c78e70d74e6f141a46d8326905ce58562f2081819c80c3921d5a3"
                    },
                    "hash": "4a59a6b249fa5485206e49ee7a10be02d810e3ca1179b14ce23d5bb83ec33e63",
                    "class_name": "RelatedNodeInfo",
                },
                "2": {
                    "node_id": "129210df929c78e70d74e6f141a46d8326905ce58562f2081819c80c3921d5a3_0",
                    "node_type": "1",
                    "metadata": {
                        "dl_doc_hash": "129210df929c78e70d74e6f141a46d8326905ce58562f2081819c80c3921d5a3",
                        "path": "#/main-text/0",
                        "page": 1,
                        "bbox": [1.0, 2.0, 3.0, 4.0],
                    },
                    "hash": "d9ce62fa272d91859825834c0a9068c4a3033be1d4b6d4e1a830420b420d7fa2",
                    "class_name": "RelatedNodeInfo",
                },
            },
            "text": "A male is called a drake and the female is called a duck, or in ornithology a hen.",
            "mimetype": "text/plain",
            "start_char_idx": 1008,
            "end_char_idx": 1090,
            "text_template": "{metadata_str}\n\n{content}",
            "metadata_template": "{key}: {value}",
            "metadata_seperator": "\n",
            "class_name": "TextNode",
        },
    ]
}


out_parse_nodes = {
    "root": [
        {
            "id_": "129210df929c78e70d74e6f141a46d8326905ce58562f2081819c80c3921d5a3_0",
            "embedding": None,
            "metadata": {
                "path": "#/main-text/0",
                "page": 1,
                "bbox": [1.0, 2.0, 3.0, 4.0],
            },
            "excluded_embed_metadata_keys": ["dl_doc_hash", "path", "page", "bbox"],
            "excluded_llm_metadata_keys": ["dl_doc_hash", "path", "page", "bbox"],
            "relationships": {
                "1": {
                    "node_id": "129210df929c78e70d74e6f141a46d8326905ce58562f2081819c80c3921d5a3",
                    "node_type": "4",
                    "metadata": {
                        "dl_doc_hash": "129210df929c78e70d74e6f141a46d8326905ce58562f2081819c80c3921d5a3"
                    },
                    "hash": "4a59a6b249fa5485206e49ee7a10be02d810e3ca1179b14ce23d5bb83ec33e63",
                    "class_name": "RelatedNodeInfo",
                }
            },
            "text": "A duckling is a young duck in downy plumage[1] or baby duck,[2] but in the food trade a young domestic duck which has just reached adult size and bulk and its meat is still fully tender, is sometimes labelled as a duckling.",
            "mimetype": "text/plain",
            "start_char_idx": None,
            "end_char_idx": None,
            "text_template": "{metadata_str}\n\n{content}",
            "metadata_template": "{key}: {value}",
            "metadata_seperator": "\n",
            "class_name": "TextNode",
        },
        {
            "id_": "129210df929c78e70d74e6f141a46d8326905ce58562f2081819c80c3921d5a3_1",
            "embedding": None,
            "metadata": {
                "path": "#/main-text/1",
                "page": 1,
                "bbox": [1.0, 2.0, 3.0, 4.0],
            },
            "excluded_embed_metadata_keys": ["dl_doc_hash", "path", "page", "bbox"],
            "excluded_llm_metadata_keys": ["dl_doc_hash", "path", "page", "bbox"],
            "relationships": {
                "1": {
                    "node_id": "129210df929c78e70d74e6f141a46d8326905ce58562f2081819c80c3921d5a3",
                    "node_type": "4",
                    "metadata": {
                        "dl_doc_hash": "129210df929c78e70d74e6f141a46d8326905ce58562f2081819c80c3921d5a3"
                    },
                    "hash": "4a59a6b249fa5485206e49ee7a10be02d810e3ca1179b14ce23d5bb83ec33e63",
                    "class_name": "RelatedNodeInfo",
                }
            },
            "text": "A male is called a drake and the female is called a duck, or in ornithology a hen.",
            "mimetype": "text/plain",
            "start_char_idx": None,
            "end_char_idx": None,
            "text_template": "{metadata_str}\n\n{content}",
            "metadata_template": "{key}: {value}",
            "metadata_seperator": "\n",
            "class_name": "TextNode",
        },
    ]
}


def _deterministic_id_func(i: int, doc: BaseNode) -> str:
    doc_dict = json.loads(doc.get_content())
    return f"{doc_dict['file-info']['document-hash']}_{i}"


def test_parse_nodes():
    li_doc = LIDocument.from_json(in_json_str)
    node_parser = DoclingNodeParser(
        id_func=_deterministic_id_func,
    )
    nodes = node_parser._parse_nodes(nodes=[li_doc])
    act_data = {"root": [n.model_dump() for n in nodes]}
    assert act_data == out_parse_nodes


def test_get_nodes_from_docs():
    with open(ROOT_DIR_PATH / "data" / "inp_li_doc.json") as f:
        data_json = f.read()
    li_doc = LIDocument.from_json(data_json)
    node_parser = DoclingNodeParser(
        id_func=_deterministic_id_func,
    )
    nodes = node_parser.get_nodes_from_documents(documents=[li_doc])
    act_data = {"root": [n.model_dump() for n in nodes]}
    assert act_data == out_get_nodes
