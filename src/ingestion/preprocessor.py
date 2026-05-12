"""Cleans raw text before chunking."""

import re
from typing import List

from langchain_core.documents import Document


def clean_text(text: str) -> str:
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" \n", "\n", text)
    return text.strip()


def preprocess_documents(documents: List[Document]) -> List[Document]:
    cleaned = []
    for doc in documents:
        content = clean_text(doc.page_content)
        if len(content) > 20:  # drop near-empty pages
            cleaned.append(Document(page_content=content, metadata=doc.metadata))
    return cleaned
