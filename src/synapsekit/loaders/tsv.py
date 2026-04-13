from __future__ import annotations

import os
import csv

from .base import Document

class TSVLoader:
    """Load tab-separated values (TSV) from a file path."""

    def __init__(self, path: str, encoding: str = "utf-8") -> None:
        self._path = path
        self._encoding = encoding

    def load(self):
        if not os.path.exists(self._path):
            raise FileNotFoundError(f"File not found: {self._path}")

        documents = []

        with open(self._path, encoding=self._encoding) as f:
            reader = csv.reader(f, delimiter="\t")

            for row in reader:
                text = " ".join(row)

                documents.append(
                    Document(text=text, metadata={"source": self._path})
                )

        return documents