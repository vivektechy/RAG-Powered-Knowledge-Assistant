import json
import os
import re

import numpy as np


class SimpleVectorStore:

    def __init__(self, path: str):

        self.path = path

        self.ids = []
        self.embeddings = None
        self.documents = []
        self.metadatas = []

        if os.path.exists(self.path):
            self._load()

    def add(
        self,
        ids,
        embeddings,
        documents,
        metadatas
    ):

        new_emb = np.array(
            embeddings,
            dtype=np.float32
        )

        if self.embeddings is None:

            self.embeddings = new_emb

        else:

            self.embeddings = np.vstack(
                [self.embeddings, new_emb]
            )

        self.ids.extend(ids)
        self.documents.extend(documents)
        self.metadatas.extend(metadatas)

    def query(
        self,
        query_embedding,
        query_text=None,
        n_results=8
    ):

        if (
            self.embeddings is None
            or len(self.ids) == 0
        ):

            return {
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]]
            }

        q = np.array(
            query_embedding,
            dtype=np.float32
        )

        emb_norms = np.linalg.norm(
            self.embeddings,
            axis=1
        )

        q_norm = np.linalg.norm(q)

        cosine = (
            self.embeddings @ q
        ) / (
            emb_norms * q_norm + 1e-10
        )

        # --------------------------------
        # Keyword / exact-match score
        # --------------------------------

        keyword_scores = np.zeros(
            len(self.documents),
            dtype=np.float32
        )

        if query_text:

            query_tokens = set(
                re.findall(
                    r"[a-zA-Z0-9%]+",
                    query_text.lower()
                )
            )

            for i, document in enumerate(
                self.documents
            ):

                doc_tokens = set(
                    re.findall(
                        r"[a-zA-Z0-9%]+",
                        document.lower()
                    )
                )

                if query_tokens:

                    overlap = (
                        query_tokens
                        & doc_tokens
                    )

                    keyword_scores[i] = (
                        len(overlap)
                        / len(query_tokens)
                    )

        # --------------------------------
        # Hybrid score
        # --------------------------------

        scores = (
            0.70 * cosine
            + 0.30 * keyword_scores
        )

        # --------------------------------
        # First-page boost for identity
        # questions
        # --------------------------------

        identity_words = {
            "name",
            "candidate",
            "person",
            "resume",
            "cv"
        }

        query_lower = (
            query_text.lower()
            if query_text
            else ""
        )

        asks_identity = any(
            word in query_lower
            for word in identity_words
        )

        if asks_identity:

            for i, metadata in enumerate(
                self.metadatas
            ):

                page = metadata.get(
                    "page",
                    0
                )

                if page == 1:

                    scores[i] += 0.12

        # --------------------------------
        # Sort
        # --------------------------------

        top_idx = np.argsort(
            -scores
        )[:n_results]

        docs = [
            self.documents[i]
            for i in top_idx
        ]

        metas = [
            self.metadatas[i]
            for i in top_idx
        ]

        distances = [
            float(
                1 - cosine[i]
            )
            for i in top_idx
        ]

        return {
            "documents": [docs],
            "metadatas": [metas],
            "distances": [distances]
        }

    def persist(self):

        data = {
            "ids": self.ids,
            "embeddings": (
                self.embeddings.tolist()
                if self.embeddings is not None
                else []
            ),
            "documents": self.documents,
            "metadatas": self.metadatas,
        }

        directory = os.path.dirname(
            self.path
        )

        if directory:
            os.makedirs(
                directory,
                exist_ok=True
            )

        with open(
            self.path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                data,
                f
            )

    def _load(self):

        with open(
            self.path,
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)

        self.ids = data["ids"]

        self.embeddings = (
            np.array(
                data["embeddings"],
                dtype=np.float32
            )
            if data["embeddings"]
            else None
        )

        self.documents = data["documents"]
        self.metadatas = data["metadatas"]

    def reset(self):

        self.ids = []
        self.embeddings = None
        self.documents = []
        self.metadatas = []

        if os.path.exists(
            self.path
        ):

            os.remove(
                self.path
            )