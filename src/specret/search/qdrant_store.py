"""Qdrant collections: one per sensor, with a named vector per representation.

Qdrant is a derived index for interactive queries, payload filtering and the MCP tools.
It is always rebuildable from data/canonical and data/resampled. Scene-scale mapping uses
search.gpu_exact instead.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator

import numpy as np
import pandas as pd
from qdrant_client import QdrantClient, models

from specret.search.vectors import library_vectors
from specret.sensors.library import SensorLibrary

ID_NAMESPACE = uuid.UUID("6f1c7a52-7d0e-4c55-9b0c-5e0e2d7f6a11")
PAYLOAD_FIELDS = (
    "spectrum_id",
    "sample_id",
    "source",
    "material_name",
    "mineral",
    "group",
    "chapter",
    "instrument",
    "measurement",
    "purity_flag",
    "quality_flag",
    "coverage_min_nm",
    "coverage_max_nm",
)
KEYWORD_INDEXES = ("source", "mineral", "group", "chapter", "measurement", "purity_flag")


def point_id(spectrum_id: str) -> str:
    return str(uuid.uuid5(ID_NAMESPACE, spectrum_id))


def collection_name(sensor_name: str) -> str:
    return f"lib_{sensor_name}"


def _clean(v):
    if isinstance(v, float) and np.isnan(v):
        return None
    if isinstance(v, np.generic):
        return v.item()
    return v


class QdrantStore:
    def __init__(self, url: str | None = None, path: str | None = None):
        self.is_server = bool(url)
        if url:
            self.client = QdrantClient(url=url)
        elif path:
            self.client = QdrantClient(path=path)
        else:
            self.client = QdrantClient(location=":memory:")

    def close(self) -> None:
        """Release the client (embedded mode holds a file lock until closed)."""
        self.client.close()

    def __enter__(self) -> QdrantStore:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def recreate(self, sensor_name: str, dims: dict[str, int]) -> str:
        name = collection_name(sensor_name)
        if self.client.collection_exists(name):
            self.client.delete_collection(name)
        self.client.create_collection(
            name,
            vectors_config={
                v: models.VectorParams(size=d, distance=models.Distance.COSINE)
                for v, d in dims.items()
            },
        )
        if self.is_server:  # embedded mode ignores payload indexes
            for field in KEYWORD_INDEXES:
                self.client.create_payload_index(name, field, models.PayloadSchemaType.KEYWORD)
        return name

    def index_library(self, lib: SensorLibrary, canonical: pd.DataFrame, batch: int = 256) -> int:
        """(Re)build the sensor's collection from a resampled library. Returns points written."""
        vecs = library_vectors(lib)
        name = self.recreate(lib.sensor.name, {v: m.shape[1] for v, (m, _) in vecs.items()})
        meta = canonical.set_index("spectrum_id")

        def points() -> Iterator[models.PointStruct]:
            for i, sid in enumerate(lib.spectrum_ids):
                vector = {v: m[i].tolist() for v, (m, ok) in vecs.items() if ok[i]}
                if not vector:
                    continue  # no usable coverage on this sensor
                row = meta.loc[sid]
                payload = {f: _clean(row[f]) for f in PAYLOAD_FIELDS if f in row.index}
                payload["spectrum_id"] = str(sid)
                yield models.PointStruct(id=point_id(str(sid)), vector=vector, payload=payload)

        n = 0
        buf: list[models.PointStruct] = []
        for p in points():
            buf.append(p)
            if len(buf) >= batch:
                self.client.upsert(name, buf)
                n, buf = n + len(buf), []
        if buf:
            self.client.upsert(name, buf)
            n += len(buf)
        return n

    def search(
        self,
        sensor_name: str,
        vector_name: str,
        query: np.ndarray,
        k: int = 10,
        filters: dict[str, str] | None = None,
    ) -> list[dict]:
        """Top-k for one unit-length query vector. `filters` are exact payload matches."""
        query_filter = None
        if filters:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(key=key, match=models.MatchValue(value=value))
                    for key, value in filters.items()
                ]
            )
        res = self.client.query_points(
            collection_name(sensor_name),
            query=np.asarray(query, dtype=np.float64).tolist(),
            using=vector_name,
            limit=k,
            query_filter=query_filter,
            with_payload=True,
        )
        return [{"score": p.score, **(p.payload or {})} for p in res.points]
