import numpy as np
import pandas as pd
import pytest

from specret.preprocess.normalize import l2_normalize
from specret.search import gpu_exact
from specret.search.gpu_exact import ExactIndex
from specret.search.qdrant_store import QdrantStore
from specret.search.retrieve import Retriever
from specret.sensors.library import SensorLibrary
from specret.sensors.sensor import Sensor


@pytest.fixture
def toy_library():
    rng = np.random.default_rng(1)
    centers = np.linspace(400, 2500, 120)
    good = ~((centers > 1340) & (centers < 1450))
    sensor = Sensor("toy", centers, np.full(120, 10.0), good)
    matrix = rng.uniform(0.05, 0.9, (40, 120))
    matrix[5, :20] = np.nan  # no full coverage, but still has SWIR2
    ids = np.array([f"s{i}" for i in range(40)])
    canonical = pd.DataFrame(
        {
            "spectrum_id": ids,
            "sample_id": [f"sample{i // 2}" for i in range(40)],
            "source": "toy",
            "material_name": [f"mat{i}" for i in range(40)],
            "mineral": [f"mat{i}" for i in range(40)],
            "chapter": "M",
            "instrument": "X",
            "measurement": ["AREF" if i % 2 else "RREF" for i in range(40)],
            "purity_flag": None,
            "coverage_min_nm": 400.0,
            "coverage_max_nm": 2500.0,
        }
    )
    return SensorLibrary(sensor, ids, matrix), canonical


def _brute(lib_vecs, q, k):
    sim = q @ lib_vecs.T
    return np.argsort(-sim, axis=1)[:, :k]


def test_exact_index_matches_brute_force_numpy_and_torch():
    rng = np.random.default_rng(2)
    lib = l2_normalize(rng.normal(size=(500, 64)))
    q = l2_normalize(rng.normal(size=(30, 64)))
    expected = _brute(lib, q, 7)
    backends = [("cpu", None)]
    if gpu_exact.torch is not None and gpu_exact.torch.cuda.is_available():
        backends.append(("cuda", None))
    for device, _ in backends:
        _, idx = ExactIndex(lib, np.arange(500), device=device).search(q, 7)
        np.testing.assert_array_equal(idx, expected)


def test_exact_index_numpy_fallback(monkeypatch):
    monkeypatch.setattr(gpu_exact, "torch", None)
    rng = np.random.default_rng(3)
    lib = l2_normalize(rng.normal(size=(200, 16)))
    q = l2_normalize(rng.normal(size=(5, 16)))
    index = ExactIndex(lib, np.arange(200))
    assert not index.use_torch
    _, idx = index.search(q, 4)
    np.testing.assert_array_equal(idx, _brute(lib, q, 4))


def test_retriever_finds_scaled_copy(toy_library):
    lib, canonical = toy_library
    r = Retriever(lib, canonical, device="cpu")
    query = lib.matrix[12] * 0.4  # brightness change: SAM is insensitive to it
    top = r.query(query, k=3)[0]
    assert top[0]["spectrum_id"] == "s12"
    assert top[0]["sam_deg"] == pytest.approx(0.0, abs=1e-3)


def test_retriever_handles_missing_query_bands(toy_library):
    lib, canonical = toy_library
    r = Retriever(lib, canonical, device="cpu")
    query = lib.matrix[12].copy()
    query[30:40] = np.nan  # bands the query lacks
    top = r.query(query, k=3)[0]
    assert top[0]["spectrum_id"] == "s12"

    nearly_empty = np.full(lib.sensor.n_bands, np.nan)
    nearly_empty[:5] = 0.3
    assert r.query(nearly_empty, k=3)[0] == []


def test_qdrant_ranking_matches_exact_search(toy_library):
    lib, canonical = toy_library
    store = QdrantStore()  # in-memory
    n = store.index_library(lib, canonical)
    assert n == 40  # s5 is indexed through its swir2 vector only

    r = Retriever(lib, canonical, device="cpu")
    query = lib.matrix[7] + 0.01
    exact_ids = [m["spectrum_id"] for m in r.query(query, k=5)[0]]
    from specret.search.vectors import query_vectors

    qv, _ = query_vectors(lib, query)["sam"]
    hits = store.search("toy", "sam", qv[0], k=5)
    assert [h["spectrum_id"] for h in hits] == exact_ids

    filtered = store.search("toy", "sam", qv[0], k=5, filters={"measurement": "AREF"})
    assert all(h["measurement"] == "AREF" for h in filtered)


def test_filter_library_by_chapter_and_purity(toy_library):
    from specret.search.retrieve import filter_library

    lib, canonical = toy_library
    grades = ("a", "b", "c", None)
    canonical = canonical.assign(
        purity_flag=[grades[(i // 2) % 4] for i in range(40)], chapter=["M", "V"] * 20
    )
    meta = canonical.set_index("spectrum_id")
    out = filter_library(lib, meta, chapters=("M",), purity=("a", "b"))
    kept = meta.loc[out.spectrum_ids]
    assert set(kept["chapter"]) == {"M"}
    assert set(kept["purity_flag"].dropna()) <= {"a", "b"}
    assert kept["purity_flag"].isna().any()  # ungraded spectra are kept
