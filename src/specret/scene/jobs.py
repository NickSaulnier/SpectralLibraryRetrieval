"""Background jobs, shared by the MCP server and the viewer (PLAN_VIEWER.md V5).

Every long operation goes through one queue in the server process, and a single worker
thread runs the jobs one at a time, so GPU work never overlaps and the queue outlives page
reloads. Kinds:
* "map": map every pixel of a scene (`specret map-scene`)
* "gas": detect plumes in an EMIT scene, an AVIRIS-NG line or a HyTES flight
  (`specret detect-plumes`, `specret detect-plumes-lwir`)
* "unmix": unmix the pixels of a drawn area (viewer/unmix_area.py)
* "persistence": repeat sightings across analyzed EMIT scenes (`specret gas-persistence`)
* "command": any other `specret` command, run as a subprocess (the setup checklist)

A job's state is "queued", "running", "done", "failed" or "cancelled", with `progress` in
0-1 (None when unknown) and a log of what it did. Cancelling a queued job drops it; a running
job stops at its next progress report (a subprocess is terminated).
"""

from __future__ import annotations

import subprocess
import sys
import threading
import time
import traceback
import uuid
from collections import deque
from collections.abc import Callable

JOBS: dict[str, dict] = {}
_LOCK = threading.Lock()
_QUEUE: deque[str] = deque()
_WORK: dict[str, Callable] = {}
_WAKE = threading.Event()
_WORKER: threading.Thread | None = None
MAX_LOG = 400  # lines kept per job
ACTIVE = ("queued", "running")


class Cancelled(Exception):
    pass


class Job:
    """What a running job's work function gets: progress, log and the cancel check."""

    def __init__(self, job_id: str):
        self.id = job_id

    @property
    def record(self) -> dict:
        return JOBS[self.id]

    def check(self) -> None:
        if self.record.get("cancel"):
            raise Cancelled

    def progress(self, fraction: float | None) -> None:
        self.check()
        self.record["progress"] = None if fraction is None else round(float(fraction), 3)

    def log(self, line: str) -> None:
        log = self.record["log"]
        log.append(str(line).rstrip())
        del log[:-MAX_LOG]
        self.check()


def _active(scene_id: str, kind: str) -> str | None:
    for job_id, job in JOBS.items():
        if job["scene_id"] == scene_id and job["kind"] == kind and job["state"] in ACTIVE:
            return job_id
    return None


def running_job(scene_id: str, kind: str = "map") -> str | None:
    """Id of the job queued or working on this scene, if any."""
    with _LOCK:
        return _active(scene_id, kind)


def submit(kind: str, scene_id: str, title: str, work: Callable[[Job], dict | None]) -> str:
    """Queue `work(job)`, or return the job already queued or running for this scene and
    kind. Whatever dict `work` returns is merged into the job's record."""
    global _WORKER
    with _LOCK:
        if (job_id := _active(scene_id, kind)) is not None:
            return job_id
        job_id = uuid.uuid4().hex[:8]
        JOBS[job_id] = {
            "job_id": job_id,
            "kind": kind,
            "scene_id": scene_id,
            "title": title,
            "state": "queued",
            "progress": 0.0,
            "created": time.time(),
            "started": None,
            "finished": None,
            "log": [],
        }
        _WORK[job_id] = work
        _QUEUE.append(job_id)
        if _WORKER is None or not _WORKER.is_alive():
            _WORKER = threading.Thread(target=_worker, daemon=True, name="specret-jobs")
            _WORKER.start()
    _WAKE.set()
    return job_id


def _worker() -> None:
    while True:
        _WAKE.wait()
        with _LOCK:
            job_id = _QUEUE.popleft() if _QUEUE else None
            if job_id is None:
                _WAKE.clear()
                continue
            rec = JOBS[job_id]
            rec.update(state="running", started=time.time())
            work = _WORK.pop(job_id)
        job = Job(job_id)
        try:
            out = work(job) or {}
            rec.update(state="done", progress=1.0, **out)
        except Cancelled:
            rec.update(state="cancelled")
        except Exception as exc:  # reported through job_status
            rec.update(state="failed", error=f"{exc}\n{traceback.format_exc()}")
        rec["finished"] = time.time()


def cancel(job_id: str) -> dict:
    """Drop a queued job, or ask a running one to stop."""
    with _LOCK:
        if job_id not in JOBS:
            raise ValueError(f"unknown job '{job_id}'")
        rec = JOBS[job_id]
        if rec["state"] == "queued":
            _QUEUE.remove(job_id)
            _WORK.pop(job_id, None)
            rec.update(state="cancelled", finished=time.time())
        elif rec["state"] == "running":
            rec["cancel"] = True
            proc = rec.get("_proc")
            if proc is not None:
                proc.terminate()
    return job_status(job_id)


def job_status(job_id: str) -> dict:
    if job_id not in JOBS:
        raise ValueError(f"unknown job '{job_id}'")
    rec = JOBS[job_id]
    job = {k: v for k, v in rec.items() if not k.startswith("_") and k != "cancel"}
    job["log"] = list(rec["log"][-50:])
    start = rec["started"]
    end = rec["finished"] or time.time()
    job["elapsed_s"] = round(end - start) if start else 0
    job["position"] = list(_QUEUE).index(job_id) + 1 if job_id in _QUEUE else None
    job["cancelling"] = bool(rec.get("cancel")) and rec["state"] == "running"
    return job


def list_jobs() -> list[dict]:
    """Every job, newest first."""
    with _LOCK:
        ids = list(JOBS)[::-1]  # insertion order: timestamps can tie on a coarse clock
    return [job_status(j) for j in ids]


# --- the kinds ----------------------------------------------------------------------------


def start_map_job(ws, scene_id: str) -> str:
    """Queue mapping a scene, or return the job already queued or mapping it."""
    if scene_id not in ws.scenes():
        raise ValueError(f"unknown scene '{scene_id}'")

    def work(job: Job) -> dict:
        from specret.scene.mapper import map_scene
        from specret.scene.outputs import write_scene_maps

        scene = ws.scene(scene_id)
        sensor = ws.sensor_for_scene(scene)
        job.log(f"mapping {scene_id} ({sensor} library)")
        smap = map_scene(
            scene,
            ws.retriever(sensor),
            ws.canonical,
            progress=False,
            confidence=ws.confidence_model(sensor),
            on_progress=job.progress,
        )
        out = ws.map_dir(scene_id)
        smap.save(out / "scene_map.npz")
        paths = write_scene_maps(smap, scene, ws.canonical, out)
        job.log(f"wrote {len(paths)} maps to {out}")
        return {"seconds": round(smap.seconds), "outputs": {k: str(v) for k, v in paths.items()}}

    return submit("map", scene_id, f"Map minerals: {scene_id}", work)


def start_gas_job(ws, scene_id: str) -> str:
    """Queue plume detection for an EMIT scene or an AVIRIS-NG line (detect, identify,
    quantify) or a HyTES flight (detect, identify); or return the job already queued."""
    src = ws.gas_sources().get(scene_id)
    if src is None:
        raise ValueError(f"no radiance for '{scene_id}'; have {sorted(ws.gas_sources())}")

    def vswir(job: Job) -> dict:
        from specret.gas.detect import prepare
        from specret.gas.outputs import write_scene
        from specret.gas.pipeline import analyze, build_plumes

        if src.sensor == "AVIRIS-NG":
            from specret.io.avirisng import open_avirisng

            scene = open_avirisng(src.path)
        else:
            from specret.io.emit_l1b import open_emit_l1b

            scene = open_emit_l1b(src.path)
        job.log(f"reading {src.sensor} radiance")
        prep = prepare(scene)
        job.progress(0.2)
        job.log("matched filters, candidates and identification")
        enh, recs = analyze(prep)
        job.progress(0.8)
        job.log(f"{len(recs)} candidates; confidence and emission rates")
        plumes = build_plumes(prep, enh, recs)
        folder = write_scene(prep, enh, recs, plumes, ws.gas_dir(scene_id).parent)
        job.log(f"{len(plumes)} plumes -> {folder}")
        return {"plumes": len(plumes), "candidates": len(recs), "outputs": str(folder)}

    def hytes(job: Job) -> dict:
        import pandas as pd

        from specret.gas.library import plume_config
        from specret.gas.lwir import analyze_flight, write_flight
        from specret.io.hytes import open_hytes

        flight = open_hytes(src.path)
        n_seg = max(1, flight.shape[0] // plume_config()["lwir"]["segment_rows"])
        done = []

        def log(msg: str) -> None:
            done.append(msg)
            job.log(msg)
            job.progress(min(0.95, len(done) / n_seg))

        df, maps = analyze_flight(flight, log=log)
        folder = write_flight(flight, df, maps, ws.gas_dir(scene_id).parent)
        plumes = folder / "plumes.csv"
        n = len(pd.read_csv(plumes)) if plumes.stat().st_size > 1 else 0
        job.log(f"{n} plumes -> {folder}")
        return {"plumes": n, "candidates": len(df), "outputs": str(folder)}

    work = hytes if src.sensor == "HyTES" else vswir
    return submit("gas", scene_id, f"Detect plumes ({src.sensor}): {scene_id}", work)


def start_unmix_job(ws, key: str, mineral_scene: str, polygon, max_pixels: int = 3000) -> str:
    """Queue unmixing the pixels of an area (viewer/unmix_area.py), or return the job."""

    def work(job: Job) -> dict:
        from specret.viewer.unmix_area import unmix_area

        meta = unmix_area(ws, key, mineral_scene, polygon, max_pixels, job.progress)
        job.log(f"{meta.get('pixels_unmixed', '?')} pixels unmixed")
        return {"result": meta}

    return submit("unmix", key, f"Unmix area: {key}", work)


def start_persistence_job(ws, scene_ids: list[str], radius_m: float = 300.0) -> str:
    """Queue repeat-sighting counts across analyzed EMIT scenes (outputs/gas/persistence.csv)."""
    known = ws.gas_scenes()
    missing = [
        s for s in scene_ids if s not in known or not (ws.gas_dir(s) / "plumes.csv").exists()
    ]
    if len(scene_ids) < 2 or missing:
        raise ValueError(f"need two or more analyzed EMIT scenes; not analyzed: {missing}")

    def work(job: Job) -> dict:
        from specret.gas.persistence import load_scene, persistence
        from specret.io.emit_l1b import open_emit_l1b

        results = []
        for i, sid in enumerate(scene_ids):
            job.log(f"reading {sid}")
            results.append(load_scene(ws.gas_dir(sid), open_emit_l1b(known[sid])))
            job.progress((i + 1) / (len(scene_ids) + 1))
        df = persistence(results, radius_m)
        path = ws.root / "outputs/gas/persistence.csv"
        df.to_csv(path, index=False)
        covered = df[df["overpasses_covering"] > 0]
        seen = int((covered["overpasses_seen"] > 0).sum())
        job.log(f"{len(df)} plumes; {len(covered)} covered by another overpass, {seen} seen again")
        return {"plumes": len(df), "covered": len(covered), "seen_again": seen}

    return submit("persistence", "all", f"Persistence across {len(scene_ids)} scenes", work)


def start_command_job(ws, argv: list[str]) -> str:
    """Queue `specret <argv>` as a subprocess in the project folder, logging its output."""

    def work(job: Job) -> dict:
        cmd = [sys.executable, "-u", "-m", "specret.cli", *argv]
        job.log("$ specret " + " ".join(argv))
        job.progress(None)
        proc = subprocess.Popen(
            cmd,
            cwd=ws.root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        job.record["_proc"] = proc
        try:
            for line in proc.stdout:
                if "Warning" not in line:  # library deprecation noise
                    job.log(line)
        except Cancelled:
            proc.terminate()
            raise
        finally:
            code = proc.wait()
            job.record.pop("_proc", None)
        job.check()
        if code != 0:
            raise RuntimeError(f"specret {' '.join(argv)} exited with code {code}")
        return {"exit_code": code}

    return submit("command", " ".join(argv), "specret " + " ".join(argv), work)
