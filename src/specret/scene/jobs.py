"""Background scene-mapping jobs, shared by the MCP server and the viewer.

A job maps every pixel of one scene in a daemon thread and writes the same outputs as
`specret map-scene`. Its state is "running", "done" or "failed", with `progress` in 0-1.
"""

from __future__ import annotations

import threading
import time
import traceback
import uuid

JOBS: dict[str, dict] = {}
_LOCK = threading.Lock()


def _running(scene_id: str) -> str | None:
    for job_id, job in JOBS.items():
        if job["scene_id"] == scene_id and job["state"] == "running":
            return job_id
    return None


def running_job(scene_id: str) -> str | None:
    """Id of the job currently mapping this scene, if any."""
    with _LOCK:
        return _running(scene_id)


def start_map_job(ws, scene_id: str) -> str:
    """Start mapping a scene, or return the job already mapping it."""
    if scene_id not in ws.scenes():
        raise ValueError(f"unknown scene '{scene_id}'")
    with _LOCK:
        if (job_id := _running(scene_id)) is not None:
            return job_id
        job_id = uuid.uuid4().hex[:8]
        JOBS[job_id] = {
            "scene_id": scene_id,
            "state": "running",
            "progress": 0.0,
            "started": time.time(),
        }

    def progress(fraction: float) -> None:
        JOBS[job_id]["progress"] = round(fraction, 3)

    def work() -> None:
        from specret.scene.mapper import map_scene
        from specret.scene.outputs import write_scene_maps

        try:
            scene = ws.scene(scene_id)
            sensor = ws.sensor_for_scene(scene)
            smap = map_scene(
                scene,
                ws.retriever(sensor),
                ws.canonical,
                progress=False,
                confidence=ws.confidence_model(sensor),
                on_progress=progress,
            )
            out = ws.map_dir(scene_id)
            smap.save(out / "scene_map.npz")
            paths = write_scene_maps(smap, scene, ws.canonical, out)
            JOBS[job_id].update(
                state="done",
                seconds=round(smap.seconds),
                outputs={k: str(v) for k, v in paths.items()},
            )
        except Exception as exc:  # reported through job_status
            JOBS[job_id].update(state="failed", error=f"{exc}\n{traceback.format_exc()}")

    threading.Thread(target=work, daemon=True).start()
    return job_id


def job_status(job_id: str) -> dict:
    if job_id not in JOBS:
        raise ValueError(f"unknown job '{job_id}'")
    job = dict(JOBS[job_id])
    job["elapsed_s"] = round(time.time() - job.pop("started"))
    return job
