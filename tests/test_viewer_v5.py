"""Viewer Phase V5 (PLAN_VIEWER.md): one job queue, cancel, setup checklist."""

import threading
import time

import pytest


def _wait(job_id, states=("done", "failed", "cancelled"), timeout=30.0):
    from specret.scene import jobs

    t0 = time.time()
    while time.time() - t0 < timeout:
        s = jobs.job_status(job_id)
        if s["state"] in states:
            return s
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} still {jobs.job_status(job_id)['state']}")


def test_jobs_run_one_at_a_time_in_order():
    from specret.scene import jobs

    order, gate = [], threading.Event()

    def work(name, wait=False):
        def run(job):
            order.append(f"start {name}")
            if wait:
                gate.wait(5)
            job.progress(0.5)
            job.log(f"{name} halfway")
            order.append(f"end {name}")
            return {"answer": name}

        return run

    a = jobs.submit("test", "a", "A", work("a", wait=True))
    b = jobs.submit("test", "b", "B", work("b"))
    assert jobs.submit("test", "b", "B again", work("b")) == b  # one per scene and kind
    _wait(a, ("running",))
    sb = jobs.job_status(b)
    assert sb["state"] == "queued" and sb["position"] == 1
    gate.set()
    assert _wait(b)["state"] == "done"
    assert order == ["start a", "end a", "start b", "end b"]
    sa = jobs.job_status(a)
    assert sa["answer"] == "a" and sa["log"] == ["a halfway"] and sa["progress"] == 1.0
    assert [j["job_id"] for j in jobs.list_jobs()][:2] == [b, a]


def test_cancel_a_queued_and_a_running_job():
    from specret.scene import jobs

    gate = threading.Event()

    def slow(job):
        for _ in range(500):
            gate.wait(0.01)
            job.progress(0.1)  # raises once cancelled

    def never(job):
        raise AssertionError("a cancelled queued job must not run")

    running = jobs.submit("test", "slow", "slow", slow)
    queued = jobs.submit("test", "never", "never", never)
    _wait(running, ("running",))
    assert jobs.cancel(queued)["state"] == "cancelled"
    assert jobs.cancel(running)["cancelling"] is True
    assert _wait(running)["state"] == "cancelled"
    with pytest.raises(ValueError):
        jobs.cancel("nope")


def test_command_job_runs_specret_and_logs_its_output(tmp_path):
    from specret.scene import jobs
    from specret.workspace import Workspace

    ok = jobs.start_command_job(Workspace(root=tmp_path), ["--help"])
    s = _wait(ok)
    assert s["state"] == "done" and s["exit_code"] == 0
    assert any("usage" in line for line in s["log"])
    bad = jobs.start_command_job(Workspace(root=tmp_path), ["no-such-command"])
    s = _wait(bad)
    assert s["state"] == "failed" and "exited with code" in s["error"]


def test_setup_endpoints_run_only_listed_steps(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from specret import project
    from specret.viewer.app import create_app
    from specret.workspace import Workspace

    items = [
        project.Item("canonical", "required", "Canonical library", False, "missing", [["ingest"]]),
        project.Item("gas", "optional", "EMIT radiance", False, "none", [["download", "emit-gas"]]),
    ]
    monkeypatch.setattr(project, "inspect", lambda: items)
    started = []
    from specret.scene import jobs

    monkeypatch.setattr(jobs, "start_command_job", lambda ws, argv: started.append(argv) or "j1")
    (tmp_path / "configs").mkdir()
    client = TestClient(create_app(Workspace(root=tmp_path)))
    got = client.get("/api/setup").json()
    assert [s["runnable"] for it in got for s in it["steps"]] == [True, False]
    assert got[0]["steps"][0]["command"] == "specret ingest"
    assert client.post("/api/setup/run", json={"argv": ["ingest"]}).json() == {"job_id": "j1"}
    assert client.post("/api/setup/run", json={"argv": ["download", "emit-gas"]}).status_code == 400
    assert client.post("/api/setup/run", json={"argv": ["serve-anything"]}).status_code == 400
    assert started == [["ingest"]]
    assert client.post("/api/jobs/persistence", json={"scenes": ["x"]}).status_code == 400
