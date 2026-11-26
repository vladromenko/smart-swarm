from smart_swarm.episode import atomic
from smart_swarm.research_status import capture_counts, current_worker, primary_workers


def worker(proc, project, pid, argv, state="R"):
    path = proc / str(pid)
    path.mkdir()
    (path / "cwd").symlink_to(project, target_is_directory=True)
    (path / "stat").write_text(f"{pid} (python) {state} " + "0 " * 18 + "1234\n")
    (path / "cmdline").write_bytes(b"\0".join(argv) + b"\0")
    return path


def test_current_worker_ignores_zombie_and_script_name_in_other_arguments(tmp_path):
    repo = tmp_path / "project"
    repo.mkdir()
    command = [b"/bin/python3", b"-u", b"scripts/capture_primary_test.py"]
    worker(tmp_path, repo, 110, command)
    worker(tmp_path, repo, 111, command, state="Z")
    worker(tmp_path, repo, 112, [b"python", b"-c", b"pass", command[-1]])
    worker(tmp_path, tmp_path, 113, command)
    assert primary_workers(repo, tmp_path) == [{"pid": 110, "stage": "capture", "start_ticks": 1234}]


def test_running_and_sleeping_states_belong_to_same_worker(tmp_path, monkeypatch):
    from smart_swarm import research_status

    path = worker(tmp_path, tmp_path, 110, [b"python", b"scripts/score_primary_test.py"])
    observations = iter((("R", "1234"), ("S", "1234")))
    monkeypatch.setattr(research_status, "process_stamp", lambda _: next(observations))
    assert current_worker(path, tmp_path) == {"pid": 110, "stage": "scoring", "start_ticks": 1234}


def test_only_independent_capture_groups_bound_to_release_are_counted(tmp_path):
    def receipt(group, seed, status, release):
        trial = tmp_path / group / str(seed)
        trial.mkdir(parents=True)
        atomic(trial / "capture.json", {"source_group": group, "status": status,
               "reservation": {"primary_test_release_sha256": release}})

    receipt("f1", 1, "accepted", "released")
    receipt("f1", 2, "accepted", "released")
    receipt("f2", 1, "running", "released")
    receipt("f2", 2, "accepted", "old")
    receipt("other", 1, "accepted", "released")
    damaged = tmp_path / "f2" / "3" / "capture.json"
    damaged.parent.mkdir()
    damaged.write_text("{invalid")
    counts, unique = capture_counts(tmp_path, {"f1", "f2"}, "released")
    assert counts == {"accepted": 2, "running": 1, "outside_release": 2, "unreadable_receipt": 1}
    assert unique == 1
