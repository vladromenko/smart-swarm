from smart_swarm.episode import atomic
from smart_swarm.research_status import capture_counts, current_worker, motor_evaluation_progress, primary_workers


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


def test_fresh_checkpoint_progress_requires_the_saved_validation_choice(tmp_path):
    checkpoint = str(tmp_path / "training/checkpoints/000085/pretrained_model")
    validation = {"checkpoint": checkpoint, "successful_resets": 0, "offline_validation_mean_loss": 0.44,
                  "resets": [{"seed": i} for i in range(20)]}
    receipt = tmp_path / "fresh/run/evaluation.json"
    receipt.parent.mkdir(parents=True)
    record = {"checkpoint": checkpoint, "evaluation_role": "fresh-test", "status": "evaluating", "requested_resets": 20,
              "results": [{"contact_valid_placement": False}]}
    atomic(receipt, record)
    assert motor_evaluation_progress(tmp_path, {})["selected_checkpoint"] is None

    atomic(tmp_path / "validation-choices.json", [validation])
    report = motor_evaluation_progress(tmp_path, {})
    assert report["selected_checkpoint"] == checkpoint
    assert report["completed_validation_candidates"] == [
        {"checkpoint": checkpoint, "successful_resets": 0, "offline_validation_mean_loss": 0.44, "resets": 20}]
    assert report["fresh_evaluations"] == [{"checkpoint": checkpoint, "status": "evaluating", "requested_resets": 20,
                                           "receipt": str(receipt), "completed_resets": 1}]
    atomic(receipt, {**record, "checkpoint": "different"})
    assert motor_evaluation_progress(tmp_path, {})["selected_checkpoint"] is None


def test_full_act_status_distinguishes_failed_fit_from_missing_comparison(tmp_path, monkeypatch):
    from smart_swarm import research_status

    monkeypatch.setattr(research_status, "EVIDENCE", tmp_path)
    monkeypatch.setattr(research_status, "PILOT", tmp_path)
    monkeypatch.setattr(research_status, "COMPACT", tmp_path)
    assert research_status.motor()["full_comparison_status"]["act"] == "awaiting_receipt"
    atomic(tmp_path / "act-clock-fit.json", {"fit_gate": False, "matching_starts": 5, "successful": 0})
    assert research_status.motor()["full_comparison_status"]["act"] == "fit_gate_failed"
    atomic(tmp_path / "act-full.json", {"status": "evaluated"})
    assert research_status.motor()["full_comparison_status"]["act"] == "evaluated"
