from hashlib import sha256
from pathlib import Path

import pytest

from smart_swarm.episode import atomic, json_bytes, load

ROOT = Path(__file__).resolve().parents[1]


def frozen_profile(path, *, status="frozen_for_primary_test"):
    profile = {"status": status, "validation_count": 48, "source_allocation_sha256": "allocation",
               "config_sha256": "protocol", "fits": {"events_only": {"checkpoint_sha256": "opaque-checkpoint",
                                                                    "threshold": .75}}}
    atomic(path, profile)
    return profile


def test_test_release_precedes_recording_and_cannot_be_rebound(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    from primary_test_release import RELEASE, bound_test_release

    profile_path = tmp_path / "profile.json"
    root = tmp_path / "test"
    profile = frozen_profile(profile_path)
    with pytest.raises(ValueError, match="receipt is absent"):
        bound_test_release(root, profile_path, profile)
    assert not root.exists()
    release = bound_test_release(root, profile_path, profile, publish=True)
    verify = bound_test_release(root, profile_path, profile)
    assert release == verify
    expected = load(root / RELEASE)
    profile["fits"]["events_only"]["threshold"] = .9
    atomic(profile_path, profile)
    with pytest.raises(ValueError, match="already been bound"):
        bound_test_release(root, profile_path, profile, publish=True)
    assert load(root / RELEASE) == expected


def test_release_cannot_open_from_a_draft_or_an_unbound_recording(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    from primary_test_release import RELEASE, bound_test_release

    profile_path = tmp_path / "profile.json"
    root = tmp_path / "test"
    profile = frozen_profile(profile_path, status="draft")
    with pytest.raises(ValueError, match="Only released profiles"):
        bound_test_release(root, profile_path, profile, publish=True)
    assert not root.exists()
    profile = frozen_profile(profile_path)
    root.mkdir()
    (root / "previous-recording").mkdir()
    with pytest.raises(ValueError, match="lack their original release"):
        bound_test_release(root, profile_path, profile, publish=True)
    assert not (root / RELEASE).exists()


def test_predictions_resume_only_with_original_profile_inputs_and_payload(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    import score_primary_test as scoring

    path = tmp_path / "predictions.json"
    identity = {"scope": "test fixture", "primary_test_release_sha256": "release",
                "profiles_sha256": "profile", "manifests": {"/input": "manifest"}}
    predictions = {"/input": {"decision": "request_inspection"}}
    monkeypatch.setattr(scoring, "predict_sources", lambda *_: predictions)
    saved, reused, _ = scoring.save_or_resume_predictions(path, identity, [], {}, lambda: None)
    assert saved == predictions and not reused
    monkeypatch.setattr(scoring, "predict_sources", lambda *_: pytest.fail("A valid resume must reuse saved predictions"))
    saved, reused, _ = scoring.save_or_resume_predictions(path, identity, [], {}, lambda: None)
    assert saved == predictions and reused
    with pytest.raises(ValueError, match="different profile"):
        scoring.save_or_resume_predictions(path, {**identity, "profiles_sha256": "modified"}, [], {}, lambda: None)
    receipt = load(path)
    receipt["predictions"]["/input"]["decision"] = "right"
    atomic(path, receipt)
    with pytest.raises(ValueError, match="checksum differs"):
        scoring.save_or_resume_predictions(path, identity, [], {}, lambda: None)
    receipt["predictions"]["/input"]["truth"] = "right"
    receipt["prediction_payload_sha256"] = sha256(json_bytes(receipt["predictions"])).hexdigest()
    atomic(path, receipt)
    with pytest.raises(ValueError, match="contain evaluator fields"):
        scoring.save_or_resume_predictions(path, identity, [], {}, lambda: None)


def test_primary_input_requires_the_bound_source_receipt(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    import score_primary_test as scoring

    source = {"source_group": "test-f1", "reservation": {"primary_test_release_sha256": "stale"}, "teams": []}
    monkeypatch.setattr(scoring, "TEST_OUTPUT", tmp_path)
    monkeypatch.setattr(scoring, "allocated_slots", lambda part: [{"source_group": source["source_group"]}])
    monkeypatch.setattr(scoring, "accepted_sources", lambda *args, **kwargs: [source])
    with pytest.raises(ValueError, match="lacks the original frozen-profile binding"):
        scoring.input_receipts({}, False, "admitted")


def test_frozen_calibration_archive_is_immutable_and_checks_member_bytes(monkeypatch, tmp_path):
    import tarfile

    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    from archive_primary_release import INDEX, publish, put, verify

    content = b"frozen profile"
    digest = sha256(content).hexdigest()
    files = {"profiles.json": content}
    index = {"schema": "smart-swarm-frozen-calibration-archive-v1", "profile_member": "profiles.json",
             "profile_sha256": digest, "validation_source_groups": 48, "validation_recording_manifests": {},
             "contents_sha256": {"profiles.json": digest}}
    first, second = tmp_path / "first.tar.gz", tmp_path / "second.tar.gz"
    publish(first, files, index)
    publish(second, files, index)
    assert first.read_bytes() == second.read_bytes()
    assert verify(first)["members_verified"] == 1
    with pytest.raises(ValueError, match="immutable calibration archive"):
        publish(first, files, index)
    altered = tmp_path / "changed.tar.gz"
    with tarfile.open(altered, "w:gz") as archive:
        put(archive, INDEX, json_bytes(index))
        put(archive, "profiles.json", b"different contents")
    with pytest.raises(ValueError, match="Archive member changed"):
        verify(altered)


def test_interrupted_test_budget_is_charged_and_limits_cannot_be_renewed(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    import primary_test_budget as budget

    path = tmp_path / budget.LEDGER
    cap = budget.TEST_LIMITS["wall_seconds"] * budget.NANOS
    start = budget.time.monotonic_ns() - 3 * budget.NANOS
    row = {"limits": budget.TEST_LIMITS, "primary_test_release_sha256": "release",
           "used_wall_ns": cap - 2 * budget.NANOS, "active_capture_wall_ns": budget.time.time_ns(),
           "active_capture_monotonic_ns": start, "active_capture_boot_id": budget.boot_identity()}
    atomic(path, row)
    with pytest.raises(budget.CaptureBudgetExhausted, match="cumulative wall-time"):
        with budget.capture_budget(tmp_path, "release"):
            pytest.fail("A previously spent budget must stop before starting another source")
    assert load(path)["status"] == "cap_reached" and load(path)["used_wall_ns"] == cap
    with pytest.raises(ValueError, match="different release"):
        with budget.capture_budget(tmp_path, "altered"):
            pytest.fail("The previous release owns the remaining cap")


def test_disk_budget_counts_unfinished_recordings_as_well_as_accepted(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    import primary_test_budget as budget

    monkeypatch.setitem(budget.TEST_LIMITS, "disk_bytes", 100)
    monkeypatch.setattr(budget, "ROLLOUT_DISK_RESERVE", 25)
    (tmp_path / "accepted.png").write_bytes(b"a" * 50)
    failed = tmp_path / "failed"
    failed.mkdir()
    (failed / "partial.png").write_bytes(b"b" * 15)
    budget.guard_space(tmp_path, 1)
    with pytest.raises(budget.CaptureBudgetExhausted, match="disk allowance"):
        budget.guard_space(tmp_path, 2)


def test_cap_closes_only_the_interrupted_capture_interval(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    import capture_primary_test as capture

    monkeypatch.setattr(capture, "TEST_OUTPUT", tmp_path)
    root = tmp_path / "reserved-case"
    for interval in ("older", "present"):
        path = root / interval
        path.mkdir(parents=True)
        atomic(path / "capture.json", {"status": "running", "reservation": {"budget_interval_id": interval}})
    capture.mark_capped_attempt({"source_group": "reserved-case"}, "present", "time cap")
    assert load(root / "older/capture.json")["status"] == "running"
    failed = load(root / "present/capture.json")
    assert failed["status"] == "interrupted" and failed["reason"] == "time cap"
