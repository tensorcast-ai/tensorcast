#  Copyright (c) 2026, TensorCast Team.

# Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tensorcast.artifact_runtime.local_ready_launcher import (
    LOCAL_READY_RUN_DIR_ENV,
    PREWARM_LOG_ENV,
    RETAINED_MANIFEST_ENV,
    TARGET_PLAN_CACHE_READY_TIMEOUT_S_ENV,
    launch_after_materializing_ready,
    prepare_launch_environment,
    run_after_materializing_ready,
    wait_for_materializing_ready_marker,
    wait_for_target_plan_cache_ready,
)
from tensorcast.artifact_runtime.local_ready_prewarm import (
    MATERIALIZING_READY_WRITE_ENV,
    RETAINED_MANIFEST_WRITE_ENV,
    SOURCE_PATH_FILTER_ENV,
    TARGET_PLAN_MANIFEST_B64_ENV,
    TARGET_PLAN_MANIFEST_CACHE_DIR_ENV,
    TARGET_PLAN_MANIFEST_ENV,
    TARGET_PLAN_MANIFEST_JSON_ENV,
    TARGET_PLAN_MANIFEST_SHA256_ENV,
    write_target_plan_manifest_cache,
)


def test_prepare_launch_environment_defaults_marker_log_and_read_manifest(
        tmp_path) -> None:
    target_manifest = tmp_path / "target-plan.json"
    retained_manifest = tmp_path / "retained.json"
    env = {
        TARGET_PLAN_MANIFEST_ENV: str(target_manifest),
        RETAINED_MANIFEST_WRITE_ENV: str(retained_manifest),
    }

    config = prepare_launch_environment(env)

    assert config.materializing_ready_marker == tmp_path / (
        "retained.json.ready.json")
    assert config.prewarm_log == tmp_path / (
        "retained.json.ready.json.prewarm.log")
    assert config.timeout_s == 300.0
    assert config.env[RETAINED_MANIFEST_ENV] == str(retained_manifest)
    assert config.env[MATERIALIZING_READY_WRITE_ENV] == str(
        config.materializing_ready_marker)
    assert config.env[PREWARM_LOG_ENV] == str(config.prewarm_log)


def test_prepare_launch_environment_preserves_explicit_read_manifest(
        tmp_path) -> None:
    env = {
        TARGET_PLAN_MANIFEST_ENV: str(tmp_path / "target-plan.json"),
        RETAINED_MANIFEST_WRITE_ENV: str(tmp_path / "retained-write.json"),
        RETAINED_MANIFEST_ENV: str(tmp_path / "retained-read.json"),
    }

    config = prepare_launch_environment(env)

    assert config.env[RETAINED_MANIFEST_ENV] == str(tmp_path /
                                                    "retained-read.json")


def test_prepare_launch_environment_defaults_retained_manifest_from_run_dir(
        tmp_path) -> None:
    run_dir = tmp_path / "run"
    env = {
        TARGET_PLAN_MANIFEST_JSON_ENV: "{\"records\": []}",
        LOCAL_READY_RUN_DIR_ENV: str(run_dir),
    }

    config = prepare_launch_environment(env)

    retained_manifest = run_dir / "retained-binding-manifest.json"
    assert config.env[RETAINED_MANIFEST_WRITE_ENV] == str(retained_manifest)
    assert config.env[RETAINED_MANIFEST_ENV] == str(retained_manifest)
    assert config.materializing_ready_marker == run_dir / (
        "retained-binding-manifest.json.ready.json")
    assert config.prewarm_log == run_dir / (
        "retained-binding-manifest.json.ready.json.prewarm.log")


def test_prepare_launch_environment_accepts_run_dir_argument(tmp_path) -> None:
    run_dir = tmp_path / "run-arg"
    env = {TARGET_PLAN_MANIFEST_JSON_ENV: "{\"records\": []}"}

    config = prepare_launch_environment(env, run_dir=run_dir)

    retained_manifest = run_dir / "retained-binding-manifest.json"
    assert config.env[LOCAL_READY_RUN_DIR_ENV] == str(run_dir)
    assert config.env[RETAINED_MANIFEST_WRITE_ENV] == str(retained_manifest)
    assert config.env[RETAINED_MANIFEST_ENV] == str(retained_manifest)


def test_prepare_launch_environment_retained_manifest_argument_wins(
        tmp_path) -> None:
    run_dir = tmp_path / "run"
    retained_manifest = tmp_path / "explicit-retained.json"
    env = {
        TARGET_PLAN_MANIFEST_JSON_ENV: "{\"records\": []}",
        LOCAL_READY_RUN_DIR_ENV: str(run_dir),
    }

    config = prepare_launch_environment(
        env,
        retained_manifest_write=retained_manifest,
    )

    assert config.env[LOCAL_READY_RUN_DIR_ENV] == str(run_dir)
    assert config.env[RETAINED_MANIFEST_WRITE_ENV] == str(retained_manifest)
    assert config.env[RETAINED_MANIFEST_ENV] == str(retained_manifest)


def test_prepare_launch_environment_defaults_retained_manifest_from_cache_identity(
        tmp_path) -> None:
    cache_dir = tmp_path / "target-plan-cache"
    env = {
        TARGET_PLAN_MANIFEST_CACHE_DIR_ENV: str(cache_dir),
        SOURCE_PATH_FILTER_ENV: str(tmp_path / "model"),
    }

    config = prepare_launch_environment(env)

    retained_manifest = Path(config.env[RETAINED_MANIFEST_WRITE_ENV])
    assert config.env[RETAINED_MANIFEST_ENV] == str(retained_manifest)
    assert retained_manifest.parent.parent.name == "retained_manifests"
    assert retained_manifest.parent.parent.parent == cache_dir
    assert retained_manifest.suffix == ".json"


def test_prepare_launch_environment_defaults_retained_manifest_next_to_manifest(
        tmp_path) -> None:
    target_manifest = tmp_path / "target-plan.json"
    env = {TARGET_PLAN_MANIFEST_ENV: str(target_manifest)}

    config = prepare_launch_environment(env)

    retained_manifest = tmp_path / "target-plan.json.retained.json"
    assert config.env[RETAINED_MANIFEST_WRITE_ENV] == str(retained_manifest)
    assert config.env[RETAINED_MANIFEST_ENV] == str(retained_manifest)


def test_prepare_launch_environment_requires_target_plan_manifest(
        tmp_path) -> None:
    env = {RETAINED_MANIFEST_WRITE_ENV: str(tmp_path / "retained.json")}

    with pytest.raises(ValueError, match=TARGET_PLAN_MANIFEST_ENV):
        prepare_launch_environment(env)


def test_prepare_launch_environment_accepts_inline_target_plan_manifest(
        tmp_path) -> None:
    retained_manifest = tmp_path / "retained.json"
    env = {
        TARGET_PLAN_MANIFEST_JSON_ENV: "{\"records\": []}",
        RETAINED_MANIFEST_WRITE_ENV: str(retained_manifest),
    }

    config = prepare_launch_environment(env)

    assert config.env[RETAINED_MANIFEST_ENV] == str(retained_manifest)
    assert config.env[TARGET_PLAN_MANIFEST_JSON_ENV] == "{\"records\": []}"


def test_prepare_launch_environment_inline_manifest_requires_output_location(
        tmp_path) -> None:
    del tmp_path
    env = {TARGET_PLAN_MANIFEST_JSON_ENV: "{\"records\": []}"}

    with pytest.raises(ValueError, match=LOCAL_READY_RUN_DIR_ENV):
        prepare_launch_environment(env)


def test_prepare_launch_environment_accepts_inline_target_plan_manifest_b64(
        tmp_path) -> None:
    retained_manifest = tmp_path / "retained.json"
    env = {
        TARGET_PLAN_MANIFEST_B64_ENV: "eyJyZWNvcmRzIjogW119",
        RETAINED_MANIFEST_WRITE_ENV: str(retained_manifest),
    }

    config = prepare_launch_environment(env)

    assert config.env[RETAINED_MANIFEST_ENV] == str(retained_manifest)
    assert config.env[TARGET_PLAN_MANIFEST_B64_ENV] == "eyJyZWNvcmRzIjogW119"


def test_prepare_launch_environment_accepts_target_plan_cache_identity(
        tmp_path) -> None:
    retained_manifest = tmp_path / "retained.json"
    env = {
        TARGET_PLAN_MANIFEST_CACHE_DIR_ENV:
        str(tmp_path / "target-plan-cache"),
        TARGET_PLAN_MANIFEST_SHA256_ENV:
        "a" * 64,
        RETAINED_MANIFEST_WRITE_ENV:
        str(retained_manifest),
    }

    config = prepare_launch_environment(env)

    assert config.env[RETAINED_MANIFEST_ENV] == str(retained_manifest)
    assert config.env[TARGET_PLAN_MANIFEST_CACHE_DIR_ENV] == str(
        tmp_path / "target-plan-cache")
    assert config.env[TARGET_PLAN_MANIFEST_SHA256_ENV] == "a" * 64


def test_prepare_launch_environment_accepts_target_plan_cache_source_index(
        tmp_path) -> None:
    retained_manifest = tmp_path / "retained.json"
    env = {
        TARGET_PLAN_MANIFEST_CACHE_DIR_ENV:
        str(tmp_path / "target-plan-cache"),
        SOURCE_PATH_FILTER_ENV:
        str(tmp_path / "model"),
        RETAINED_MANIFEST_WRITE_ENV:
        str(retained_manifest),
    }

    config = prepare_launch_environment(env)

    assert config.env[RETAINED_MANIFEST_ENV] == str(retained_manifest)
    assert config.env[SOURCE_PATH_FILTER_ENV] == str(tmp_path / "model")


def _write_ready_target_plan_cache(cache_dir: Path,
                                   source_path: str) -> dict[str, object]:
    return write_target_plan_manifest_cache(
        cache_dir,
        {
            "schema_version": 1,
            "producer": "test",
            "intent_kind": "local_ready_target_plan",
            "records_by_source_path": {
                source_path: [{
                    "schema_version": 1,
                    "intent_key": "intent-key",
                    "source_path": source_path,
                    "target_proto_b64": "dGFyZ2V0",
                    "expected_member": {
                        "member_count": 1,
                    },
                }],
            },
        },
        source_path=source_path,
        producer="test",
    )


def test_wait_for_target_plan_cache_ready_sets_sha_from_source_index(
        tmp_path) -> None:
    cache_dir = tmp_path / "target-plan-cache"
    source_path = str(tmp_path / "model")
    summary = _write_ready_target_plan_cache(cache_dir, source_path)
    env = {
        TARGET_PLAN_MANIFEST_CACHE_DIR_ENV: str(cache_dir),
        SOURCE_PATH_FILTER_ENV: source_path,
    }

    ready = wait_for_target_plan_cache_ready(env, timeout_s=0.0)

    assert ready is not None
    assert ready["event"] == "target_plan_manifest_cache_ready"
    assert ready["manifest_sha256"] == summary["manifest_sha256"]
    assert env[TARGET_PLAN_MANIFEST_SHA256_ENV] == summary["manifest_sha256"]
    assert Path(str(ready["manifest_path"])).exists()


def test_wait_for_target_plan_cache_ready_reports_timeout(
        tmp_path) -> None:
    env = {
        TARGET_PLAN_MANIFEST_CACHE_DIR_ENV:
        str(tmp_path / "target-plan-cache"),
        SOURCE_PATH_FILTER_ENV:
        str(tmp_path / "model"),
    }

    with pytest.raises(TimeoutError, match="target-plan cache readiness"):
        wait_for_target_plan_cache_ready(env, timeout_s=0.0)


def test_wait_for_materializing_ready_marker_fails_fast(tmp_path) -> None:
    marker = tmp_path / "ready.json"
    marker.write_text(
        json.dumps({
            "ready": False,
            "event": "materializing_records_failed",
            "error_message": "prefetch boom",
        }),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="prefetch boom"):
        wait_for_materializing_ready_marker(marker, timeout_s=1.0)


def test_wait_for_materializing_ready_marker_reports_child_exit(
        tmp_path) -> None:
    log_path = tmp_path / "prewarm.log"
    log_path.write_text("child failed\n", encoding="utf-8")

    class _ExitedProc:

        def poll(self):
            return 17

    with pytest.raises(RuntimeError, match="rc=17") as exc_info:
        wait_for_materializing_ready_marker(
            tmp_path / "missing-ready.json",
            prewarm_proc=_ExitedProc(),
            prewarm_log=log_path,
            timeout_s=1.0,
        )
    assert "child failed" in str(exc_info.value)


def test_launch_after_materializing_ready_execs_with_manifest_env(
        tmp_path) -> None:
    target_manifest = tmp_path / "target-plan.json"
    retained_manifest = tmp_path / "retained.json"
    marker = tmp_path / "ready.json"
    env = {
        TARGET_PLAN_MANIFEST_ENV: str(target_manifest),
        RETAINED_MANIFEST_WRITE_ENV: str(retained_manifest),
    }
    captured: dict[str, object] = {}

    class _RunningProc:

        def poll(self):
            return None

        def terminate(self):
            captured["terminated"] = True

        def wait(self, timeout=None):
            del timeout
            return 0

        def kill(self):
            captured["killed"] = True

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = list(cmd)
        child_env = kwargs["env"]
        marker_path = Path(child_env[MATERIALIZING_READY_WRITE_ENV])
        marker_path.write_text(
            json.dumps({
                "ready": True,
                "event": "materializing_records_ready",
            }),
            encoding="utf-8",
        )
        return _RunningProc()

    class _ExecCalled(Exception):
        pass

    def fake_exec(file, args, exec_env):
        captured["exec"] = (file, list(args), dict(exec_env))
        raise _ExecCalled()

    with pytest.raises(_ExecCalled):
        launch_after_materializing_ready(
            ["python", "-c", "print('ready')"],
            env=env,
            materializing_ready_write=marker,
            timeout_s=1.0,
            exec_fn=fake_exec,
            popen_factory=fake_popen,
        )

    assert captured["cmd"][:3] == [
        "python",
        "-m",
        "tensorcast.artifact_runtime.local_ready_prewarm",
    ] or captured["cmd"][1:3] == [
        "-m",
        "tensorcast.artifact_runtime.local_ready_prewarm",
    ]
    file, args, exec_env = captured["exec"]
    assert file == "python"
    assert args == ["python", "-c", "print('ready')"]
    assert exec_env[RETAINED_MANIFEST_ENV] == str(retained_manifest)
    assert exec_env[MATERIALIZING_READY_WRITE_ENV] == str(marker)


def test_launch_after_materializing_ready_accepts_run_dir_argument(
        tmp_path) -> None:
    target_manifest = tmp_path / "target-plan.json"
    run_dir = tmp_path / "run"
    env = {TARGET_PLAN_MANIFEST_ENV: str(target_manifest)}
    captured: dict[str, object] = {}

    class _RunningProc:

        def poll(self):
            return None

        def terminate(self):
            captured["terminated"] = True

        def wait(self, timeout=None):
            del timeout
            return 0

        def kill(self):
            captured["killed"] = True

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = list(cmd)
        child_env = kwargs["env"]
        Path(child_env[MATERIALIZING_READY_WRITE_ENV]).write_text(
            json.dumps({
                "ready": True,
                "event": "materializing_records_ready",
            }),
            encoding="utf-8",
        )
        return _RunningProc()

    class _ExecCalled(Exception):
        pass

    def fake_exec(file, args, exec_env):
        captured["exec"] = (file, list(args), dict(exec_env))
        raise _ExecCalled()

    with pytest.raises(_ExecCalled):
        launch_after_materializing_ready(
            ["python", "-c", "print('ready')"],
            env=env,
            run_dir=run_dir,
            timeout_s=1.0,
            exec_fn=fake_exec,
            popen_factory=fake_popen,
        )

    _file, _args, exec_env = captured["exec"]
    retained_manifest = run_dir / "retained-binding-manifest.json"
    assert exec_env[LOCAL_READY_RUN_DIR_ENV] == str(run_dir)
    assert exec_env[RETAINED_MANIFEST_WRITE_ENV] == str(retained_manifest)
    assert exec_env[RETAINED_MANIFEST_ENV] == str(retained_manifest)


def test_launch_after_materializing_ready_waits_for_target_plan_cache(
        tmp_path) -> None:
    cache_dir = tmp_path / "target-plan-cache"
    source_path = str(tmp_path / "model")
    run_dir = tmp_path / "run"
    summary = _write_ready_target_plan_cache(cache_dir, source_path)
    env = {
        TARGET_PLAN_MANIFEST_CACHE_DIR_ENV: str(cache_dir),
        SOURCE_PATH_FILTER_ENV: source_path,
    }
    captured: dict[str, object] = {}

    class _RunningProc:

        def poll(self):
            return None

        def terminate(self):
            captured["terminated"] = True

        def wait(self, timeout=None):
            del timeout
            return 0

        def kill(self):
            captured["killed"] = True

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = list(cmd)
        child_env = kwargs["env"]
        captured["prewarm_env"] = dict(child_env)
        Path(child_env[MATERIALIZING_READY_WRITE_ENV]).write_text(
            json.dumps({
                "ready": True,
                "event": "materializing_records_ready",
            }),
            encoding="utf-8",
        )
        return _RunningProc()

    class _ExecCalled(Exception):
        pass

    def fake_exec(file, args, exec_env):
        captured["exec"] = (file, list(args), dict(exec_env))
        raise _ExecCalled()

    with pytest.raises(_ExecCalled):
        launch_after_materializing_ready(
            ["python", "-c", "print('ready')"],
            env=env,
            run_dir=run_dir,
            target_plan_cache_ready_timeout_s=0.0,
            timeout_s=1.0,
            exec_fn=fake_exec,
            popen_factory=fake_popen,
        )

    prewarm_env = captured["prewarm_env"]
    assert isinstance(prewarm_env, dict)
    assert (prewarm_env[TARGET_PLAN_MANIFEST_SHA256_ENV] ==
            summary["manifest_sha256"])
    _file, _args, exec_env = captured["exec"]
    assert exec_env[TARGET_PLAN_MANIFEST_SHA256_ENV] == summary[
        "manifest_sha256"]
    assert exec_env[LOCAL_READY_RUN_DIR_ENV] == str(run_dir)


def test_launch_after_materializing_ready_reads_cache_wait_timeout_from_env(
        tmp_path) -> None:
    cache_dir = tmp_path / "target-plan-cache"
    source_path = str(tmp_path / "model")
    summary = _write_ready_target_plan_cache(cache_dir, source_path)
    env = {
        TARGET_PLAN_MANIFEST_CACHE_DIR_ENV:
        str(cache_dir),
        SOURCE_PATH_FILTER_ENV:
        source_path,
        TARGET_PLAN_CACHE_READY_TIMEOUT_S_ENV:
        "0",
        LOCAL_READY_RUN_DIR_ENV:
        str(tmp_path / "run"),
    }
    captured: dict[str, object] = {}

    class _RunningProc:

        def poll(self):
            return None

        def terminate(self):
            captured["terminated"] = True

        def wait(self, timeout=None):
            del timeout
            return 0

        def kill(self):
            captured["killed"] = True

    def fake_popen(cmd, **kwargs):
        del cmd
        child_env = kwargs["env"]
        captured["prewarm_env"] = dict(child_env)
        Path(child_env[MATERIALIZING_READY_WRITE_ENV]).write_text(
            json.dumps({
                "ready": True,
                "event": "materializing_records_ready",
            }),
            encoding="utf-8",
        )
        return _RunningProc()

    class _ExecCalled(Exception):
        pass

    def fake_exec(file, args, exec_env):
        captured["exec"] = (file, list(args), dict(exec_env))
        raise _ExecCalled()

    with pytest.raises(_ExecCalled):
        launch_after_materializing_ready(
            ["python", "-c", "print('ready')"],
            env=env,
            timeout_s=1.0,
            exec_fn=fake_exec,
            popen_factory=fake_popen,
        )

    prewarm_env = captured["prewarm_env"]
    assert isinstance(prewarm_env, dict)
    assert (prewarm_env[TARGET_PLAN_MANIFEST_SHA256_ENV] ==
            summary["manifest_sha256"])


def test_launch_after_materializing_ready_terminates_prewarm_on_exec_failure(
        tmp_path) -> None:
    target_manifest = tmp_path / "target-plan.json"
    retained_manifest = tmp_path / "retained.json"
    marker = tmp_path / "ready.json"
    env = {
        TARGET_PLAN_MANIFEST_ENV: str(target_manifest),
        RETAINED_MANIFEST_WRITE_ENV: str(retained_manifest),
    }
    captured: dict[str, object] = {}

    class _RunningProc:

        def poll(self):
            return None

        def terminate(self):
            captured["terminated"] = True

        def wait(self, timeout=None):
            del timeout
            return 0

        def kill(self):
            captured["killed"] = True

    def fake_popen(cmd, **kwargs):
        del cmd
        child_env = kwargs["env"]
        Path(child_env[MATERIALIZING_READY_WRITE_ENV]).write_text(
            json.dumps({
                "ready": True,
                "event": "materializing_records_ready",
            }),
            encoding="utf-8",
        )
        return _RunningProc()

    def fake_exec(*_args):
        raise OSError("exec failed")

    with pytest.raises(OSError, match="exec failed"):
        launch_after_materializing_ready(
            ["python", "-c", "print('ready')"],
            env=env,
            materializing_ready_write=marker,
            timeout_s=1.0,
            exec_fn=fake_exec,
            popen_factory=fake_popen,
        )

    assert captured["terminated"] is True


def test_run_after_materializing_ready_supervises_and_reaps_prewarm(
        tmp_path) -> None:
    target_manifest = tmp_path / "target-plan.json"
    retained_manifest = tmp_path / "retained.json"
    marker = tmp_path / "ready.json"
    env = {
        TARGET_PLAN_MANIFEST_ENV: str(target_manifest),
        RETAINED_MANIFEST_WRITE_ENV: str(retained_manifest),
    }
    captured: dict[str, object] = {}

    class _ExitedPrewarmProc:

        def poll(self):
            return 0

        def wait(self, timeout=None):
            captured["prewarm_wait_timeout"] = timeout
            return 0

        def terminate(self):
            captured["prewarm_terminated"] = True

        def kill(self):
            captured["prewarm_killed"] = True

    class _ExitedCommandProc:

        def poll(self):
            return 17

        def wait(self):
            captured["command_waited"] = True
            return 17

        def terminate(self):
            captured["command_terminated"] = True

        def kill(self):
            captured["command_killed"] = True

    def fake_prewarm_popen(cmd, **kwargs):
        captured["prewarm_cmd"] = list(cmd)
        child_env = kwargs["env"]
        Path(child_env[MATERIALIZING_READY_WRITE_ENV]).write_text(
            json.dumps({
                "ready": True,
                "event": "materializing_records_ready",
            }),
            encoding="utf-8",
        )
        return _ExitedPrewarmProc()

    def fake_command_popen(cmd, **kwargs):
        captured["command"] = list(cmd)
        captured["command_env"] = dict(kwargs["env"])
        return _ExitedCommandProc()

    rc = run_after_materializing_ready(
        ["python", "-c", "print('ready')"],
        env=env,
        materializing_ready_write=marker,
        timeout_s=1.0,
        popen_factory=fake_prewarm_popen,
        command_popen_factory=fake_command_popen,
    )

    assert rc == 17
    assert captured["command"] == ["python", "-c", "print('ready')"]
    assert captured["command_env"][RETAINED_MANIFEST_ENV] == str(
        retained_manifest)
    assert captured["command_waited"] is True
    assert captured["prewarm_wait_timeout"] == 0
    assert "prewarm_terminated" not in captured


def test_run_after_materializing_ready_terminates_command_on_prewarm_failure(
        tmp_path) -> None:
    target_manifest = tmp_path / "target-plan.json"
    retained_manifest = tmp_path / "retained.json"
    marker = tmp_path / "ready.json"
    env = {
        TARGET_PLAN_MANIFEST_ENV: str(target_manifest),
        RETAINED_MANIFEST_WRITE_ENV: str(retained_manifest),
    }
    captured: dict[str, object] = {}

    class _FailedPrewarmProc:

        def poll(self):
            return 19

        def wait(self, timeout=None):
            captured["prewarm_wait_timeout"] = timeout
            return 19

        def terminate(self):
            captured["prewarm_terminated"] = True

        def kill(self):
            captured["prewarm_killed"] = True

    class _RunningCommandProc:

        def poll(self):
            return None

        def wait(self, timeout=None):
            captured["command_wait_timeout"] = timeout
            return -15

        def terminate(self):
            captured["command_terminated"] = True

        def kill(self):
            captured["command_killed"] = True

    def fake_prewarm_popen(cmd, **kwargs):
        del cmd
        child_env = kwargs["env"]
        Path(child_env[MATERIALIZING_READY_WRITE_ENV]).write_text(
            json.dumps({
                "ready": True,
                "event": "materializing_records_ready",
            }),
            encoding="utf-8",
        )
        return _FailedPrewarmProc()

    def fake_command_popen(cmd, **kwargs):
        captured["command"] = list(cmd)
        captured["command_env"] = dict(kwargs["env"])
        return _RunningCommandProc()

    rc = run_after_materializing_ready(
        ["python", "-c", "print('ready')"],
        env=env,
        materializing_ready_write=marker,
        timeout_s=1.0,
        popen_factory=fake_prewarm_popen,
        command_popen_factory=fake_command_popen,
    )

    assert rc == 19
    assert captured["command"] == ["python", "-c", "print('ready')"]
    assert captured["command_env"][RETAINED_MANIFEST_ENV] == str(
        retained_manifest)
    assert captured["command_terminated"] is True
    assert captured["command_wait_timeout"] == 5.0
    assert captured["prewarm_wait_timeout"] == 0
    assert "prewarm_terminated" not in captured


def test_run_after_materializing_ready_terminates_prewarm_on_command_start_failure(
        tmp_path) -> None:
    target_manifest = tmp_path / "target-plan.json"
    retained_manifest = tmp_path / "retained.json"
    marker = tmp_path / "ready.json"
    env = {
        TARGET_PLAN_MANIFEST_ENV: str(target_manifest),
        RETAINED_MANIFEST_WRITE_ENV: str(retained_manifest),
    }
    captured: dict[str, object] = {}

    class _RunningPrewarmProc:

        def poll(self):
            return None

        def terminate(self):
            captured["prewarm_terminated"] = True

        def wait(self, timeout=None):
            captured["prewarm_wait_timeout"] = timeout
            return 0

        def kill(self):
            captured["prewarm_killed"] = True

    def fake_prewarm_popen(cmd, **kwargs):
        del cmd
        child_env = kwargs["env"]
        Path(child_env[MATERIALIZING_READY_WRITE_ENV]).write_text(
            json.dumps({
                "ready": True,
                "event": "materializing_records_ready",
            }),
            encoding="utf-8",
        )
        return _RunningPrewarmProc()

    def failing_command_popen(cmd, **kwargs):
        del cmd, kwargs
        raise RuntimeError("command failed to start")

    with pytest.raises(RuntimeError, match="command failed to start"):
        run_after_materializing_ready(
            ["python", "-c", "print('ready')"],
            env=env,
            materializing_ready_write=marker,
            timeout_s=1.0,
            popen_factory=fake_prewarm_popen,
            command_popen_factory=failing_command_popen,
        )

    assert captured["prewarm_terminated"] is True
