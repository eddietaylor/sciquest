"""Bounded, tool-free Codex generation with isolated CLI state and usage receipts."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time

from sciquest import blind
from sciquest.io import content_hash, file_hash, utc_now

DISABLED_FEATURES = (
    "shell_tool",
    "unified_exec",
    "multi_agent",
    "apps",
    "plugins",
    "hooks",
    "skill_search",
    "view_image",
    "image_generation",
    "browser_use",
    "computer_use",
    "goals",
    "sleep_tool",
    "workspace_dependencies",
    "unbounded_connection_retries",
    "memories",
    "external_agent_memory_import",
    "chronicle",
    "remote_plugin",
    "multi_agent_v2",
    "code_mode",
    "code_mode_host",
    "in_app_browser",
    "in_app_chat",
    "shell_snapshot",
    "daemon_auto_start",
)


def describe_backend(binary: str) -> dict:
    path = Path(shutil.which(binary) or binary).resolve(strict=True)
    version = subprocess.run(
        [str(path), "--version"], check=True, capture_output=True, text=True, timeout=10
    ).stdout.strip()
    return dict(
        kind="codex_cli",
        binary=str(path),
        binary_sha256=file_hash(path),
        version=version,
        tools="disabled; non-message events reject a call",
        disabled_features=list(DISABLED_FEATURES),
        host_skill_discovery=False,
        project_instructions=False,
        user_config=False,
        fresh_session_per_call=True,
        memory_read=False,
        memory_write=False,
        filesystem="empty-root allowlist; auth-only private client state",
        auth="existing CLI file authentication in a temporary private mount",
    )


def parse_events(trace: str) -> dict:
    """Fail closed on tools, failed turns and missing usage; don't guess token counts."""
    completed = []
    messages = []
    for line in trace.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue  # CLI diagnostics on stderr share the bounded log.
        if not isinstance(event, dict):
            continue
        kind = event.get("type")
        if kind in ("turn.failed", "error"):
            raise ValueError("Backend reported a failed model turn")
        if kind in ("item.started", "item.completed"):
            item = event.get("item", {})
            if item.get("type") not in ("agent_message", "reasoning", "error"):
                raise ValueError("Tool or unsupported item in tool-free experiment")
            if kind == "item.completed" and item.get("type") == "agent_message":
                messages.append(item.get("text"))
        if kind == "turn.completed":
            completed.append(event.get("usage"))
    if len(completed) != 1 or not messages:
        raise ValueError(
            "Exactly one completed model turn with a final message is required"
        )
    usage = completed[0]
    if not isinstance(usage, dict):
        raise ValueError("Missing model usage")
    for key in (
        "input_tokens",
        "output_tokens",
        "cached_input_tokens",
        "reasoning_output_tokens",
    ):
        if key in ("input_tokens", "output_tokens") or key in usage:
            if type(usage.get(key)) is not int or usage[key] < 0:
                raise ValueError("Invalid or missing token usage")
    return dict(usage=usage, final_message=messages[-1])


def run_call(
    directory: Path,
    prompt: str,
    schema: dict,
    spec: dict,
    protected_root: Path,
    timeout: float,
) -> dict:
    """Run once. An attempted or interrupted call is never silently retried."""
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "prompt.txt").write_text(prompt)
    (directory / "schema.json").write_text(json.dumps(schema, indent=2))
    receipt = dict(
        started_at=utc_now(),
        prompt_hash=content_hash(prompt),
        schema_hash=content_hash(schema),
        backend_spec_hash=content_hash(spec),
        status="failed",
        usage=None,
        timeout_seconds=timeout,
    )
    started = time.monotonic()
    trace_path = directory / "trace.jsonl"
    try:
        if (
            file_hash(Path(spec["backend"]["binary"]))
            != spec["backend"]["binary_sha256"]
        ):
            raise ValueError("Frozen backend executable changed")
        with tempfile.TemporaryDirectory(prefix="sciquest-model-client-") as tmp:
            temp = Path(tmp)
            workspace = temp / "worker"
            workspace.mkdir()
            client = temp / "client"
            client.mkdir(mode=0o700)
            # Keep the caller's CODEX_HOME value. The mount supplies private,
            # writable client state at the SAME pathname inside this namespace.
            # The original auth/state are never modified or copied into artifacts.
            auth_root = Path(
                os.environ.get("CODEX_HOME", str(Path.home() / ".codex"))
            ).resolve()
            source_auth = auth_root / "auth.json"
            if not source_auth.is_file():
                raise ValueError(
                    "Isolated backend requires existing CLI file authentication"
                )
            shutil.copyfile(source_auth, client / "auth.json")
            (client / "auth.json").chmod(0o600)
            (workspace / "schema.json").write_text(json.dumps(schema))
            shown = "/tmp/sciquest-workspace"
            argv = [
                spec["backend"]["binary"],
                "exec",
                "--json",
                "--skip-git-repo-check",
                "--ignore-user-config",
                "--ignore-rules",
                "--ephemeral",
                "--sandbox",
                "read-only",
                "--model",
                spec["model"],
                "-c",
                "model_reasoning_effort=" + json.dumps(spec["reasoning_effort"]),
                "-c",
                'web_search="disabled"',
                "-c",
                "project_doc_max_bytes=0",
                "-c",
                "memories.use_memories=false",
                "-c",
                "memories.generate_memories=false",
                "--enable",
                "skip_host_skill_discovery",
            ]
            for feature in DISABLED_FEATURES:
                argv += ["--disable", feature]
            argv += [
                "--output-schema",
                shown + "/schema.json",
                "--output-last-message",
                shown + "/answer.json",
                "-",
            ]
            prefix = blind.prefix(
                workspace, client_binary=Path(spec["backend"]["binary"])
            )
            prefix += ["--bind", str(client), str(auth_root)]
            env = blind.environment(client=True)
            command = prefix + [
                sys.executable,
                blind.WORKER,
                "command",
                str(16 * 1024 * 1024),
                "--",
                *argv,
            ]
            receipt["command_hash"] = content_hash(argv)
            with trace_path.open("xb") as output:
                process = subprocess.Popen(
                    command,
                    cwd=workspace,
                    env=env,
                    stdin=subprocess.PIPE,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                try:
                    process.communicate(input=prompt.encode(), timeout=timeout)
                except subprocess.TimeoutExpired:
                    receipt["status"] = "timed_out"
                finally:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
                    if process.stdin and not process.stdin.closed:
                        process.stdin.close()
            receipt["returncode"] = process.returncode
            parsed = parse_events(trace_path.read_text(errors="replace"))
            receipt["usage"] = parsed["usage"]
            if process.returncode != 0:
                raise ValueError("Backend process did not succeed")
            answer = workspace / "answer.json"
            if (
                not answer.is_file()
                or answer.is_symlink()
                or answer.stat().st_size > 1024 * 1024
            ):
                raise ValueError("Missing or oversized structured model response")
            response = json.loads(answer.read_text())
            if response != json.loads(parsed["final_message"]):
                raise ValueError("Final model event and response file disagree")
            (directory / "response.json").write_text(json.dumps(response, indent=2))
            receipt.update(
                status="succeeded",
                response=response,
                response_hash=content_hash(response),
            )
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        receipt["error"] = str(exc)
    finally:
        receipt.update(
            finished_at=utc_now(), elapsed_seconds=time.monotonic() - started
        )
        if trace_path.exists():
            receipt["trace_sha256"] = file_hash(trace_path)
        (directory / "receipt.json").write_text(json.dumps(receipt, indent=2))
    return receipt
