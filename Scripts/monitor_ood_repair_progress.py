"""
Read-only progress monitor for the current BERTML/TabPFN OOD repair runs.

The script never edits repair manifests or result folders.  It only:
  - finds the latest BERTML and TabPFN repair manifests,
  - counts success/failed/unfinished commands,
  - inspects active Python child processes, and
  - estimates a rough remaining time from completed command durations.

Typical usage from the BERT_ML repo root:

  # One-shot status
  python Scripts\\monitor_ood_repair_progress.py

  # Refresh every 5 minutes
  python Scripts\\monitor_ood_repair_progress.py --watch --interval 300

  # Refresh every 5 minutes and append snapshots to a log
  python Scripts\\monitor_ood_repair_progress.py --watch --interval 300 --log-file output\\ood_repair_backups\\ood_repair_progress.log
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKUP_ROOT = REPO_ROOT / "output" / "ood_repair_backups"


@dataclass
class ActiveTask:
    kind: str
    pid: int
    parent_pid: int | None
    created_at: datetime | None
    elapsed_seconds: float | None
    command_line: str
    method: str | None = None
    alloy: str | None = None
    target: str | None = None
    runtime: str | None = None
    feature_mode: str | None = None


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}


def parse_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    for fmt in ("%Y%m%d_%H%M%S", "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(text[:26] if "%f" in fmt else text[:19], fmt)
        except ValueError:
            pass
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).replace(tzinfo=None)
    except Exception:
        return None


def format_seconds(seconds: float | int | None) -> str:
    if seconds is None:
        return "未知"
    seconds = int(max(0, round(float(seconds))))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def format_percent(done: int, total: int) -> str:
    if total <= 0:
        return "0.0%"
    return f"{done / total * 100:.1f}%"


def safe_median(values: list[float]) -> float | None:
    clean = [float(v) for v in values if isinstance(v, (int, float)) and float(v) > 0]
    if not clean:
        return None
    return float(statistics.median(clean))


def manifest_sort_key(path: Path) -> tuple[int, float]:
    payload = read_json(path)
    stamp = parse_datetime(payload.get("started_at"))
    stamp_value = int(stamp.timestamp()) if stamp else 0
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = 0.0
    return stamp_value, mtime


def latest_manifest(prefix: str, explicit: Path | None = None) -> Path | None:
    if explicit:
        path = explicit if explicit.is_absolute() else REPO_ROOT / explicit
        return path if path.exists() else None
    candidates = sorted(BACKUP_ROOT.glob(f"{prefix}_*/repair_manifest.json"), key=manifest_sort_key)
    if not candidates:
        return None
    execute_candidates = [p for p in candidates if read_json(p).get("execute") is True]
    return (execute_candidates or candidates)[-1]


def iter_manifests(prefix: str) -> list[Path]:
    return sorted(BACKUP_ROOT.glob(f"{prefix}_*/repair_manifest.json"), key=manifest_sort_key)


def status_of(command: dict[str, Any]) -> str:
    status = str(command.get("status") or "").lower()
    returncode = command.get("returncode")
    if status == "success" or returncode == 0:
        return "success"
    if status == "failed" or (returncode not in (None, 0)):
        return "failed"
    if status:
        return status
    return "unknown"


def is_terminal(command: dict[str, Any]) -> bool:
    return status_of(command) in {"success", "failed"}


def command_key(command: dict[str, Any]) -> tuple[str, str, str, str, str, str]:
    return (
        str(command.get("kind") or ""),
        str(command.get("runtime") or command.get("feature_mode") or ""),
        str(command.get("method") or ""),
        str(command.get("alloy") or ""),
        str(command.get("target") or ""),
        str(command.get("feature_mode") or ""),
    )


def history_keys(command: dict[str, Any]) -> list[tuple[str, ...]]:
    kind = str(command.get("kind") or "")
    method = str(command.get("method") or "")
    alloy = str(command.get("alloy") or "")
    target = str(command.get("target") or "")
    runtime = str(command.get("runtime") or "")
    feature_mode = str(command.get("feature_mode") or "")
    if kind == "tabpfn_target":
        return [
            ("full", kind, runtime, method, alloy, target),
            ("runtime_method", kind, runtime, method),
            ("feature_method", kind, feature_mode, method),
            ("runtime", kind, runtime),
            ("method", kind, method),
            ("kind", kind),
        ]
    return [
        ("full", kind, method, alloy, target),
        ("method", kind, method),
        ("kind", kind),
    ]


def collect_duration_history(prefix: str) -> tuple[dict[tuple[str, ...], list[float]], dict[tuple[str, ...], list[float]]]:
    success_history: dict[tuple[str, ...], list[float]] = defaultdict(list)
    terminal_history: dict[tuple[str, ...], list[float]] = defaultdict(list)
    for manifest_path in iter_manifests(prefix):
        manifest = read_json(manifest_path)
        for command in manifest.get("commands") or []:
            if command.get("dry_run"):
                continue
            elapsed = command.get("elapsed_seconds")
            if not isinstance(elapsed, (int, float)) or elapsed <= 0:
                continue
            if is_terminal(command):
                for key in history_keys(command):
                    terminal_history[key].append(float(elapsed))
            if status_of(command) == "success":
                for key in history_keys(command):
                    success_history[key].append(float(elapsed))
    return success_history, terminal_history


def predict_seconds(
    command: dict[str, Any],
    success_history: dict[tuple[str, ...], list[float]],
    terminal_history: dict[tuple[str, ...], list[float]],
) -> tuple[float | None, str]:
    for key in history_keys(command):
        estimate = safe_median(success_history.get(key, []))
        if estimate:
            return estimate, key[0] + ":success"
    for key in history_keys(command):
        estimate = safe_median(terminal_history.get(key, []))
        if estimate:
            return estimate, key[0] + ":terminal"
    return None, "no-history"


def powershell_python_processes() -> list[dict[str, Any]]:
    ps = r"""
$ErrorActionPreference = 'SilentlyContinue'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
Get-CimInstance Win32_Process -Filter "name = 'python.exe'" |
  ForEach-Object {
    [pscustomobject]@{
      ProcessId = $_.ProcessId
      ParentProcessId = $_.ParentProcessId
      CreationDate = $_.CreationDate.ToString('o')
      CommandLine = $_.CommandLine
    }
  } | ConvertTo-Json -Depth 4 -Compress
"""
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps],
            cwd=str(REPO_ROOT),
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=20,
        )
    except Exception:
        return []
    text = (result.stdout or "").strip()
    if not text:
        return []
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return []
    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


def regex_arg(command_line: str, name: str) -> str | None:
    # Supports --name value, --name "value with spaces", and --name=value.
    pattern = rf"--{re.escape(name)}(?:=|\s+)(?:\"([^\"]*)\"|'([^']*)'|([^\s]+))"
    match = re.search(pattern, command_line)
    if not match:
        return None
    for group in match.groups():
        if group is not None:
            return group.strip('"')
    return None


def parse_process_time(value: Any) -> datetime | None:
    parsed = parse_datetime(value)
    if parsed:
        return parsed
    return None


def result_dir_alloy(command_line: str) -> str | None:
    result_dir = regex_arg(command_line, "result_dir")
    if not result_dir:
        return None
    parts = [part for part in re.split(r"[\\/]+", result_dir.strip('"')) if part]
    for idx, part in enumerate(parts):
        if part.startswith("experiment1_all_ml_models_") and idx + 1 < len(parts):
            return parts[idx + 1]
    return None


def active_tasks() -> list[ActiveTask]:
    now = datetime.now()
    tasks: list[ActiveTask] = []
    for proc in powershell_python_processes():
        command_line = str(proc.get("CommandLine") or "")
        if not command_line:
            continue
        created_at = parse_process_time(proc.get("CreationDate"))
        elapsed = (now - created_at).total_seconds() if created_at else None
        pid = int(proc.get("ProcessId") or 0)
        parent_pid = proc.get("ParentProcessId")
        parent_pid = int(parent_pid) if parent_pid is not None else None

        if "src.TabPFN.train_tabpfn_ood" in command_line:
            feature_mode = regex_arg(command_line, "feature_mode")
            runtime = None
            if feature_mode == "numeric":
                runtime = "TabPFN-2.5-Plus-Numeric"
            elif feature_mode == "text":
                runtime = "TabPFN-2.5-Plus-Text"
            tasks.append(
                ActiveTask(
                    kind="tabpfn_target",
                    pid=pid,
                    parent_pid=parent_pid,
                    created_at=created_at,
                    elapsed_seconds=elapsed,
                    command_line=command_line,
                    method=regex_arg(command_line, "split_strategy"),
                    alloy=regex_arg(command_line, "alloy_type"),
                    target=regex_arg(command_line, "target_col"),
                    runtime=runtime,
                    feature_mode=feature_mode,
                )
            )
            continue

        if "src.pipelines.ood_pipeline" in command_line and "output\\ood_results" in command_line:
            tasks.append(
                ActiveTask(
                    kind="ml_target",
                    pid=pid,
                    parent_pid=parent_pid,
                    created_at=created_at,
                    elapsed_seconds=elapsed,
                    command_line=command_line,
                    method=regex_arg(command_line, "split_strategy"),
                    alloy=result_dir_alloy(command_line),
                    target=regex_arg(command_line, "target_column"),
                )
            )
    return tasks


def task_matches_command(task: ActiveTask, command: dict[str, Any]) -> bool:
    if task.kind != str(command.get("kind") or ""):
        return False
    for field in ("method", "alloy", "target"):
        task_value = getattr(task, field)
        command_value = command.get(field)
        if task_value and command_value and str(task_value) != str(command_value):
            return False
    if task.kind == "tabpfn_target":
        if task.feature_mode and command.get("feature_mode") and task.feature_mode != command.get("feature_mode"):
            return False
        if task.runtime and command.get("runtime") and task.runtime != command.get("runtime"):
            return False
    return True


def estimate_eta(
    commands: list[dict[str, Any]],
    tasks: list[ActiveTask],
    jobs: int,
    success_history: dict[tuple[str, ...], list[float]],
    terminal_history: dict[tuple[str, ...], list[float]],
) -> tuple[float | None, str]:
    unfinished = [command for command in commands if not is_terminal(command)]
    if not unfinished:
        return 0.0, "done"

    known_predictions: list[float] = []
    predicted: dict[int, tuple[float | None, str]] = {}
    for idx, command in enumerate(unfinished):
        pred, source = predict_seconds(command, success_history, terminal_history)
        predicted[idx] = (pred, source)
        if pred:
            known_predictions.append(pred)

    fallback = safe_median(known_predictions)
    active_indices: set[int] = set()
    active_loads: list[float] = []
    prediction_sources = Counter(source for pred, source in predicted.values() if pred)

    for task in tasks:
        matched_idx = None
        for idx, command in enumerate(unfinished):
            if idx in active_indices:
                continue
            if task_matches_command(task, command):
                matched_idx = idx
                break
        if matched_idx is None:
            continue
        active_indices.add(matched_idx)
        pred, _source = predicted[matched_idx]
        pred = pred or fallback
        if pred is None:
            if task.elapsed_seconds:
                # Low-confidence fallback: assume the active task is roughly two thirds done.
                pred = task.elapsed_seconds * 1.5
            else:
                continue
        elapsed = float(task.elapsed_seconds or 0)
        remaining = pred - elapsed
        if remaining < 0:
            # If a task already exceeded historical median, keep a small tail estimate instead of zero.
            remaining = max(min(pred * 0.25, 1800.0), 60.0)
        active_loads.append(remaining)

    queued_loads: list[float] = []
    for idx, command in enumerate(unfinished):
        if idx in active_indices:
            continue
        pred, _source = predicted[idx]
        pred = pred or fallback
        if pred is None:
            return None, "no-duration-history"
        queued_loads.append(pred)

    worker_count = max(1, int(jobs or 1))
    loads = active_loads[:worker_count]
    while len(loads) < worker_count:
        loads.append(0.0)
    for duration in queued_loads:
        min_idx = min(range(len(loads)), key=lambda idx: loads[idx])
        loads[min_idx] += duration

    source_text = ", ".join(f"{name}={count}" for name, count in prediction_sources.most_common(3)) or "fallback"
    if not active_indices and tasks:
        source_text += "; active-unmatched"
    return max(loads) if loads else 0.0, source_text


def summarize_manifest(
    label: str,
    prefix: str,
    manifest_path: Path | None,
    all_tasks: list[ActiveTask],
) -> tuple[list[str], bool]:
    lines: list[str] = []
    if not manifest_path:
        lines.append(f"## {label}: 未找到 manifest")
        return lines, False

    manifest = read_json(manifest_path)
    commands = list(manifest.get("commands") or [])
    jobs = int(manifest.get("jobs") or 1)
    status_counts = Counter(status_of(command) for command in commands)
    total = len(commands)
    success = status_counts.get("success", 0)
    failed = status_counts.get("failed", 0)
    terminal = success + failed
    unfinished = max(0, total - terminal)
    relevant_tasks = [task for task in all_tasks if any(task_matches_command(task, cmd) for cmd in commands if not is_terminal(cmd))]
    success_history, terminal_history = collect_duration_history(prefix)
    eta_seconds, eta_source = estimate_eta(commands, relevant_tasks, jobs, success_history, terminal_history)
    avg_current = safe_median(
        [float(command.get("elapsed_seconds")) for command in commands if status_of(command) == "success" and isinstance(command.get("elapsed_seconds"), (int, float))]
    )

    started = manifest.get("started_at")
    finished = manifest.get("finished_at")
    manifest_rel = manifest_path.resolve()
    try:
        manifest_rel = manifest_rel.relative_to(REPO_ROOT.resolve())
    except Exception:
        pass

    lines.append(f"## {label}")
    lines.append(f"- manifest: {manifest_rel}")
    lines.append(f"- started_at: {started} | execute={manifest.get('execute')} | jobs={jobs} | finished_at={finished or '未完成'}")
    lines.append(
        f"- 进度: success={success}, failed={failed}, unfinished={unfinished}, total={total}, 完成率={format_percent(terminal, total)}"
    )
    if status_counts:
        lines.append("- manifest状态计数: " + ", ".join(f"{key}={value}" for key, value in sorted(status_counts.items())))
    if avg_current:
        lines.append(f"- 当前 manifest 已成功任务中位耗时: {format_seconds(avg_current)}")
    if relevant_tasks:
        lines.append(f"- 当前真实活跃子进程: {len(relevant_tasks)}")
        for task in relevant_tasks[:8]:
            name_bits = [task.runtime, task.feature_mode, task.method, task.alloy, task.target]
            name = " | ".join(str(bit) for bit in name_bits if bit)
            lines.append(f"  - PID={task.pid} 已运行 {format_seconds(task.elapsed_seconds)}: {name}")
    else:
        lines.append("- 当前真实活跃子进程: 0")
    if eta_seconds is None:
        lines.append(f"- ETA: 暂无足够历史样本；若仍有活跃子进程，请以“已运行时长”为下限观察。({eta_source})")
    elif eta_seconds <= 0:
        lines.append("- ETA: 已完成/无需等待")
    else:
        finish_at = datetime.now() + timedelta(seconds=eta_seconds)
        lines.append(f"- 粗略 ETA: 还需约 {format_seconds(eta_seconds)}，预计完成 {finish_at.strftime('%Y-%m-%d %H:%M:%S')}（依据: {eta_source}）")
    return lines, unfinished > 0 or bool(relevant_tasks)


def build_report(args: argparse.Namespace) -> tuple[str, bool]:
    bert_manifest = latest_manifest("llm_env_repair", args.bert_manifest)
    tabpfn_manifest = latest_manifest("tabpfn25_llm_repair", args.tabpfn_manifest)
    tasks = active_tasks()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    sections: list[str] = [f"# OOD repair progress @ {now}", ""]
    bert_lines, bert_running = summarize_manifest("BERTML OOD repair", "llm_env_repair", bert_manifest, tasks)
    tab_lines, tab_running = summarize_manifest("TabPFN-2.5 OOD repair", "tabpfn25_llm_repair", tabpfn_manifest, tasks)
    sections.extend(bert_lines)
    sections.append("")
    sections.extend(tab_lines)
    sections.append("")
    unrelated_active = [
        task
        for task in tasks
        if not (
            bert_manifest and any(task_matches_command(task, cmd) for cmd in (read_json(bert_manifest).get("commands") or []))
        )
        and not (
            tabpfn_manifest and any(task_matches_command(task, cmd) for cmd in (read_json(tabpfn_manifest).get("commands") or []))
        )
    ]
    if unrelated_active:
        sections.append("## 其他 OOD Python 子进程")
        for task in unrelated_active[:8]:
            name_bits = [task.kind, task.runtime, task.feature_mode, task.method, task.alloy, task.target]
            sections.append(
                f"- PID={task.pid} 已运行 {format_seconds(task.elapsed_seconds)}: "
                + " | ".join(str(bit) for bit in name_bits if bit)
            )
        sections.append("")
    sections.append("注：ETA 是基于当前/历史 manifest 的已完成任务耗时估算；LOCO 和 TabPFN API 受数据量/API 排队影响，属于粗略值。")
    return "\n".join(sections), bool(bert_running or tab_running)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor current BERTML and TabPFN OOD repair progress.")
    parser.add_argument("--bert-manifest", type=Path, default=None, help="Explicit BERTML repair_manifest.json.")
    parser.add_argument("--tabpfn-manifest", type=Path, default=None, help="Explicit TabPFN repair_manifest.json.")
    parser.add_argument("--watch", action="store_true", help="Keep printing progress snapshots.")
    parser.add_argument("--interval", type=int, default=300, help="Seconds between snapshots in --watch mode.")
    parser.add_argument("--max-iterations", type=int, default=0, help="Stop after N snapshots in --watch mode; 0 means unlimited.")
    parser.add_argument("--log-file", type=Path, default=None, help="Append snapshots to this file.")
    parser.add_argument("--stop-when-done", action="store_true", help="In --watch mode, exit after all selected tasks appear finished.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    iteration = 0
    while True:
        report, still_running = build_report(args)
        print(report, flush=True)
        if args.log_file:
            log_path = args.log_file if args.log_file.is_absolute() else REPO_ROOT / args.log_file
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(report)
                handle.write("\n\n" + "=" * 80 + "\n\n")

        iteration += 1
        if not args.watch:
            break
        if args.max_iterations and iteration >= args.max_iterations:
            break
        if args.stop_when_done and not still_running:
            break
        time.sleep(max(1, int(args.interval)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
