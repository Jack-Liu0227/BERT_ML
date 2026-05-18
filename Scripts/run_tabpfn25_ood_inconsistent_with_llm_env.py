"""
Backup and rerun inconsistent TabPFN-2.5 OOD outputs with the `llm` conda env.

Default behavior is safe:
  - Relaunches itself with C:\\Users\\HK\\miniconda3\\envs\\llm\\python.exe.
  - Reads the TabPFN verify-only repair manifest.
  - Selects only TabPFN-2.5-Plus-Numeric and TabPFN-2.5-Plus-Text issues.
  - Dry-runs unless --execute is passed.

Typical commands from the BERT_ML repo root:

  # Inspect the 64 selected TabPFN-2.5 issue tasks.
  python Scripts\\run_tabpfn25_ood_inconsistent_with_llm_env.py --jobs 1

  # Backup existing old outputs and rerun selected issue tasks.
  python Scripts\\run_tabpfn25_ood_inconsistent_with_llm_env.py --execute --jobs 1

The script writes a manifest under:
  output\\ood_repair_backups\\tabpfn25_llm_repair_YYYYMMDD_HHMMSS\\repair_manifest.json
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LLM_PYTHON = Path(r"C:\Users\HK\miniconda3\envs\llm\python.exe")
DEFAULT_REPORT = REPO_ROOT / "output" / "ood_repair_backups" / "repair_20260516_215409" / "repair_manifest.json"

TARGET_RUNTIMES = {"TabPFN-2.5-Plus-Numeric", "TabPFN-2.5-Plus-Text"}

OOD_METHOD_ORDER = [
    "sparse_x_single",
    "sparse_y_single",
    "sparse_x_cluster",
    "sparse_y_cluster",
    "loco",
]

TABPFN_CONFIGS = {
    "sparse_x_single": "tabpfn_all_sparse_x_single",
    "sparse_y_single": "tabpfn_all_sparse_y_single",
    "sparse_x_cluster": "tabpfn_all_sparse_x_cluster",
    "sparse_y_cluster": "tabpfn_all_sparse_y_cluster",
    "loco": "tabpfn_all_loco",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backup and rerun inconsistent TabPFN-2.5 OOD outputs with the llm env.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--llm-python",
        type=Path,
        default=DEFAULT_LLM_PYTHON,
        help="Python executable of the target llm environment.",
    )
    parser.add_argument(
        "--no-relaunch",
        action="store_true",
        help="Do not relaunch into --llm-python. Mainly for debugging.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually move old outputs to backup and rerun. Without this, dry-run only.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of TabPFN target tasks to run concurrently. Use 1 for API-safe serial execution.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT,
        help="Verify-only repair_manifest.json containing TabPFN issues.",
    )
    parser.add_argument(
        "--resume-manifest",
        type=Path,
        default=None,
        help=(
            "Resume a previous tabpfn25 repair_manifest.json by rerunning only commands whose status is not success. "
            "Already-successful commands are skipped. Existing partial output dirs for resumed tasks are moved aside."
        ),
    )
    parser.add_argument(
        "--verify-after",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run post-rerun TabPFN verification and write remaining TabPFN-2.5 issues to the manifest.",
    )
    return parser.parse_args()


def same_executable(left: Path, right: Path) -> bool:
    try:
        return left.resolve().samefile(right.resolve())
    except Exception:
        return str(left.resolve()).lower() == str(right.resolve()).lower()


def relaunch_with_llm_if_needed(args: argparse.Namespace) -> int | None:
    if args.no_relaunch:
        return None
    llm_python = args.llm_python
    if not llm_python.exists():
        raise FileNotFoundError(f"llm Python not found: {llm_python}")
    if same_executable(Path(sys.executable), llm_python):
        return None

    cmd = [str(llm_python), str(Path(__file__).resolve()), *sys.argv[1:], "--no-relaunch"]
    print("[INFO] Relaunching with llm env:", flush=True)
    print("       " + " ".join(quote(part) for part in cmd), flush=True)
    result = subprocess.run(cmd, cwd=str(REPO_ROOT), text=True)
    return int(result.returncode)


def quote(value: object) -> str:
    text = str(value)
    return f'"{text}"' if any(ch in text for ch in " ()%\\") else text


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except Exception:
        return str(path)


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._=-" else "_" for ch in value).strip("_") or "item"


def method_sort_index(method: str) -> int:
    try:
        return OOD_METHOD_ORDER.index(method)
    except ValueError:
        return len(OOD_METHOD_ORDER)


def issue_key(issue: dict[str, Any]) -> tuple[int, str, str, str]:
    return (
        method_sort_index(str(issue.get("method", ""))),
        str(issue.get("runtime", "")),
        str(issue.get("alloy", "")),
        str(issue.get("target", "")),
    )


def load_selected_issues(report_path: Path) -> list[dict[str, Any]]:
    if not report_path.is_absolute():
        report_path = REPO_ROOT / report_path
    if not report_path.exists():
        raise FileNotFoundError(f"verify-only report not found: {report_path}")

    payload = read_json(report_path)
    issues = []
    for issue in payload.get("initial_issues", []):
        if issue.get("kind") != "tabpfn":
            continue
        if issue.get("runtime") not in TARGET_RUNTIMES:
            continue
        method = str(issue.get("method", ""))
        if method not in TABPFN_CONFIGS:
            continue
        issues.append(dict(issue))
    return sorted(issues, key=issue_key)


def issue_from_command(command: dict[str, Any], reason: str) -> dict[str, Any]:
    runtime = str(command["runtime"])
    method = str(command["method"])
    alloy = str(command["alloy"])
    target = str(command["target"])
    feature_mode = str(command.get("feature_mode") or feature_mode_for_runtime(runtime))
    return {
        "kind": "tabpfn",
        "method": method,
        "alloy": alloy,
        "target": target,
        "runtime": runtime,
        "backend": "api",
        "feature_mode": feature_mode,
        "result_dir": rel(tabpfn_result_dir(runtime, method, alloy, target)),
        "reasons": [reason],
    }


def load_resume_issues(resume_manifest: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_path = resume_manifest if resume_manifest.is_absolute() else REPO_ROOT / resume_manifest
    if not manifest_path.exists():
        raise FileNotFoundError(f"resume manifest not found: {manifest_path}")
    payload = read_json(manifest_path)
    commands = list(payload.get("commands") or [])
    selected = []
    skipped_success = 0
    skipped_other = 0
    seen: set[tuple[str, str, str, str]] = set()
    for command in commands:
        if command.get("kind") != "tabpfn_target":
            skipped_other += 1
            continue
        if command.get("runtime") not in TARGET_RUNTIMES:
            skipped_other += 1
            continue
        if command.get("status") == "success" and command.get("returncode") == 0:
            skipped_success += 1
            continue
        required = ["runtime", "method", "alloy", "target"]
        if any(not command.get(name) for name in required):
            skipped_other += 1
            continue
        key = (
            str(command["runtime"]),
            str(command["method"]),
            str(command["alloy"]),
            str(command["target"]),
        )
        if key in seen:
            continue
        seen.add(key)
        selected.append(issue_from_command(command, f"resume_from:{manifest_path}"))

    info = {
        "resume_manifest": str(manifest_path),
        "resume_original_started_at": payload.get("started_at"),
        "resume_original_finished_at": payload.get("finished_at"),
        "resume_original_command_count": len(commands),
        "resume_skipped_success_count": skipped_success,
        "resume_skipped_other_count": skipped_other,
        "resume_selected_count": len(selected),
    }
    return sorted(selected, key=issue_key), info


def feature_mode_for_runtime(runtime: str) -> str:
    if runtime == "TabPFN-2.5-Plus-Numeric":
        return "numeric"
    if runtime == "TabPFN-2.5-Plus-Text":
        return "text"
    raise ValueError(f"Unsupported runtime: {runtime}")


def tabpfn_result_dir(runtime: str, method: str, alloy: str, target: str) -> Path:
    runtime_dir = runtime
    method_dirs = {
        "sparse_x_single": "sparse_x_single_k5",
        "sparse_y_single": "sparse_y_single_k5",
        "sparse_x_cluster": "sparse_x_cluster_k5",
        "sparse_y_cluster": "sparse_y_cluster_k5",
        "loco": "loco_k5",
    }
    dataset_names = {
        "Al": "aluminum",
        "HEA": "hea",
        "MatbenchSteels": "matbench_steels_ood",
        "Steel": "steel",
        "Ti": "titanium",
    }
    return REPO_ROOT / "output" / f"ood_results_{runtime_dir}" / method_dirs[method] / alloy / dataset_names[alloy] / target


def build_command(issue: dict[str, Any]) -> list[str]:
    method = str(issue["method"])
    runtime = str(issue["runtime"])
    return [
        sys.executable,
        "-m",
        "src.TabPFN.train_tabpfn_ood",
        "--alloy_type",
        str(issue["alloy"]),
        "--target_col",
        str(issue["target"]),
        "--backend",
        "api",
        "--feature_mode",
        feature_mode_for_runtime(runtime),
        "--base_path",
        str(REPO_ROOT),
        "--test_size",
        "0.2",
        "--random_state",
        "42",
        "--split_strategy",
        method,
        "--extrapolation_side",
        "low_to_high",
        "--sparse_candidate_pool_size",
        "500",
        "--sparse_cluster_count",
        "5",
        "--sparse_samples_per_cluster",
        "1",
        "--sparse_neighbors_per_seed",
        "5",
        "--loco_cluster_count",
        "5",
        "--baseline_num_folds",
        "5",
    ]


def backup_path_for(source: Path, backup_root: Path) -> Path:
    try:
        relative = source.resolve().relative_to(REPO_ROOT.resolve())
    except Exception:
        relative = Path(source.name)
    candidate = backup_root / relative
    if not candidate.exists():
        return candidate
    stamp = datetime.now().strftime("%H%M%S_%f")
    return candidate.with_name(f"{candidate.name}__{stamp}")


def backup_existing(source: Path, backup_root: Path, *, execute: bool) -> str | None:
    if not source.exists():
        return None
    destination = backup_path_for(source, backup_root)
    if execute:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))
    return str(destination)


def clear_progress_for_selected(issues: list[dict[str, Any]]) -> int:
    progress_file = REPO_ROOT / ".batch_progress_tabpfn_ood.json"
    if not progress_file.exists():
        return 0
    payload = read_json(progress_file)
    removed = 0
    for issue in issues:
        config_name = TABPFN_CONFIGS[str(issue["method"])]
        section = payload.get(config_name)
        if not isinstance(section, dict):
            continue
        key = f"{issue['alloy']}_{issue['target']}"
        if key in section:
            section.pop(key, None)
            removed += 1
        if not section:
            payload.pop(config_name, None)
    if removed:
        write_json(progress_file, payload)
    return removed


def run_command(cmd: list[str], *, execute: bool, log_path: Path) -> dict[str, Any]:
    display = " ".join(quote(part) for part in cmd)
    print(f"[CMD] {display}")
    if not execute:
        return {
            "command": list(cmd),
            "returncode": None,
            "dry_run": True,
            "log_path": rel(log_path),
        }

    started = datetime.now()
    started_monotonic = time.monotonic()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
        log_file.write(f"# started_at={started.isoformat()}\n")
        log_file.write(f"# cwd={REPO_ROOT}\n")
        log_file.write(f"# command={display}\n\n")
        log_file.flush()
        result = subprocess.run(
            list(cmd),
            cwd=str(REPO_ROOT),
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        finished = datetime.now()
        elapsed = time.monotonic() - started_monotonic
        log_file.write(f"\n# finished_at={finished.isoformat()} returncode={result.returncode} elapsed_seconds={elapsed:.1f}\n")

    return {
        "command": list(cmd),
        "returncode": result.returncode,
        "dry_run": False,
        "log_path": rel(log_path),
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "elapsed_seconds": round(time.monotonic() - started_monotonic, 3),
    }


def make_command_record(issue: dict[str, Any], log_path: Path, jobs: int, execute: bool) -> dict[str, Any]:
    return {
        "kind": "tabpfn_target",
        "runtime": issue.get("runtime"),
        "backend": "api",
        "feature_mode": feature_mode_for_runtime(str(issue["runtime"])),
        "method": issue.get("method"),
        "alloy": issue.get("alloy"),
        "target": issue.get("target"),
        "jobs_requested": jobs,
        "status": "running" if execute else "dry_run",
        "log_path": rel(log_path),
    }


def rerun_selected(
    issues: list[dict[str, Any]],
    *,
    execute: bool,
    jobs: int,
    backup_root: Path,
    manifest: dict[str, Any],
    manifest_path: Path,
) -> None:
    log_dir = backup_root / "logs" / "tabpfn"
    pending = []
    for issue in issues:
        cmd = build_command(issue)
        log_path = (
            log_dir
            / f"{safe_name(str(issue['runtime']))}__{safe_name(str(issue['method']))}__"
            / f"{safe_name(str(issue['alloy']))}__{safe_name(str(issue['target']))}.log"
        )
        record = make_command_record(issue, log_path, jobs, execute)
        manifest.setdefault("commands", []).append(record)
        pending.append((issue, record, cmd, log_path))
        if not execute:
            result = run_command(cmd, execute=False, log_path=log_path)
            record.update(result)
    write_json(manifest_path, manifest)
    if not execute:
        return

    def run_one(item: tuple[dict[str, Any], dict[str, Any], list[str], Path]) -> tuple[dict[str, Any], dict[str, Any]]:
        issue, record, cmd, log_path = item
        result = run_command(cmd, execute=True, log_path=log_path)
        record.update(result)
        record["status"] = "success" if result.get("returncode") == 0 else "failed"
        return issue, record

    max_workers = max(1, int(jobs))
    if max_workers == 1:
        for item in pending:
            run_one(item)
            write_json(manifest_path, manifest)
        return

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_item = {executor.submit(run_one, item): item for item in pending}
        for future in as_completed(future_to_item):
            issue, record, _cmd, _log_path = future_to_item[future]
            try:
                future.result()
            except Exception as exc:  # noqa: BLE001
                record["status"] = "failed"
                record["returncode"] = -1
                record["error"] = repr(exc)
                record["issue"] = {
                    "runtime": issue.get("runtime"),
                    "method": issue.get("method"),
                    "alloy": issue.get("alloy"),
                    "target": issue.get("target"),
                }
            write_json(manifest_path, manifest)


def selected_summary(issues: list[dict[str, Any]]) -> dict[str, int]:
    summary: dict[str, int] = {}
    for issue in issues:
        key = f"{issue['runtime']}|{issue['method']}|{issue['alloy']}|{issue['target']}"
        summary[key] = summary.get(key, 0) + 1
    return dict(sorted(summary.items()))


def post_verify_selected() -> list[dict[str, Any]]:
    import repair_ood_sparse_loco as repair  # noqa: WPS433

    post = repair.detect_tabpfn_issues(include_api=True)
    selected = []
    for issue in post:
        if issue.runtime in TARGET_RUNTIMES:
            selected.append(repair.issue_to_dict(issue))
    return sorted(selected, key=lambda item: (str(item.get("runtime")), method_sort_index(str(item.get("method"))), str(item.get("alloy")), str(item.get("target"))))


def main() -> int:
    args = parse_args()
    relaunched_code = relaunch_with_llm_if_needed(args)
    if relaunched_code is not None:
        return relaunched_code

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    scripts_dir = REPO_ROOT / "Scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    import numpy as np  # noqa: WPS433
    import pandas as pd  # noqa: WPS433
    import sklearn  # noqa: WPS433

    report_path = args.report if args.report.is_absolute() else REPO_ROOT / args.report
    resume_info = None
    if args.resume_manifest:
        selected, resume_info = load_resume_issues(args.resume_manifest)
    else:
        selected = load_selected_issues(report_path)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_root = REPO_ROOT / "output" / "ood_repair_backups" / f"tabpfn25_llm_repair_{timestamp}"
    manifest_path = backup_root / "repair_manifest.json"

    runtime_counts: dict[str, int] = {}
    for issue in selected:
        runtime = str(issue.get("runtime"))
        runtime_counts[runtime] = runtime_counts.get(runtime, 0) + 1

    manifest: dict[str, Any] = {
        "started_at": timestamp,
        "repo_root": str(REPO_ROOT),
        "python": sys.executable,
        "versions": {
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "execute": bool(args.execute),
        "jobs": int(args.jobs),
        "source_report": rel(report_path),
        "resume": bool(args.resume_manifest),
        "resume_info": resume_info,
        "target_runtimes": sorted(TARGET_RUNTIMES),
        "selected_issue_count": len(selected),
        "selected_runtime_counts": dict(sorted(runtime_counts.items())),
        "selected_issue_summary": selected_summary(selected),
        "selected_issues": selected,
        "backups": [],
        "commands": [],
        "failed_commands": [],
        "post_verify_issues": [],
    }
    write_json(manifest_path, manifest)

    print(f"[INFO] repo={REPO_ROOT}")
    print(f"[INFO] python={sys.executable}")
    print(f"[INFO] sklearn={sklearn.__version__} numpy={np.__version__} pandas={pd.__version__}")
    print(f"[INFO] execute={args.execute} jobs={args.jobs}")
    print(f"[INFO] report={report_path}")
    if resume_info:
        print(f"[INFO] resume_manifest={resume_info['resume_manifest']}")
        print(
            "[INFO] resume commands: "
            f"original={resume_info['resume_original_command_count']} "
            f"skipped_success={resume_info['resume_skipped_success_count']} "
            f"selected={resume_info['resume_selected_count']}"
        )
    print(f"[INFO] backup_root={backup_root}")
    print(f"[INFO] selected={len(selected)} runtime_counts={runtime_counts}")

    for issue in selected[:80]:
        reasons = ", ".join(str(reason) for reason in issue.get("reasons", [])[:4])
        print(f"[ISSUE] {issue['runtime']}:{issue['method']}:{issue['alloy']}:{issue['target']} -> {reasons}")

    if not selected:
        print(f"[INFO] nothing to run. manifest={manifest_path}")
        manifest["finished_at"] = datetime.now().strftime("%Y%m%d_%H%M%S")
        write_json(manifest_path, manifest)
        return 0

    print("[INFO] backing up selected existing TabPFN-2.5 result directories...")
    backup_root.mkdir(parents=True, exist_ok=True)
    for issue in selected:
        source = REPO_ROOT / str(issue["result_dir"])
        backup_dest = backup_existing(source, backup_root, execute=args.execute)
        if backup_dest:
            manifest["backups"].append(
                {
                    "kind": "tabpfn_target",
                    "issue": issue,
                    "source": rel(source),
                    "destination": rel(Path(backup_dest)),
                    "dry_run": not args.execute,
                }
            )
            write_json(manifest_path, manifest)

    if not args.execute:
        print("[DRY-RUN] No files moved and no commands executed. Add --execute to backup and rerun.")
        rerun_selected(
            selected,
            execute=False,
            jobs=args.jobs,
            backup_root=backup_root,
            manifest=manifest,
            manifest_path=manifest_path,
        )
        manifest["finished_at"] = datetime.now().strftime("%Y%m%d_%H%M%S")
        write_json(manifest_path, manifest)
        print(f"[INFO] dry-run manifest={manifest_path}")
        return 0

    print("[INFO] clearing TabPFN progress entries for selected targets...")
    manifest["progress_entries_cleared"] = clear_progress_for_selected(selected)
    write_json(manifest_path, manifest)

    print("[INFO] rerunning selected TabPFN-2.5 OOD targets...")
    rerun_selected(
        selected,
        execute=True,
        jobs=args.jobs,
        backup_root=backup_root,
        manifest=manifest,
        manifest_path=manifest_path,
    )

    failed_commands = [cmd for cmd in manifest.get("commands", []) if cmd.get("returncode") not in (0, None)]
    manifest["failed_commands"] = failed_commands
    write_json(manifest_path, manifest)

    if args.verify_after:
        print("[INFO] post-run TabPFN verification...")
        manifest["post_verify_issues"] = post_verify_selected()
        print(f"[INFO] post_verify selected TabPFN-2.5 issues={len(manifest['post_verify_issues'])}")

    manifest["finished_at"] = datetime.now().strftime("%Y%m%d_%H%M%S")
    write_json(manifest_path, manifest)
    print(f"[INFO] manifest={manifest_path}")

    if failed_commands:
        print(f"[WARN] failed commands={len(failed_commands)}")
        return 1
    return 0 if not manifest.get("post_verify_issues") else 1


if __name__ == "__main__":
    raise SystemExit(main())
