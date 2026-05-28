from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
LLM_PYTHON = Path(r"C:\Users\HK\miniconda3\envs\llm\python.exe")
AUDIT_CSV = REPO_ROOT / "output" / "ood_summary_reports" / "Combined" / "data" / "final_combined_family_incomplete_audit.csv"
PROGRESS_FILE = REPO_ROOT / ".batch_progress_ood.json"

METHOD_TO_CONFIG = {
    "RandomCV": "experiment1_all_ml_models_random_cv_baseline",
    "Extrapolation": "experiment1_all_ml_models_extrapolation",
    "LOCO": "experiment1_all_ml_models_loco",
    "SparseXcluster": "experiment1_all_ml_models_sparse_x_cluster",
    "SparseXsingle": "experiment1_all_ml_models_sparse_x_single",
    "SparseYcluster": "experiment1_all_ml_models_sparse_y_cluster",
    "SparseYsingle": "experiment1_all_ml_models_sparse_y_single",
}
METHOD_TO_RAW = {
    "RandomCV": "random_cv_baseline",
    "Extrapolation": "target_extrapolation",
    "LOCO": "loco",
    "SparseXcluster": "sparse_x_cluster",
    "SparseXsingle": "sparse_x_single",
    "SparseYcluster": "sparse_y_cluster",
    "SparseYsingle": "sparse_y_single",
}
METHOD_ORDER = list(METHOD_TO_CONFIG)
MODEL_NAMES = ["xgboost", "sklearn_rf", "lightgbm", "mlp", "catboost"]
MODEL_DIR_NAMES = {
    "xgboost": "xgboost_results",
    "sklearn_rf": "sklearn_rf_results",
    "lightgbm": "lightgbm_results",
    "mlp": "mlp_results",
    "catboost": "catboost_results",
}
STEM_TO_ALLOY_TARGET = {
    "Al__aluminum__UTSMPa": ("Al", "UTS(MPa)"),
    "HEA__hea__UTSMPa": ("HEA", "UTS(MPa)"),
    "HEA__hea__YSMPa": ("HEA", "YS(MPa)"),
    "HEA__hea__Elpct": ("HEA", "El(%)"),
    "Steel__steel__UTSMPa": ("Steel", "UTS(MPa)"),
    "Steel__steel__Elpct": ("Steel", "El(%)"),
    "Ti__titanium__UTSMPa": ("Ti", "UTS(MPa)"),
    "Ti__titanium__Elpct": ("Ti", "El(%)"),
    "MatbenchSteel__matbench_steels_ood__yieldstrength": ("MatbenchSteels", "yield strength"),
}
DEFAULT_EXCLUDED_STEMS = {"Steel__steel__YSMPa"}


def task_group_key(task: dict[str, Any]) -> tuple[str, str]:
    return str(task["method"]), str(task["alloy"])


def same_executable(left: Path, right: Path) -> bool:
    try:
        return left.resolve().samefile(right.resolve())
    except Exception:
        return str(left.resolve()).lower() == str(right.resolve()).lower()


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except Exception:
        return str(path)


def safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(text)).strip("_")


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rerun incomplete Traditional-ML OOD cells used by final triptych figures.")
    parser.add_argument("--audit-csv", type=Path, default=AUDIT_CSV)
    parser.add_argument("--llm-python", type=Path, default=LLM_PYTHON)
    parser.add_argument("--no-relaunch", action="store_true")
    parser.add_argument("--execute", action="store_true", help="Actually move old dirs and rerun. Default is dry-run.")
    parser.add_argument("--jobs", type=int, default=1, help="Number of target-level commands to run concurrently.")
    parser.add_argument("--methods", nargs="+", choices=METHOD_ORDER, default=METHOD_ORDER)
    parser.add_argument("--stems", nargs="+", default=None, help="Optional task stems to include.")
    parser.add_argument("--include-hea-ys", action="store_true", help="Include HEA YS Traditional LOCO if present in audit.")
    parser.add_argument("--include-matbench-partial", action="store_true", default=True, help="Include Matbench partial 4/5 cells; default true.")
    parser.add_argument("--no-include-matbench-partial", dest="include_matbench_partial", action="store_false")
    parser.add_argument("--skip-backup", action="store_true", help="Do not move existing result dirs before rerun.")
    return parser.parse_args()


def relaunch(args: argparse.Namespace) -> int | None:
    if args.no_relaunch or same_executable(Path(sys.executable), args.llm_python):
        return None
    if not args.llm_python.exists():
        raise FileNotFoundError(f"llm Python not found: {args.llm_python}")
    cmd = [str(args.llm_python), str(Path(__file__).resolve()), *sys.argv[1:], "--no-relaunch"]
    print("[INFO] Relaunching with llm env:")
    print("       " + " ".join(f'\"{x}\"' if " " in x else x for x in cmd))
    return subprocess.run(cmd, cwd=str(REPO_ROOT), text=True).returncode


def load_audit_tasks(args: argparse.Namespace) -> list[dict[str, Any]]:
    if not args.audit_csv.exists():
        raise FileNotFoundError(f"audit CSV not found: {args.audit_csv}")
    rows: list[dict[str, Any]] = []
    with args.audit_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("model_family") != "Traditional":
                continue
            stem = str(row.get("task_stem", ""))
            method = str(row.get("ood_method", ""))
            if stem in DEFAULT_EXCLUDED_STEMS:
                continue
            if args.stems and stem not in set(args.stems):
                continue
            if method not in set(args.methods):
                continue
            if stem == "HEA__hea__YSMPa" and not args.include_hea_ys:
                # HEA YS is excluded unless explicitly requested, matching the earlier scope discussion.
                continue
            if stem not in STEM_TO_ALLOY_TARGET:
                continue
            model_count = int(float(row.get("model_count") or 0))
            expected = int(float(row.get("expected") or 5))
            if model_count >= expected:
                continue
            if stem.startswith("MatbenchSteel") and model_count > 0 and not args.include_matbench_partial:
                continue
            alloy, target = STEM_TO_ALLOY_TARGET[stem]
            rows.append(
                {
                    "task_stem": stem,
                    "method": method,
                    "alloy": alloy,
                    "target": target,
                    "model_count": model_count,
                    "expected": expected,
                }
            )
    # De-duplicate and stable-sort.
    seen = set()
    tasks = []
    for row in rows:
        key = (row["method"], row["alloy"], row["target"])
        if key in seen:
            continue
        seen.add(key)
        tasks.append(row)
    return sorted(tasks, key=lambda r: (METHOD_ORDER.index(r["method"]), r["alloy"], r["target"]))


def build_command(task: dict[str, Any]) -> list[str]:
    return [
        str(Path(sys.executable)),
        str(Path(__file__).resolve()),
        "--run-one",
    ]


def result_dir_for(task: dict[str, Any]) -> Path:
    import argparse as _argparse
    from src.pipelines.batch_configs_ood import ALLOY_CONFIGS_OOD, BATCH_CONFIGS_OOD, get_ood_method_meta
    from src.pipelines.run_batch_ood import build_result_dir

    config_name = METHOD_TO_CONFIG[task["method"]]
    args = _argparse.Namespace(**BATCH_CONFIGS_OOD[config_name])
    alloy_cfg = ALLOY_CONFIGS_OOD[task["alloy"]].copy()
    alloy_cfg["target_column"] = task["target"]
    return REPO_ROOT / build_result_dir(config_name, task["alloy"], alloy_cfg, args, get_ood_method_meta(METHOD_TO_RAW[task["method"]]))


def command_for_task(task: dict[str, Any]) -> list[str]:
    return [
        str(sys.executable),
        "-m",
        "src.pipelines.run_batch_ood",
        "--config",
        METHOD_TO_CONFIG[task["method"]],
        "--alloys",
        task["alloy"],
        "--targets",
        task["target"],
        "--resume",
        "--jobs",
        "1",
    ]


def build_run_groups(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse target-level audit rows to safe process-level commands.

    run_batch_ood operates at config/alloy level.  Grouping by method+alloy and
    passing all requested targets prevents duplicate concurrent processes from
    writing the same output/progress entries while still allowing safe
    parallelism across independent method/alloy groups.
    """

    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for task in tasks:
        key = task_group_key(task)
        group = grouped.setdefault(
            key,
            {
                "method": task["method"],
                "alloy": task["alloy"],
                "targets": [],
                "tasks": [],
            },
        )
        if task["target"] not in group["targets"]:
            group["targets"].append(task["target"])
        group["tasks"].append(task)

    groups = list(grouped.values())
    for group in groups:
        group["targets"] = sorted(group["targets"])
    return sorted(groups, key=lambda g: (METHOD_ORDER.index(g["method"]), g["alloy"]))


def command_for_group(group: dict[str, Any]) -> list[str]:
    return [
        str(sys.executable),
        "-m",
        "src.pipelines.run_batch_ood",
        "--config",
        METHOD_TO_CONFIG[group["method"]],
        "--alloys",
        group["alloy"],
        "--targets",
        *group["targets"],
        "--resume",
        "--jobs",
        "1",
    ]


def remove_progress_entries(tasks: list[dict[str, Any]], execute: bool) -> list[dict[str, Any]]:
    payload = load_json(PROGRESS_FILE)
    removed = []
    for task in tasks:
        cfg = METHOD_TO_CONFIG[task["method"]]
        cfg_payload = payload.setdefault(cfg, {})
        for model in MODEL_NAMES:
            key = f"{task['alloy']}_{task['target']}_{model}"
            old = cfg_payload.pop(key, None)
            if old is not None:
                removed.append({"config": cfg, "key": key, "old_status": old})
    if execute:
        write_json(PROGRESS_FILE, payload)
    return removed


def backup_result_dir(task: dict[str, Any], backup_root: Path, execute: bool, skip_backup: bool) -> dict[str, Any] | None:
    src = result_dir_for(task)
    if not src.exists() or skip_backup:
        return None
    dest = backup_root / "old_outputs" / src.relative_to(REPO_ROOT)
    if execute:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            dest = dest.with_name(dest.name + "__" + datetime.now().strftime("%H%M%S"))
        shutil.move(str(src), str(dest))
    return {"source": rel(src), "destination": rel(dest), "moved": bool(execute)}


def run_task(task: dict[str, Any], log_dir: Path, execute: bool) -> dict[str, Any]:
    cmd = command_for_task(task)
    log_path = log_dir / f"{safe_name(task['method'])}__{safe_name(task['alloy'])}__{safe_name(task['target'])}.log"
    record = {
        **task,
        "config": METHOD_TO_CONFIG[task["method"]],
        "command": cmd,
        "log_path": rel(log_path),
        "status": "dry_run" if not execute else "running",
        "returncode": None,
    }
    if not execute:
        return record
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        log.write("COMMAND: " + " ".join(cmd) + "\n\n")
        log.flush()
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), stdout=log, stderr=subprocess.STDOUT, text=True)
    record["returncode"] = int(proc.returncode)
    record["status"] = "success" if proc.returncode == 0 else "failed"
    return record


def run_group(group: dict[str, Any], log_dir: Path, execute: bool) -> dict[str, Any]:
    cmd = command_for_group(group)
    target_part = "__".join(safe_name(target) for target in group["targets"])
    log_path = log_dir / f"{safe_name(group['method'])}__{safe_name(group['alloy'])}__{target_part}.log"
    record = {
        **group,
        "config": METHOD_TO_CONFIG[group["method"]],
        "command": cmd,
        "log_path": rel(log_path),
        "status": "dry_run" if not execute else "running",
        "returncode": None,
    }
    if not execute:
        return record
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        log.write("COMMAND: " + " ".join(cmd) + "\n\n")
        log.flush()
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), stdout=log, stderr=subprocess.STDOUT, text=True)
    record["returncode"] = int(proc.returncode)
    record["status"] = "success" if proc.returncode == 0 else "failed"
    return record


def parse_run_one_marker() -> bool:
    # Reserved so accidental old dry-run manifests do not use this script as worker.
    return "--run-one" in sys.argv


def main() -> int:
    if parse_run_one_marker():
        raise SystemExit("--run-one is not a public mode")
    args = parse_args()
    code = relaunch(args)
    if code is not None:
        return int(code)

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_root = REPO_ROOT / "output" / "ood_repair_backups" / f"traditional_final_missing_repair_{ts}"
    manifest_path = backup_root / "repair_manifest.json"
    log_dir = backup_root / "logs"

    tasks = load_audit_tasks(args)
    run_groups = build_run_groups(tasks)
    manifest: dict[str, Any] = {
        "started_at": ts,
        "repo_root": str(REPO_ROOT),
        "python": sys.executable,
        "execute": bool(args.execute),
        "jobs": int(args.jobs),
        "audit_csv": str(args.audit_csv),
        "include_hea_ys": bool(args.include_hea_ys),
        "include_matbench_partial": bool(args.include_matbench_partial),
        "task_count": len(tasks),
        "tasks": tasks,
        "run_group_count": len(run_groups),
        "run_groups": run_groups,
        "backups": [],
        "progress_entries_removed": [],
        "commands": [],
    }
    write_json(manifest_path, manifest)

    print(f"[INFO] repo={REPO_ROOT}")
    print(f"[INFO] python={sys.executable}")
    print(f"[INFO] execute={args.execute} jobs={args.jobs}")
    print(f"[INFO] selected target-level tasks={len(tasks)}")
    for task in tasks:
        print(f"[TASK] {task['method']} | {task['alloy']} | {task['target']} ({task['model_count']}/{task['expected']})")
    print(f"[INFO] safe process groups={len(run_groups)}")
    for group in run_groups:
        print(f"[GROUP] {group['method']} | {group['alloy']} | targets={', '.join(group['targets'])}")
    print(f"[INFO] manifest={manifest_path}")
    if not tasks:
        return 0

    if args.execute:
        backup_root.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        backup = backup_result_dir(task, backup_root, execute=args.execute, skip_backup=args.skip_backup)
        if backup:
            manifest["backups"].append({**task, **backup})
            write_json(manifest_path, manifest)

    removed = remove_progress_entries(tasks, execute=args.execute)
    manifest["progress_entries_removed"] = removed
    write_json(manifest_path, manifest)

    if not args.execute:
        manifest["commands"] = [run_group(group, log_dir, execute=False) for group in run_groups]
        write_json(manifest_path, manifest)
        print("[DRY-RUN] No files moved and no commands executed. Add --execute to rerun.")
        return 0

    workers = max(1, int(args.jobs))
    if workers == 1:
        for group in run_groups:
            rec = run_group(group, log_dir, execute=True)
            manifest["commands"].append(rec)
            write_json(manifest_path, manifest)
            print(f"[DONE] {group['method']} | {group['alloy']} | {', '.join(group['targets'])} -> {rec['status']}")
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(run_group, group, log_dir, True): group for group in run_groups}
            for fut in as_completed(futs):
                group = futs[fut]
                try:
                    rec = fut.result()
                except Exception as exc:
                    rec = {**group, "status": "failed", "returncode": -1, "error": repr(exc)}
                manifest["commands"].append(rec)
                write_json(manifest_path, manifest)
                print(f"[DONE] {group['method']} | {group['alloy']} | {', '.join(group['targets'])} -> {rec['status']}")

    failed = [c for c in manifest["commands"] if c.get("returncode") not in (0, None)]
    manifest["failed_commands"] = failed
    manifest["finished_at"] = datetime.now().strftime("%Y%m%d_%H%M%S")
    write_json(manifest_path, manifest)
    print(f"[INFO] manifest={manifest_path}")
    if failed:
        print(f"[WARN] failed commands={len(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
