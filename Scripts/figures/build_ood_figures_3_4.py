from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Scripts.figures.ood_figures_3_4.audit import run_audit
from Scripts.figures.ood_figures_3_4.config import default_config_path, load_config
from Scripts.figures.ood_figures_3_4.discovery import run_discovery
from Scripts.figures.ood_figures_3_4.figure3 import run_figure3
from Scripts.figures.ood_figures_3_4.figure4 import run_figure4
from Scripts.figures.ood_figures_3_4.figure4_sweep import run_figure4_sweep
from Scripts.figures.ood_figures_3_4.manifest import write_manifest
from Scripts.figures.ood_figures_3_4.prepare_figure3 import run_prepare_figure3


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build maintainable OOD Figure 3/4 outputs.")
    parser.add_argument(
        "--stage",
        choices=["discover", "audit", "prepare-figure3", "figure3", "figure4-sweep", "figure4", "all"],
        default="audit",
    )
    parser.add_argument("--config", type=Path, default=default_config_path())
    parser.add_argument("--output-root", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config, output_root_override=args.output_root)

    stages = ["discover", "audit", "prepare-figure3", "figure3", "figure4-sweep", "figure4"] if args.stage == "all" else [args.stage]
    results = {}
    for stage in stages:
        if stage == "discover":
            results[stage] = run_discovery(cfg).to_manifest()
        elif stage == "audit":
            results[stage] = run_audit(cfg).to_manifest()
        elif stage == "prepare-figure3":
            results[stage] = run_prepare_figure3(cfg).to_manifest()
        elif stage == "figure3":
            results[stage] = run_figure3(cfg).to_manifest()
        elif stage == "figure4-sweep":
            results[stage] = run_figure4_sweep(cfg).to_manifest()
        elif stage == "figure4":
            results[stage] = run_figure4(cfg).to_manifest()

    write_manifest(
        cfg.output_root / "manifest.json",
        stage=args.stage,
        config_path=cfg.config_path,
        output_root=cfg.output_root,
        payload={"stages": results},
    )
    print(f"Wrote OOD Figure 3/4 outputs to {cfg.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
