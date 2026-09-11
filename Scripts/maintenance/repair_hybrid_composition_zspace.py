from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Scripts.figures import plot_w_error_relationships as plotter
from Scripts.ood_w_error_rebuild import publish_hybrid_figures as publisher
from Scripts.ood_w_error_rebuild.common import read_csv, write_csv, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repair published Hybrid Z-space rows for sources aligned by composition."
    )
    parser.add_argument("--full-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--embedding-data-dir", type=Path, required=True)
    parser.add_argument("--subset", default="combined")
    parser.add_argument("--task-id", default="Matbench Steel-YS")
    parser.add_argument("--model", default="GPT-5.4")
    parser.add_argument("--formats", nargs="+", default=["png"], choices=["png", "pdf", "svg"])
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_dir = publisher.hybrid_csv_dir(args.full_root.resolve(), args.subset)
    raw_samples = read_csv(csv_dir / "w_error_samples_long.csv")
    task_ids = pd.Series(
        [
            plotter.task_id_from_values(row["alloy_family"], row["property"])
            for _, row in raw_samples.iterrows()
        ],
        index=raw_samples.index,
    )
    selected = raw_samples[
        task_ids.eq(args.task_id) & raw_samples["model"].astype(str).eq(args.model)
    ].copy()
    if selected.empty:
        raise ValueError(f"No raw samples matched {args.task_id} / {args.model}")

    inventory = read_csv(csv_dir / "w_source_inventory.csv")
    source_dirs = set(selected["source_split_dir"].astype(str))
    inventory = inventory[inventory["source_split_dir"].astype(str).isin(source_dirs)].copy()
    repaired = publisher.build_three_space_joined(
        selected,
        inventory,
        args.embedding_data_dir.resolve(),
    )
    repaired_z = repaired[
        repaired["space"].astype(str).eq("Z-space")
        & pd.to_numeric(repaired["sample_w_contribution"], errors="coerce").notna()
    ]
    if len(repaired_z) != len(selected):
        raise RuntimeError(
            f"Expected {len(selected)} repaired Z rows, found {len(repaired_z)}"
        )

    output_dir = args.output_dir.resolve()
    joined_path = output_dir / "csv" / "w_error_samples_long.csv"
    joined = read_csv(joined_path)
    replace_mask = joined["task_id"].astype(str).eq(args.task_id) & joined["model"].astype(str).eq(args.model)
    joined = pd.concat([joined.loc[~replace_mask], repaired], ignore_index=True, sort=False)
    write_csv(joined, joined_path)

    phase_wide = plotter.build_phase_diagram_samples_wide(
        joined.drop(columns=["__w_row_id"], errors="ignore")
    )
    write_csv(phase_wide, output_dir / "csv" / "phase_diagram_samples_wide.csv")
    write_csv(plotter.build_correlations(joined), output_dir / "csv" / "w_error_correlations.csv")

    repaired_figures = pd.concat(
        [
            plotter.plot_task_model_figures(repaired, output_dir, args.formats, args.dpi),
            plotter.plot_method_figures(repaired, output_dir, args.formats, args.dpi),
            plotter.plot_phase_diagram_figures(
                phase_wide[phase_wide["task_id"].astype(str).eq(args.task_id)],
                output_dir,
                args.formats,
                args.dpi,
            ),
        ],
        ignore_index=True,
        sort=False,
    )
    manifest_path = output_dir / "csv" / "figure_manifest.csv"
    manifest = read_csv(manifest_path)
    refreshed_paths = set(repaired_figures["figure"].astype(str))
    manifest = manifest[~manifest["figure"].astype(str).isin(refreshed_paths)].copy()
    manifest = pd.concat([manifest, repaired_figures], ignore_index=True, sort=False)
    write_csv(manifest[plotter.FIGURE_MANIFEST_COLUMNS], manifest_path)

    audit = {
        "status": "ok",
        "subset": args.subset,
        "task_id": args.task_id,
        "model": args.model,
        "raw_rows": int(len(selected)),
        "repaired_z_rows": int(len(repaired_z)),
        "refreshed_figures": int(len(repaired_figures)),
    }
    write_json(audit, output_dir / "csv" / "composition_zspace_repair_audit.json")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
