from __future__ import annotations

import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Scripts.figures.ood_figures_3_4.build_training_size_task_figures import (  # noqa: E402
    parse_args as parse_common_args,
    run_build,
)


def parse_args(argv: list[str] | None = None):
    return parse_common_args(
        argv,
        default_output_dir=Path("output/training_size_ood/figures_no_bert"),
        description=(
            "Build eight-task MAE, degradation-ratio, and Random-CV outputs without BERT."
        ),
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_build(args, excluded_display_methods=("BERT-best",))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
