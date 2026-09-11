"""Compose the eight task-level Figure 4 PNGs into a supplementary plate.

The task figures are kept as the source panels.  This script only assembles
them into a publication-sized 2-column by 4-row plate and does not recalculate
or alter any result data.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


DEFAULT_TASK_ORDER = (
    "Al__aluminum__UTS_MPa",
    "HEA__hea__El",
    "HEA__hea__UTS_MPa",
    "MatbenchSteels__matbench_steels_ood__yield_strength",
    "Steel__steel__El",
    "Steel__steel__UTS_MPa",
    "Ti__titanium__El",
    "Ti__titanium__UTS_MPa",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assemble Figure 4 task PNGs into a supplementary figure plate."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(
            "D:/XJTU/Research/PHD/ood/Figures/figure4-data/combined"
        ),
        help="Directory containing the eight task-level PNG files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "D:/XJTU/Research/PHD/ood/Figures/figure4-data/combined/supplementary"
        ),
        help="Directory for the assembled PNG and PDF.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="Output resolution for the raster plate.",
    )
    parser.add_argument(
        "--write-pdf",
        action="store_true",
        help="Also write a PDF plate; omitted for the PNG-only publication bundle.",
    )
    return parser.parse_args()


def compose_figure(
    input_dir: Path,
    output_dir: Path,
    *,
    dpi: int = 600,
    write_pdf: bool = False,
) -> tuple[Path, Path | None]:
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = [input_dir / f"{task_id}.png" for task_id in DEFAULT_TASK_ORDER]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing task figure PNGs:\n" + "\n".join(missing)
        )

    # A portrait 2x4 plate keeps each original task figure large enough for
    # supplementary-material zooming while preserving the native panel layout.
    fig, axes = plt.subplots(
        4,
        2,
        figsize=(16.0, 27.5),
        gridspec_kw={"wspace": 0.025, "hspace": 0.075},
    )
    axes = axes.ravel()
    for index, (ax, task_id, path) in enumerate(zip(axes, DEFAULT_TASK_ORDER, paths)):
        image = mpimg.imread(path)
        ax.imshow(image, interpolation="none")
        ax.axis("off")
        ax.text(
            0.008,
            0.988,
            f"{chr(ord('a') + index)}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=13,
            fontweight="bold",
            color="black",
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.88,
                "pad": 1.5,
            },
            zorder=5,
        )

    fig.subplots_adjust(
        left=0.015,
        right=0.985,
        top=0.985,
        bottom=0.015,
        wspace=0.025,
        hspace=0.075,
    )
    png_path = output_dir / "figure4_supplementary_all_tasks.png"
    fig.savefig(png_path, dpi=dpi, facecolor="white")
    pdf_path: Path | None = None
    if write_pdf:
        pdf_path = output_dir / "figure4_supplementary_all_tasks.pdf"
        fig.savefig(pdf_path, facecolor="white")
    plt.close(fig)
    return png_path, pdf_path


def main() -> None:
    args = _parse_args()
    png_path, pdf_path = compose_figure(
        args.input_dir,
        args.output_dir,
        dpi=args.dpi,
        write_pdf=args.write_pdf,
    )
    print(f"PNG: {png_path}")
    if pdf_path is not None:
        print(f"PDF: {pdf_path}")


if __name__ == "__main__":
    main()
