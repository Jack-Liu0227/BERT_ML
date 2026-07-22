from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image


DEFAULT_INPUT = Path("output/imagegen/figure1_distribution_shift.png")
DEFAULT_OUTPUT_STEM = Path("output/imagegen/figure1_distribution_shift-v2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replace Figure 1 panel b with a target/feature-space severity map."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-stem", type=Path, default=DEFAULT_OUTPUT_STEM)
    return parser.parse_args()


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.linewidth": 0.8,
        }
    )


def draw_severity_map(ax: plt.Axes) -> None:
    quadrant_colors = {
        "in_distribution": "#F4F6F3",
        "target": "#FFF2E8",
        "feature": "#EAF5F3",
        "both": "#F4EFF5",
    }
    for x, y, key in [
        (0.0, 0.0, "in_distribution"),
        (0.5, 0.0, "target"),
        (0.0, 0.5, "feature"),
        (0.5, 0.5, "both"),
    ]:
        ax.add_patch(
            Rectangle(
                (x, y),
                0.5,
                0.5,
                facecolor=quadrant_colors[key],
                edgecolor="none",
                zorder=0,
            )
        )

    line_color = "#A7ADB2"
    ax.axvline(0.5, color=line_color, lw=0.9, ls=(0, (4, 3)), zorder=1)
    ax.axhline(0.5, color=line_color, lw=0.9, ls=(0, (4, 3)), zorder=1)

    labels = [
        (0.25, 0.73, "Covariate shift"),
        (0.75, 0.73, "Both shifts"),
        (0.25, 0.27, "In-distribution"),
        (0.75, 0.27, "Target shift"),
    ]
    for x, y, label in labels:
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=6.3,
            fontweight="bold",
            color="#171A1C",
            zorder=3,
        )

    candidate_color = "#F05A38"
    ax.scatter(
        [0.18],
        [0.16],
        s=62,
        marker="o",
        color="#174A8B",
        edgecolor="white",
        linewidth=0.8,
        zorder=4,
    )
    ax.scatter(
        [0.18, 0.82, 0.82],
        [0.84, 0.16, 0.84],
        s=68,
        marker="D",
        color=candidate_color,
        edgecolor="white",
        linewidth=0.8,
        zorder=4,
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    arrow_style = dict(arrowstyle="-|>", color="#111111", lw=0.85, mutation_scale=8)
    ax.annotate(
        "",
        xy=(1.07, -0.07),
        xytext=(-0.04, -0.07),
        xycoords="axes fraction",
        arrowprops=arrow_style,
        annotation_clip=False,
    )
    ax.annotate(
        "",
        xy=(-0.04, 1.07),
        xytext=(-0.04, -0.07),
        xycoords="axes fraction",
        arrowprops=arrow_style,
        annotation_clip=False,
    )

    ax.text(
        0.5,
        -0.18,
        "Target-space severity",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=6.5,
        fontweight="bold",
        color="#171A1C",
    )
    ax.text(
        -0.18,
        0.5,
        "Feature-space severity",
        transform=ax.transAxes,
        ha="center",
        va="center",
        rotation=90,
        fontsize=6.5,
        fontweight="bold",
        color="#171A1C",
    )
    ax.text(-0.04, -0.11, "low", transform=ax.transAxes, ha="center", va="top", fontsize=5.2)
    ax.text(1.05, -0.11, "high", transform=ax.transAxes, ha="center", va="top", fontsize=5.2)
    ax.text(-0.08, -0.01, "low", transform=ax.transAxes, ha="right", va="center", fontsize=5.2)
    ax.text(-0.08, 1.02, "high", transform=ax.transAxes, ha="right", va="center", fontsize=5.2)


def build_figure(input_path: Path) -> plt.Figure:
    with Image.open(input_path) as source:
        background = source.convert("RGB")

    aspect = background.height / background.width
    figure_width = 7.1
    fig = plt.figure(figsize=(figure_width, figure_width * aspect), facecolor="white")

    background_ax = fig.add_axes([0, 0, 1, 1], zorder=0)
    background_ax.imshow(background, extent=(0, 1, 0, 1), aspect="auto")
    background_ax.set_axis_off()
    background_ax.add_patch(
        Rectangle(
            (0.342, 0.045),
            0.326,
            0.91,
            transform=background_ax.transAxes,
            facecolor="white",
            edgecolor="none",
            zorder=2,
        )
    )

    fig.text(0.355, 0.925, "b", fontsize=11.5, fontweight="bold", ha="left", va="center")
    fig.text(
        0.512,
        0.925,
        "Two-axis severity map",
        fontsize=8.0,
        fontweight="bold",
        ha="center",
        va="center",
    )

    map_ax = fig.add_axes([0.392, 0.255, 0.245, 0.435], zorder=3)
    draw_severity_map(map_ax)
    return fig


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input figure not found: {args.input}")

    configure_style()
    args.output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig = build_figure(args.input)
    fig.savefig(args.output_stem.with_suffix(".svg"), bbox_inches=None, pad_inches=0)
    fig.savefig(args.output_stem.with_suffix(".pdf"), bbox_inches=None, pad_inches=0)
    fig.savefig(args.output_stem.with_suffix(".png"), dpi=300, bbox_inches=None, pad_inches=0)
    fig.savefig(
        args.output_stem.with_name(f"{args.output_stem.name}_print.png"),
        dpi=600,
        bbox_inches=None,
        pad_inches=0,
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
