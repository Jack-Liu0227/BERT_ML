"""Editable and programmatic sources for Figure 4 Panel A."""

from __future__ import annotations

import argparse
from pathlib import Path
from xml.etree import ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


COLORS = {
    "ink": "#17202A",
    "edge": "#26313D",
    "dataset_fill": "#F5F8FC",
    "dataset_stroke": "#31547B",
    "top_fill": "#FBE4E6",
    "top_stroke": "#C85159",
    "loco_fill": "#DDEEF4",
    "loco_stroke": "#348CA2",
    "random_fill": "#DDF0EA",
    "random_stroke": "#2F8B7D",
    "train_fill": "#E4ECFA",
    "train_stroke": "#3D6FB4",
    "neutral_fill": "#F3F5F7",
    "neutral_stroke": "#7A8491",
    "sweep_fill": "#EEF4FD",
    "sweep_stroke": "#61758E",
    "progress": "#4475B8",
    "progress_track": "#D8DEE8",
}


def draw_panel_a(ax: plt.Axes) -> None:
    """Draw the split protocol with the same structure as the draw.io source."""
    ax.set_axis_off()

    def box(
        xy: tuple[float, float],
        width: float,
        height: float,
        label: str,
        fill: str,
        stroke: str,
        *,
        fontsize: float = 8.3,
    ) -> None:
        ax.add_patch(
            FancyBboxPatch(
                xy,
                width,
                height,
                boxstyle="round,pad=0.007,rounding_size=0.018",
                facecolor=fill,
                edgecolor=stroke,
                linewidth=1.0,
            )
        )
        ax.text(
            xy[0] + width / 2,
            xy[1] + height / 2,
            label,
            ha="center",
            va="center",
            fontsize=fontsize,
            fontfamily="Arial",
            fontweight="bold",
            color=COLORS["ink"],
        )

    positions = {
        "full": (0.02, 0.33),
        "top": (0.20, 0.69),
        "remaining": (0.20, 0.40),
        "loco": (0.43, 0.69),
        "candidate": (0.43, 0.40),
        "random": (0.67, 0.49),
        "train": (0.67, 0.22),
    }
    box(positions["full"], 0.13, 0.34, "Full\ndataset", COLORS["dataset_fill"], COLORS["dataset_stroke"], fontsize=9)
    box(positions["top"], 0.19, 0.16, "Top-20% property\nextreme test set", COLORS["top_fill"], COLORS["top_stroke"])
    box(positions["remaining"], 0.19, 0.14, "Remaining ~80%", COLORS["neutral_fill"], COLORS["neutral_stroke"])
    box(positions["loco"], 0.21, 0.16, "Farthest LOCO\nfrom remaining ~80%", COLORS["loco_fill"], COLORS["loco_stroke"])
    box(positions["candidate"], 0.21, 0.14, "Candidate data\nafter LOCO removal", COLORS["neutral_fill"], COLORS["neutral_stroke"])
    box(positions["random"], 0.18, 0.14, "Random-CV\nheld-out test set", COLORS["random_fill"], COLORS["random_stroke"])
    box(positions["train"], 0.18, 0.14, "Training pool", COLORS["train_fill"], COLORS["train_stroke"])

    def right_center(name: str, width: float, height: float, y_fraction: float = 0.5) -> tuple[float, float]:
        x, y = positions[name]
        return x + width, y + height * y_fraction

    def left_center(name: str, height: float, y_fraction: float = 0.5) -> tuple[float, float]:
        x, y = positions[name]
        return x, y + height * y_fraction

    arrows = [
        (right_center("full", 0.13, 0.34, 0.72), left_center("top", 0.16)),
        (right_center("full", 0.13, 0.34, 0.35), left_center("remaining", 0.14)),
        (right_center("remaining", 0.19, 0.14, 0.75), left_center("loco", 0.16)),
        (right_center("remaining", 0.19, 0.14, 0.30), left_center("candidate", 0.14)),
        (right_center("candidate", 0.21, 0.14, 0.68), left_center("random", 0.14)),
        (right_center("candidate", 0.21, 0.14, 0.28), left_center("train", 0.14)),
    ]
    for start, end in arrows:
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=9,
                linewidth=1.0,
                color=COLORS["edge"],
                connectionstyle="angle3,angleA=0,angleB=90",
            )
        )

    ax.text(0.77, 0.91, "Training-size sweep", ha="center", fontsize=9.5, fontfamily="Arial", fontweight="bold", color=COLORS["ink"])
    for index, fraction in enumerate([0.2, 0.4, 0.6, 0.8, 1.0]):
        x = 0.88
        y = 0.76 - index * 0.145
        width, height = 0.105, 0.10
        box((x, y), width, height, f"{int(fraction * 100)}%", COLORS["sweep_fill"], COLORS["sweep_stroke"], fontsize=8.5)
        track_x, track_y, track_width = x + 0.012, y + 0.018, 0.055
        ax.add_patch(Rectangle((track_x, track_y), track_width, 0.012, facecolor=COLORS["progress_track"], edgecolor="none"))
        ax.add_patch(Rectangle((track_x, track_y), track_width * fraction, 0.012, facecolor=COLORS["progress"], edgecolor="none"))
        ax.add_patch(
            FancyArrowPatch(
                right_center("train", 0.18, 0.14),
                (x, y + height / 2),
                arrowstyle="-|>",
                mutation_scale=8,
                linewidth=0.8,
                color=COLORS["edge"],
                connectionstyle="angle3,angleA=0,angleB=90",
            )
        )

    ax.text(0.76, 0.17, "Nested deterministic prefixes", ha="center", fontsize=7.3, fontfamily="Arial", color="#586575")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Data split and training-size sweep", fontsize=12, fontfamily="Arial", fontweight="bold", pad=8)


def write_panel_a_drawio(output: Path) -> None:
    """Write an editable draw.io source with anchored, non-overlapping flow lines."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    mxfile = ET.Element(
        "mxfile",
        {
            "host": "app.diagrams.net",
            "modified": "2026-07-18T00:00:00.000Z",
            "agent": "Codex",
            "version": "30.3.11",
        },
    )
    diagram = ET.SubElement(mxfile, "diagram", {"id": "figure4-panel-a", "name": "Panel A"})
    graph = ET.SubElement(
        diagram,
        "mxGraphModel",
        {
            "dx": "1500",
            "dy": "760",
            "grid": "1",
            "gridSize": "10",
            "guides": "1",
            "tooltips": "1",
            "connect": "1",
            "arrows": "1",
            "fold": "1",
            "page": "1",
            "pageScale": "1",
            "pageWidth": "1500",
            "pageHeight": "760",
            "math": "0",
            "shadow": "0",
        },
    )
    root = ET.SubElement(graph, "root")
    ET.SubElement(root, "mxCell", {"id": "0"})
    ET.SubElement(root, "mxCell", {"id": "1", "parent": "0"})

    def vertex(
        cell_id: str,
        value: str,
        x: float,
        y: float,
        width: float,
        height: float,
        style: str,
    ) -> None:
        cell = ET.SubElement(
            root,
            "mxCell",
            {
                "id": cell_id,
                "value": value,
                "style": style,
                "vertex": "1",
                "parent": "1",
            },
        )
        ET.SubElement(
            cell,
            "mxGeometry",
            {
                "x": str(x),
                "y": str(y),
                "width": str(width),
                "height": str(height),
                "as": "geometry",
            },
        )

    def edge(
        cell_id: str,
        source: str,
        target: str,
        *,
        exit_y: float = 0.5,
        entry_y: float = 0.5,
    ) -> None:
        style = (
            "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=18;"
            "html=1;strokeColor=#26313D;strokeWidth=2;endArrow=block;endFill=1;"
            f"exitX=1;exitY={exit_y};exitDx=0;exitDy=0;"
            f"entryX=0;entryY={entry_y};entryDx=0;entryDy=0;"
        )
        cell = ET.SubElement(
            root,
            "mxCell",
            {
                "id": cell_id,
                "edge": "1",
                "parent": "1",
                "source": source,
                "target": target,
                "style": style,
            },
        )
        ET.SubElement(cell, "mxGeometry", {"relative": "1", "as": "geometry"})

    box_style = (
        "rounded=1;arcSize=14;whiteSpace=wrap;html=1;fillColor={fill};"
        "strokeColor={stroke};strokeWidth=2;fontColor=#17202A;fontSize=16;"
        "fontStyle=1;fontFamily=Arial;align=center;verticalAlign=middle;spacing=8;"
    )
    text_style = (
        "text;html=1;strokeColor=none;fillColor=none;fontColor=#17202A;"
        "fontFamily=Arial;align=center;verticalAlign=middle;"
    )
    vertex("title", "Data split and training-size sweep", 360, 18, 760, 44, text_style + "fontSize=24;fontStyle=1;")
    for cell_id, label, x in [
        ("step-1", "1  Property-extreme holdout", 255),
        ("step-2", "2  Distant-cluster holdout", 585),
        ("step-3", "3  Random-CV holdout", 905),
        ("step-4", "4  Training-size sweep", 1215),
    ]:
        vertex(cell_id, label, x, 72, 280, 30, text_style + "fontSize=13;fontStyle=1;fontColor=#536273;")

    vertex(
        "full",
        "Full dataset",
        35,
        275,
        175,
        210,
        box_style.format(fill=COLORS["dataset_fill"], stroke=COLORS["dataset_stroke"])
        + "shape=cylinder3;boundedLbl=1;backgroundOutline=1;",
    )
    vertex("top", "Top-20% property<br>extreme test set", 285, 130, 250, 120, box_style.format(fill=COLORS["top_fill"], stroke=COLORS["top_stroke"]))
    vertex("remaining", "Remaining ~80%", 285, 365, 250, 105, box_style.format(fill=COLORS["neutral_fill"], stroke=COLORS["neutral_stroke"]))
    vertex("loco", "Farthest LOCO<br>from remaining ~80%", 610, 130, 270, 120, box_style.format(fill=COLORS["loco_fill"], stroke=COLORS["loco_stroke"]))
    vertex("candidate", "Candidate data<br>after LOCO removal", 610, 365, 270, 105, box_style.format(fill=COLORS["neutral_fill"], stroke=COLORS["neutral_stroke"]))
    vertex("random", "Random-CV<br>held-out test set", 930, 285, 250, 105, box_style.format(fill=COLORS["random_fill"], stroke=COLORS["random_stroke"]))
    vertex("train", "Training pool", 930, 475, 250, 105, box_style.format(fill=COLORS["train_fill"], stroke=COLORS["train_stroke"]))
    vertex("prefix-note", "Nested deterministic prefixes", 930, 595, 250, 28, text_style + "fontSize=12;fontColor=#586575;")
    vertex("sweep-hub", "", 1205, 518, 12, 12, "ellipse;html=1;fillColor=#26313D;strokeColor=#26313D;")

    edge("edge-full-top", "full", "top", exit_y=0.28)
    edge("edge-full-remaining", "full", "remaining", exit_y=0.72)
    edge("edge-remaining-loco", "remaining", "loco", exit_y=0.25)
    edge("edge-remaining-candidate", "remaining", "candidate", exit_y=0.75)
    edge("edge-candidate-random", "candidate", "random", exit_y=0.25)
    edge("edge-candidate-train", "candidate", "train", exit_y=0.75)
    edge("edge-train-hub", "train", "sweep-hub")

    card_style = (
        "rounded=1;arcSize=16;whiteSpace=wrap;html=1;fillColor=#EEF4FD;"
        "strokeColor=#61758E;strokeWidth=2;fontColor=#17202A;fontSize=16;"
        "fontStyle=1;fontFamily=Arial;align=right;verticalAlign=middle;spacingRight=18;"
    )
    track_style = "rounded=1;arcSize=50;html=1;fillColor=#D8DEE8;strokeColor=none;"
    fill_style = "rounded=1;arcSize=50;html=1;fillColor=#4475B8;strokeColor=none;"
    for index, fraction in enumerate([0.2, 0.4, 0.6, 0.8, 1.0]):
        y = 105 + index * 116
        card_id = f"sweep-card-{index}"
        vertex(card_id, f"{int(fraction * 100)}%", 1260, y, 220, 82, card_style)
        vertex(f"sweep-track-{index}", "", 1282, y + 49, 105, 12, track_style)
        vertex(f"sweep-fill-{index}", "", 1282, y + 49, 105 * fraction, 12, fill_style)
        edge(f"edge-hub-sweep-{index}", "sweep-hub", card_id)

    ET.indent(mxfile, space="  ")
    output.write_text(ET.tostring(mxfile, encoding="unicode"), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Render editable Figure 4 Panel A")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/training_size_ood/figures/sources/figure4_panel_a_source"),
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.0, 4.6), constrained_layout=True)
    draw_panel_a(ax)
    for extension in ["png", "pdf", "svg"]:
        fig.savefig(args.output.with_suffix(f".{extension}"), dpi=600, bbox_inches="tight")
    write_panel_a_drawio(args.output.with_suffix(".drawio"))
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
