"""Render a paper-style ``unet_topo_fusion`` schema with matplotlib.

The visual language is intentionally closer to hand-authored model diagrams:
soft pastel blocks, dark ink-like outlines, explicit feature-map glyphs, and a
manually composed layout rather than an auto-routed graph.

Examples:
    >>> parser = build_arg_parser()
    >>> args = parser.parse_args(["--output", "/tmp/schema.svg"])
    >>> args.output.suffix
    '.svg'
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon  # noqa: E402

INK = "#2F4B73"
MAIN_FILL = "#F7E8D7"
BLUE_FILL = "#CBD7EF"
GREEN_FILL = "#D8E7D1"
MAP_FILL = "#FFF1CC"
WHITE = "#FFFFFF"


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the schema renderer.

    Returns:
        argparse.ArgumentParser: Configured parser.
    """

    parser = argparse.ArgumentParser(
        description="Render a paper-style unet_topo_fusion architecture figure."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/unet_topo_fusion_schema.svg"),
        help="Primary output path. The file suffix selects the image format.",
    )
    parser.add_argument(
        "--also-formats",
        nargs="*",
        default=["pdf"],
        help="Extra sibling formats to render, for example: pdf png",
    )
    return parser


def _add_round_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    *,
    facecolor: str,
    fontsize: int = 15,
    weight: str = "normal",
    zorder: int = 3,
) -> FancyBboxPatch:
    """Draw a rounded rectangle with centered text.

    Args:
        ax (plt.Axes): Target axes.
        x (float): Left coordinate.
        y (float): Bottom coordinate.
        w (float): Width.
        h (float): Height.
        text (str): Box label.
        facecolor (str): Fill color.
        fontsize (int): Font size.
        weight (str): Font weight.
        zorder (int): Drawing order.

    Returns:
        FancyBboxPatch: Added patch.
    """

    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.16",
        linewidth=1.8,
        edgecolor=INK,
        facecolor=facecolor,
        zorder=zorder,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        color="#1F2A37",
        family="DejaVu Sans",
        zorder=zorder + 1,
    )
    return patch


def _add_trapezoid(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    *,
    facecolor: str,
) -> Polygon:
    """Draw a DINO-style trapezoid block.

    Args:
        ax (plt.Axes): Target axes.
        x (float): Left coordinate.
        y (float): Bottom coordinate.
        w (float): Width.
        h (float): Height.
        text (str): Center label.
        facecolor (str): Fill color.

    Returns:
        Polygon: Added patch.
    """

    pts = [
        (x + 0.18 * w, y),
        (x + 0.82 * w, y),
        (x + w, y + h),
        (x, y + h),
    ]
    patch = Polygon(
        pts,
        closed=True,
        facecolor=facecolor,
        edgecolor=INK,
        linewidth=1.8,
        zorder=3,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=18,
        color="#1F2A37",
        family="DejaVu Sans",
        zorder=4,
    )
    return patch


def _add_feature_map(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    label: str | None = None,
    label_position: str = "right",
    zorder: int = 2,
) -> None:
    """Draw a small stacked feature-map glyph.

    Args:
        ax (plt.Axes): Target axes.
        x (float): Left coordinate.
        y (float): Bottom coordinate.
        w (float): Width.
        h (float): Height.
        label (str | None): Optional text label.
        label_position (str): One of ``right``, ``above``, or ``below``.
        zorder (int): Drawing order.
    """

    dx = 0.12 * w
    dy = 0.12 * h
    back = Polygon(
        [
            (x + dx, y + dy),
            (x + w + dx, y + dy),
            (x + w + dx, y + h + dy),
            (x + dx, y + h + dy),
        ],
        closed=True,
        facecolor=WHITE,
        edgecolor=INK,
        linewidth=1.6,
        zorder=zorder,
    )
    front = Polygon(
        [(x, y), (x + w, y), (x + w, y + h), (x, y + h)],
        closed=True,
        facecolor=MAP_FILL,
        edgecolor=INK,
        linewidth=1.6,
        zorder=zorder + 1,
    )
    ax.add_patch(back)
    ax.add_patch(front)
    if label:
        if label_position == "above":
            tx, ty, ha, va = x + w + 0.12, y + h + 0.16, "left", "bottom"
        elif label_position == "below":
            tx, ty, ha, va = x + w * 0.55, y - 0.22, "center", "top"
        else:
            tx, ty, ha, va = x + w + 0.42, y + h * 0.52, "left", "center"
        ax.text(
            tx,
            ty,
            label,
            ha=ha,
            va=va,
            fontsize=12,
            color="#2B3440",
            family="DejaVu Sans",
        )


def _arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    dashed: bool = False,
    connection: str = "arc3,rad=0.0",
    linewidth: float = 1.8,
    zorder: int = 1,
) -> None:
    """Draw an arrow between two points.

    Args:
        ax (plt.Axes): Target axes.
        start (tuple[float, float]): Start point.
        end (tuple[float, float]): End point.
        dashed (bool): Use dashed styling.
        connection (str): Matplotlib connection style.
        linewidth (float): Stroke width for the arrow.
        zorder (int): Drawing order.
    """

    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=linewidth,
        linestyle=(0, (3, 3)) if dashed else "solid",
        color=INK,
        connectionstyle=connection,
        zorder=zorder,
    )
    ax.add_patch(patch)


def _text(ax: plt.Axes, x: float, y: float, text: str, *, size: int = 16) -> None:
    """Draw standalone text.

    Args:
        ax (plt.Axes): Target axes.
        x (float): X coordinate.
        y (float): Y coordinate.
        text (str): Text content.
        size (int): Font size.
    """

    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=size,
        color="#1F2A37",
        family="DejaVu Sans",
    )


def draw_schema() -> plt.Figure:
    """Compose the publication-style architecture figure.

    Returns:
        plt.Figure: Rendered figure object.
    """

    fig, ax = plt.subplots(figsize=(16.2, 7.1))
    ax.set_xlim(0, 18.3)
    ax.set_ylim(0.0, 7.25)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # Input thumbnail with a simple aerial-tile motif.
    img_box = FancyBboxPatch(
        (0.9, 5.5),
        1.15,
        1.15,
        boxstyle="round,pad=0.0,rounding_size=0.04",
        linewidth=1.0,
        edgecolor="#9AA4B2",
        facecolor="#EBEEF2",
        zorder=1,
    )
    ax.add_patch(img_box)
    ax.add_patch(plt.Circle((1.28, 6.18), 0.22, color="#384B3C", zorder=2))
    ax.add_patch(plt.Circle((1.56, 5.92), 0.19, color="#435D47", zorder=2))
    ax.plot([0.95, 1.95], [5.64, 6.55], color="#8A8F96", linewidth=1.1, zorder=3)
    ax.plot([1.05, 1.86], [6.42, 5.68], color="#C9CDD3", linewidth=4.0, zorder=2)
    ax.plot([1.05, 1.86], [6.42, 5.68], color="#8A8F96", linewidth=0.9, zorder=3)
    _text(ax, 1.48, 6.95, "Image", size=16)

    # Output thumbnail.
    out_box = FancyBboxPatch(
        (16.6, 5.5),
        1.15,
        1.15,
        boxstyle="round,pad=0.0,rounding_size=0.04",
        linewidth=1.0,
        edgecolor="#9AA4B2",
        facecolor="#EBEEF2",
        zorder=1,
    )
    ax.add_patch(out_box)
    ax.add_patch(plt.Circle((17.18, 6.03), 0.28, color="#E9D66F", zorder=2))
    ax.add_patch(plt.Circle((17.16, 6.00), 0.19, color="#B45467", zorder=3))
    ax.add_patch(plt.Circle((17.22, 5.98), 0.11, color="#67B0A7", zorder=4))
    _text(ax, 17.18, 6.95, "Mask", size=16)

    # Backbone + adapter stack.
    _add_trapezoid(ax, 0.8, 3.32, 1.75, 1.42, "DINO\nv3", facecolor=BLUE_FILL)
    ax.plot([2.95, 4.3], [1.75, 1.75], color=INK, linewidth=1.6, linestyle=(0, (3, 3)))
    ax.plot([2.95, 2.95], [1.75, 5.05], color=INK, linewidth=1.6, linestyle=(0, (3, 3)))
    ax.plot([4.3, 4.3], [1.75, 5.05], color=INK, linewidth=1.6, linestyle=(0, (3, 3)))
    ax.plot([2.95, 4.3], [5.05, 5.05], color=INK, linewidth=1.6, linestyle=(0, (3, 3)))
    _text(ax, 3.62, 1.28, "LoRA Adapters", size=15)

    adapter_centers = [(3.62, 4.52), (3.62, 3.68), (3.62, 2.84), (3.62, 2.0)]
    for cx, cy in adapter_centers:
        _add_round_box(
            ax,
            cx - 0.34,
            cy - 0.22,
            0.68,
            0.44,
            "",
            facecolor=BLUE_FILL,
            fontsize=1,
        )
    for (_, cy1), (_, cy2) in zip(adapter_centers, adapter_centers[1:]):
        _arrow(ax, (3.62, cy1 - 0.22), (3.62, cy2 + 0.22))

    # Input routes.
    _arrow(ax, (1.48, 5.5), (1.48, 4.77))
    _arrow(ax, (2.05, 6.07), (9.2, 6.07), connection="angle,angleA=0,angleB=90,rad=0")
    _arrow(ax, (1.9, 3.32), (3.3, 4.7), connection="arc3,rad=-0.08")
    _arrow(ax, (1.78, 3.32), (3.3, 3.86), connection="arc3,rad=-0.04")
    _arrow(ax, (1.66, 3.32), (3.3, 3.02), connection="arc3,rad=0.03")
    _arrow(ax, (1.54, 3.32), (3.3, 2.18), connection="arc3,rad=0.09")

    # Fusion and tap synthesis.
    _add_round_box(
        ax, 5.05, 3.78, 1.42, 0.62, "Layer\nFusion", facecolor=MAIN_FILL, fontsize=15
    )
    for _, cy in adapter_centers:
        _arrow(ax, (3.96, cy), (5.05, 4.09))
    _add_feature_map(ax, 6.9, 3.67, 0.18, 0.74)
    ax.text(
        6.82,
        3.46,
        "Fused map",
        ha="center",
        va="top",
        fontsize=11,
        color="#2B3440",
        family="DejaVu Sans",
    )
    _arrow(ax, (6.47, 4.09), (6.9, 4.04))

    tap_boxes = [
        ("Shallow tap\n32, H/8, W/8", 7.55, 4.36),
        ("Mid tap\n64, H/16, W/16", 7.55, 2.96),
        ("Deep tap\n64, H/32, W/32", 7.55, 1.56),
    ]
    tap_maps = [(9.02, 4.33), (9.02, 2.93), (9.02, 1.53)]
    for (label, x, y), (mx, my) in zip(tap_boxes, tap_maps):
        _add_round_box(ax, x, y, 1.6, 0.76, label, facecolor=MAIN_FILL, fontsize=13)
        _add_feature_map(ax, mx + 0.15, my, 0.16, 0.74)
    _arrow(ax, (7.08, 4.04), (7.55, 4.74))
    _arrow(ax, (7.08, 4.04), (7.55, 3.34))
    _arrow(ax, (7.08, 4.04), (7.55, 1.94))

    # Decoder ladder.
    _add_round_box(
        ax, 10.55, 1.62, 1.48, 0.6, "Decoder", facecolor=GREEN_FILL, fontsize=15
    )
    _add_round_box(
        ax, 12.62, 3.02, 1.48, 0.6, "Decoder", facecolor=GREEN_FILL, fontsize=15
    )
    _add_round_box(
        ax, 14.68, 4.42, 1.48, 0.6, "Decoder", facecolor=GREEN_FILL, fontsize=15
    )
    _add_round_box(
        ax, 16.15, 4.42, 1.08, 0.6, "Refine", facecolor=MAIN_FILL, fontsize=15
    )

    _add_feature_map(
        ax, 12.34, 1.48, 0.15, 0.72, label="256, H/8, W/8", label_position="below"
    )
    _add_feature_map(
        ax, 14.42, 2.88, 0.15, 0.72, label="128, H/4, W/4", label_position="below"
    )
    _add_feature_map(ax, 16.42, 4.18, 0.15, 0.72)
    _add_feature_map(ax, 17.72, 4.2, 0.16, 0.76)
    ax.text(
        16.48,
        3.86,
        "64, H/2, W/2",
        ha="center",
        va="top",
        fontsize=12,
        color="#2B3440",
        family="DejaVu Sans",
    )
    ax.text(
        17.82,
        3.9,
        "32, H, W",
        ha="center",
        va="top",
        fontsize=12,
        color="#2B3440",
        family="DejaVu Sans",
    )

    _arrow(ax, (9.18, 1.9), (10.55, 1.92))
    _arrow(ax, (9.18, 3.3), (12.62, 3.32))
    _arrow(ax, (9.18, 4.7), (14.68, 4.72))
    _arrow(ax, (12.03, 1.92), (12.34, 1.84))
    _arrow(ax, (12.42, 2.2), (12.42, 3.0))
    _arrow(ax, (14.1, 3.32), (14.42, 3.24))
    _arrow(ax, (14.5, 3.6), (14.5, 4.4))
    _arrow(ax, (16.16, 4.72), (16.42, 4.54))
    _arrow(ax, (16.16, 4.72), (16.15, 4.72))
    _arrow(ax, (17.23, 4.72), (17.72, 4.58))

    # RGB priors.
    _add_round_box(
        ax, 8.35, 5.34, 1.72, 0.62, "Spatial Priors", facecolor=MAIN_FILL, fontsize=15
    )
    _add_feature_map(ax, 10.5, 5.22, 0.15, 0.66)
    _add_feature_map(ax, 11.1, 5.22, 0.15, 0.66)
    ax.text(
        10.58,
        4.98,
        "H/4",
        ha="center",
        va="top",
        fontsize=11,
        color="#2B3440",
        family="DejaVu Sans",
    )
    ax.text(
        11.18,
        4.98,
        "H/2",
        ha="center",
        va="top",
        fontsize=11,
        color="#2B3440",
        family="DejaVu Sans",
    )
    _arrow(ax, (9.2, 6.07), (9.2, 5.82))
    _arrow(ax, (10.07, 5.65), (10.5, 5.55))
    _arrow(ax, (10.07, 5.65), (11.1, 5.55))
    _arrow(
        ax,
        (10.58, 5.25),
        (13.36, 3.62),
        dashed=True,
        connection="arc3,rad=-0.06",
        linewidth=1.5,
    )
    _arrow(ax, (11.18, 5.25), (15.42, 5.02), dashed=True, linewidth=1.5)

    # Deep supervision heads at H/8.
    _add_round_box(
        ax, 10.15, 5.96, 1.44, 0.54, "Aux logits", facecolor=GREEN_FILL, fontsize=14
    )
    _add_round_box(
        ax,
        11.72,
        5.96,
        1.78,
        0.54,
        "Skeleton logits",
        facecolor=GREEN_FILL,
        fontsize=14,
    )
    ax.plot(
        [12.42, 12.42],
        [2.2, 5.55],
        color=INK,
        linewidth=1.45,
        linestyle=(0, (3, 3)),
        zorder=1,
    )
    _arrow(
        ax,
        (12.42, 5.55),
        (10.87, 5.96),
        dashed=True,
        connection="arc3,rad=0.04",
        linewidth=1.45,
    )
    _arrow(
        ax,
        (12.42, 5.55),
        (12.6, 5.96),
        dashed=True,
        connection="arc3,rad=0.0",
        linewidth=1.45,
    )

    # Boundary refinement.
    _add_round_box(
        ax, 15.2, 5.86, 1.6, 0.58, "Edge + Gate", facecolor=MAIN_FILL, fontsize=15
    )
    _arrow(ax, (16.7, 5.02), (16.0, 5.86), connection="arc3,rad=0.08")
    _arrow(
        ax,
        (16.8, 5.86),
        (17.02, 5.04),
        dashed=True,
        connection="arc3,rad=-0.08",
        linewidth=1.5,
    )

    # Output to mask.
    _arrow(ax, (17.18, 5.5), (17.18, 5.02))

    fig.tight_layout(pad=0.2)
    return fig


def render_outputs(primary_output: Path, extra_formats: list[str]) -> list[Path]:
    """Render the figure to the requested output formats.

    Args:
        primary_output (Path): Primary output path.
        extra_formats (list[str]): Extra sibling format suffixes.

    Returns:
        list[Path]: Rendered paths.
    """

    fig = draw_schema()
    rendered: list[Path] = []
    primary_output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(primary_output, dpi=300, bbox_inches="tight", facecolor="white")
    if primary_output.suffix.lower() == ".svg":
        _compact_svg(primary_output)
    rendered.append(primary_output)

    seen = {primary_output.suffix.lower().lstrip(".")}
    for fmt in extra_formats:
        clean_fmt = fmt.strip().lower().lstrip(".")
        if not clean_fmt or clean_fmt in seen:
            continue
        sibling = primary_output.with_suffix(f".{clean_fmt}")
        fig.savefig(sibling, dpi=300, bbox_inches="tight", facecolor="white")
        if sibling.suffix.lower() == ".svg":
            _compact_svg(sibling)
        rendered.append(sibling)
        seen.add(clean_fmt)

    plt.close(fig)
    return rendered


def _compact_svg(path: Path) -> None:
    """Collapse SVG whitespace so generated figures stay within repo limits.

    Args:
        path (Path): SVG file to compact in place.
    """

    text = path.read_text(encoding="utf-8")
    compact = re.sub(r">\s+<", "><", text.strip())
    path.write_text(compact + "\n", encoding="utf-8")


def main() -> int:
    """Run the CLI entrypoint.

    Returns:
        int: Process exit code.
    """

    args = build_arg_parser().parse_args()
    rendered = render_outputs(args.output, list(args.also_formats))
    print("Rendered schema to:")
    for path in rendered:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
