#!/usr/bin/env python3
r"""Scatter plots of the per-code NN-vs-FEM errors of a generalization run.

Reads the tab-separated metrics file written by
evaluate_dirichlet_generalization.py (one row per evaluated Dirichlet code
vector) and draws one marker per row plus a dashed line at the mean:

  * the MSE plot is ALWAYS written (to -o/--output, default: INPUT with the
    suffix .png);
  * the rel_l2 plot is written in addition when the file has a ``rel_l2``
    column (to --out-rel, default: <INPUT stem>_rel_l2.png next to INPUT).

Metrics as defined by the evaluator, at the mesh vertices:
    mse    = mean((u_nn - u_fem)^2)
    rel_l2 = sqrt(sum((u_nn - u_fem)^2) / sum(u_fem^2))

Input format: blank lines and lines whose first non-whitespace character is
'#' (the evaluator's '# key: value' metadata) are skipped; the first remaining
line is the header.  The columns ``sample_index`` and ``mse`` are required,
``rel_l2`` is optional, all other columns (``source_<i>_value``) are ignored.

With --plot-frac < 1 only a random subset of the rows is plotted (seeded by
--plot-seed, same subset for both plots); the mean line then refers to the
plotted subset.

Rendering: plotly is optional.  When it can be imported the PNG is exported
with Plotly/Kaleido, with a fallback to an equivalent Matplotlib figure if that
export fails.  When importing plotly raises ImportError (normally because it
is not installed) the Matplotlib figure is drawn directly and the 'Wrote ...'
line says so; matplotlib alone is therefore enough (only --show needs plotly).

Pipeline: last step of the generalization study.  train_screen.py trains the
model, prepare_dirichlet_splits.py (optional) builds the train / interp /
extrap code lists, evaluate_dirichlet_generalization.py writes the metrics TSV
and this script plots it.

Usage (from the repository root):

    python plot_generalization_mse.py results/generalization/dc/eval/metrics_interp.txt

    python plot_generalization_mse.py results/generalization/dc/eval/metrics_interp.txt \
        --out-rel results/generalization/dc/eval/rel_l2_interp.png \
        --plot-frac 1.0 --title-rel "NN vs FEM rel_l2 (interp)"

The second command writes results/generalization/dc/eval/metrics_interp.png
(MSE, default name) and results/generalization/dc/eval/rel_l2_interp.png.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable

try:
    import plotly.graph_objects as go
except ImportError:
    # plotly is optional: if it cannot be imported (normally: not installed)
    # the figures are drawn with Matplotlib.
    go = None


def parse_args() -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(
        description=(
            "Read a TSV produced by evaluate_dirichlet_generalization.py and "
            "write scatter-plot PNG file(s): with Plotly/Kaleido when plotly "
            "can be imported (Matplotlib if that export fails), with "
            "Matplotlib when plotly cannot be imported. Lines whose "
            "first non-whitespace character is '#' are treated as metadata."
        )
    )
    parser.add_argument("input", type=Path, help="input evaluation TSV")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help=(
            "output PNG for mse, always written "
            "(default: INPUT with the suffix .png)"
        ),
    )
    parser.add_argument(
        "--out-rel",
        type=Path,
        default=None,
        help="output PNG for rel_l2 (default: INPUT stem + _rel_l2.png if column present)",
    )
    parser.add_argument(
        "--title",
        default="NN vs FEM: MSE",
        help="figure title for mse plot",
    )
    parser.add_argument(
        "--title-rel",
        default="NN vs FEM: rel_l2 = ||u_nn-u_fem||_2 / ||u_fem||_2",
        help="figure title for rel_l2 plot",
    )
    parser.add_argument(
        "--plot-frac",
        type=float,
        default=1.0,
        help="fraction of points to plot (e.g. 0.3 for large train sets)",
    )
    parser.add_argument(
        "--plot-seed",
        type=int,
        default=0,
        help="RNG seed used when --plot-frac < 1",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help=(
            "also open the mse figure in Plotly's configured viewer/browser "
            "(requires plotly)"
        ),
    )
    return parser.parse_args()


def non_comment_lines(path: Path) -> Iterable[str]:
    """Yield the lines of ``path`` that are neither blank nor '#' comments."""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for line in handle:
            if line.strip() and not line.lstrip().startswith("#"):
                yield line


def load_samples(
    path: Path,
) -> tuple[list[int], list[float], list[float] | None]:
    """Read the metrics TSV.

    Returns ``(sample_indices, mse, rel_l2)``; ``rel_l2`` is None when the
    file has no ``rel_l2`` column.  Raises ValueError on missing columns,
    non-numeric or non-finite values, negative sample indices or an empty
    table.
    """
    if not path.is_file():
        raise FileNotFoundError(f"input TSV does not exist: {path}")

    reader = csv.DictReader(non_comment_lines(path), delimiter="\t")
    if reader.fieldnames is None:
        raise ValueError(f"input TSV has no header/data rows: {path}")

    required = {"sample_index", "mse"}
    missing = required.difference(reader.fieldnames)
    if missing:
        raise ValueError(
            f"input TSV is missing column(s) {sorted(missing)}; "
            f"found {reader.fieldnames}"
        )
    has_rel = "rel_l2" in reader.fieldnames

    sample_indices: list[int] = []
    errors: list[float] = []
    rels: list[float] = []
    for row_number, row in enumerate(reader, start=2):
        try:
            sample_index = int(row["sample_index"])
            mse = float(row["mse"])
            rel = float(row["rel_l2"]) if has_rel else float("nan")
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid numeric value in TSV data row {row_number}: {row}"
            ) from exc
        if sample_index < 0:
            raise ValueError(
                f"negative sample_index in TSV data row {row_number}: {row}"
            )
        if not math.isfinite(mse):
            raise ValueError(
                f"non-finite value in TSV data row {row_number}: {row}"
            )
        sample_indices.append(sample_index)
        errors.append(mse)
        if has_rel:
            if not math.isfinite(rel):
                raise ValueError(
                    f"non-finite rel_l2 in TSV data row {row_number}: {row}"
                )
            rels.append(rel)

    if not sample_indices:
        raise ValueError(f"input TSV contains no samples: {path}")
    return sample_indices, errors, (rels if has_rel else None)


def build_figure(
    sample_indices: list[int],
    errors: list[float],
    title: str,
    yaxis_title: str,
    color: str = "#636EFA",
    value_name: str = "MSE",
) -> go.Figure:
    """Build the Plotly scatter figure with a dashed mean line (needs plotly)."""
    mean_y = math.fsum(errors) / len(errors)
    figure = go.Figure(
        go.Scatter(
            x=sample_indices,
            y=errors,
            mode="markers",
            marker={
                "size": 7,
                "opacity": 0.85,
                "color": color,
            },
            hovertemplate=(
                "generalization sample=%{x}<br>"
                f"{value_name}=%{{y:.6g}}<extra></extra>"
            ),
            showlegend=False,
        )
    )
    figure.add_hline(
        y=mean_y,
        line_color="#EF553B",
        line_dash="dash",
        line_width=2,
        annotation_text=f"Mean {value_name} = {mean_y:.6g}",
        annotation_position="top right",
    )
    figure.update_layout(
        title={"text": title, "x": 0.0, "xanchor": "left"},
        template="plotly",
        xaxis_title="Generalization sample index",
        yaxis_title=yaxis_title,
        hovermode="closest",
        showlegend=False,
    )
    figure.update_yaxes(rangemode="tozero")
    return figure


def write_png_with_matplotlib(
    sample_indices: list[int],
    errors: list[float],
    title: str,
    output: Path,
    yaxis_title: str,
    color: str = "#636EFA",
    value_name: str = "MSE",
) -> None:
    """Chrome-free Matplotlib rendering of the same scatter plot.

    Used when plotly could not be imported and as the fallback when
    Plotly/Kaleido cannot export a PNG.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(14, 9))
    mean_y = math.fsum(errors) / len(errors)
    figure.patch.set_facecolor("white")
    axis.set_facecolor("#E5ECF6")
    axis.scatter(
        sample_indices,
        errors,
        color=color,
        s=30,
        alpha=0.85,
        edgecolors="none",
    )
    axis.axhline(
        mean_y,
        color="#EF553B",
        linestyle="--",
        linewidth=2,
    )
    axis.text(
        0.99,
        mean_y,
        f"Mean {value_name} = {mean_y:.6g}",
        color="#EF553B",
        ha="right",
        va="bottom",
        transform=axis.get_yaxis_transform(),
    )
    axis.set_title(title, loc="left")
    axis.set_xlabel("Generalization sample index")
    axis.set_ylabel(yaxis_title)
    axis.set_ylim(bottom=0)
    axis.grid(True, color="white", linewidth=1.2)
    axis.set_axisbelow(True)
    for spine in axis.spines.values():
        spine.set_visible(False)
    figure.tight_layout()
    figure.savefig(output, format="png", dpi=200, bbox_inches="tight")
    plt.close(figure)


def write_one_png(
    sample_indices: list[int],
    values: list[float],
    output: Path,
    title: str,
    yaxis_title: str,
    color: str,
    value_name: str,
) -> go.Figure | None:
    """Write one scatter plot to ``output`` (PNG).

    Returns the Plotly figure, or None when plotly could not be imported and
    the PNG was drawn with Matplotlib.
    """
    if go is None:
        # plotly could not be imported: draw the Matplotlib figure directly.
        output.parent.mkdir(parents=True, exist_ok=True)
        write_png_with_matplotlib(
            sample_indices,
            values,
            title,
            output,
            yaxis_title,
            color,
            value_name,
        )
        print(
            f"Wrote {len(sample_indices)} samples to {output.resolve()} "
            "using Matplotlib (plotly could not be imported)"
        )
        return None
    figure = build_figure(
        sample_indices, values, title, yaxis_title, color, value_name
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        figure.write_image(
            str(output), format="png", width=1400, height=900, scale=2
        )
        backend = "Plotly/Kaleido"
    except Exception as plotly_error:
        print(
            "Plotly/Kaleido PNG export failed "
            f"({type(plotly_error).__name__}: {plotly_error})."
        )
        print("Falling back to Matplotlib; Chrome is not required.")
        write_png_with_matplotlib(
            sample_indices,
            values,
            title,
            output,
            yaxis_title,
            color,
            value_name,
        )
        backend = "Matplotlib fallback"
    print(
        f"Wrote {len(sample_indices)} samples to {output.resolve()} "
        f"using {backend}"
    )
    return figure


def subsample(
    sample_indices: list[int],
    values: list[float],
    frac: float,
    seed: int,
) -> tuple[list[int], list[float]]:
    """Return a seeded random subset of the points, in their original order.

    ``frac >= 1`` returns the inputs unchanged; otherwise
    ``max(1, round(n * frac))`` points are kept.
    """
    if frac >= 1.0:
        return sample_indices, values
    if frac <= 0.0:
        raise ValueError("--plot-frac must be in (0, 1]")
    import random

    n = len(values)
    m = max(1, int(round(n * frac)))
    rng = random.Random(seed)
    sel = sorted(rng.sample(range(n), m))
    return [sample_indices[i] for i in sel], [values[i] for i in sel]


def main() -> None:
    """Write the MSE plot and, if the column exists, the rel_l2 plot."""
    args = parse_args()
    if args.show and go is None:
        raise SystemExit(
            "--show needs plotly, which could not be imported "
            "(pip install plotly). "
            "Run without --show to write the PNG file(s) with Matplotlib."
        )
    output = args.output if args.output is not None else args.input.with_suffix(".png")
    if output.suffix.lower() != ".png":
        corrected_output = output.with_suffix(".png")
        print(
            f"Output path {output} is not PNG; writing {corrected_output} instead."
        )
        output = corrected_output

    sample_indices, errors, rels = load_samples(args.input)
    idx_m, err_m = subsample(
        sample_indices, errors, args.plot_frac, args.plot_seed
    )
    figure = write_one_png(
        idx_m,
        err_m,
        output,
        args.title,
        "MSE",
        "#636EFA",
        "MSE",
    )

    if rels is not None:
        out_rel = args.out_rel
        if out_rel is None:
            out_rel = args.input.with_name(args.input.stem + "_rel_l2.png")
        if out_rel.suffix.lower() != ".png":
            out_rel = out_rel.with_suffix(".png")
        idx_r, rel_m = subsample(
            sample_indices, rels, args.plot_frac, args.plot_seed
        )
        write_one_png(
            idx_r,
            rel_m,
            out_rel,
            args.title_rel,
            "rel_l2",
            "#AB63FA",
            "rel_l2",
        )

    if args.show:
        figure.show()


if __name__ == "__main__":
    main()
