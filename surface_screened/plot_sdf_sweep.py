#!/usr/bin/env python3
r"""Line plot of an NN-vs-FEM error metric versus the SDF grid resolution.

Figure script of the SDF-resolution sweep: train_screen.py is run at several
--sdf-res (and matching --fd-halo) values, every run is compared with the FEM
reference by compare_nn_fem_mse.py, and this script plots one metric of those
reports against the resolution.  Metrics, at the mesh vertices:
    rmse   = sqrt(mean((u_nn - u_fem)^2))
    rel_l2 = sqrt(sum((u_nn - u_fem)^2) / sum(u_fem^2))

Input: one or more text files made of blocks of 'key: value' metric lines as
written by compare_nn_fem_mse.py, each block optionally preceded by a label
line 'SDF_RES' or 'SDF_RES,FD_HALO':

    200,6
    mse: <float>
    rmse: <float>
    rel_l2: <float>
    mean_abs_error: <float>
    max_abs_error: <float>

    250,7
    mse: <float>
    ...

This is the summary.txt written by run_sdf_sweep.sh
(results/sdf_sweep/<name>/summary.txt).  Parsing rules, applied to each line
after stripping surrounding whitespace:

  * blank lines are ignored;
  * a line starting with '<metric>:' contributes one y value;
  * every other line containing ':' is ignored (the other metrics and, if
    whole reports are pasted, their nn_npz: / fem_npz: / field: /
    n_vertices: lines);
  * a line without ':' is a label: the text before the first comma is read
    as a number and becomes the x value of the next '<metric>:' value (the
    part after the comma, FD_HALO, is not used); non-numeric labels are
    ignored;
  * a value without a pending label gets its 1-based position as x.

The x axis is titled 'SDF_RES' when at least one label was used and 'index'
otherwise.  --metric auto (the default) plots rel_l2 if the file contains at
least one line starting with 'rel_l2:', otherwise rmse.

Output: one PNG per input, <stem>_<metric>.png next to the txt (for example
results/sdf_sweep/fandisk/summary_rel_l2.png).  -o/--output overrides that
path and is only allowed with a single input file.

Rendering: plotly is optional.  When it can be imported the PNG is exported
with Plotly/Kaleido, with a fallback to Matplotlib if that export fails.  When
importing plotly raises ImportError (normally because it is not installed) the
Matplotlib figure is drawn directly and the 'Wrote ...' line says so;
matplotlib alone is therefore enough.

Usage (from the repository root):

    python plot_sdf_sweep.py results/sdf_sweep/fandisk/summary.txt

    python plot_sdf_sweep.py results/sdf_sweep/fandisk/summary.txt \
        --metric rmse -o results/sdf_sweep/fandisk/rmse.png

    # several sweeps at once (the shell expands the pattern), one PNG each
    python plot_sdf_sweep.py results/sdf_sweep/*/summary.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    import plotly.graph_objects as go
except ImportError:
    # plotly is optional: if it cannot be imported (normally: not installed)
    # the figure is drawn with Matplotlib.
    go = None


def parse_args():
    """Parse the command line; -o/--output is rejected with several inputs."""
    p = argparse.ArgumentParser(
        description=(
            "Plot rmse or rel_l2 against the SDF resolution from sweep "
            "summary files: blocks of compare_nn_fem_mse.py metric lines, "
            "each optionally preceded by an 'SDF_RES' or 'SDF_RES,FD_HALO' "
            "label line."
        )
    )
    p.add_argument(
        "txt",
        type=Path,
        nargs="+",
        help="input txt file(s), e.g. results/sdf_sweep/<name>/summary.txt",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=(
            "output png, only allowed with a single input "
            "(default: TXT stem + _<metric>.png next to each TXT)"
        ),
    )
    p.add_argument(
        "--metric",
        choices=("auto", "rmse", "rel_l2"),
        default="auto",
        help=(
            "which field to plot (auto: rel_l2 if the file has a line "
            "starting with 'rel_l2:', else rmse)"
        ),
    )
    args = p.parse_args()
    if args.output is not None and len(args.txt) > 1:
        p.error("-o/--output can only be used with a single input file")
    return args


def resolve_metric(path: Path, metric: str) -> str:
    """Return the metric to plot.

    'auto' means rel_l2 if ``path`` contains at least one line starting with
    'rel_l2:' (same stripping as load_metric), otherwise rmse.
    """
    if metric != "auto":
        return metric
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip().startswith("rel_l2:"):
                return "rel_l2"
    return "rmse"


def parse_label(s: str) -> float | None:
    """Parse leading SDF_RES label: '50' or '50,3' -> 50."""
    head = s.split(",", 1)[0].strip()
    try:
        return float(head)
    except ValueError:
        return None


def load_metric(path: Path, key: str) -> tuple[list[float], list[float], bool]:
    """Return (x, y, used_labels). x uses leading SDF_RES labels when present."""
    values: list[float] = []
    xs: list[float] = []
    pending_x: float | None = None
    used_labels = False
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith(f"{key}:"):
                values.append(float(s.split(":", 1)[1].strip()))
                if pending_x is not None:
                    xs.append(pending_x)
                    used_labels = True
                else:
                    xs.append(float(len(values)))
                pending_x = None
                continue
            if ":" in s:
                # other metric lines (mse:/rmse:/...); keep pending label
                continue
            label = parse_label(s)
            if label is not None:
                pending_x = label
    if not values:
        raise ValueError(f"no {key}: lines in {path}")
    return xs, values, used_labels


def write_png_with_matplotlib(
    xs: list[float],
    ys: list[float],
    out: Path,
    title: str,
    yaxis: str,
    xlabel: str,
):
    """Matplotlib rendering of the line plot.

    Used when plotly could not be imported and as the fallback when the
    Plotly PNG export fails.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig2, ax = plt.subplots(figsize=(10, 6))
    ax.plot(xs, ys, "-o", markersize=6)
    ax.set_title(title, loc="left")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(yaxis)
    ax.set_xticks(xs)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig2)


def write_png(
    xs: list[float],
    ys: list[float],
    out: Path,
    title: str,
    yaxis: str,
    xlabel: str,
):
    """Write the line plot to ``out`` (PNG), with Plotly or Matplotlib."""
    if go is None:
        # plotly could not be imported: draw the Matplotlib figure directly.
        out.parent.mkdir(parents=True, exist_ok=True)
        write_png_with_matplotlib(xs, ys, out, title, yaxis, xlabel)
        print(
            f"Wrote {out.resolve()} "
            f"(matplotlib; plotly could not be imported), n={len(ys)}"
        )
        return
    fig = go.Figure(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines+markers",
            marker=dict(size=8),
            line=dict(width=2),
            showlegend=False,
        )
    )
    fig.update_layout(
        title={"text": title, "x": 0.0, "xanchor": "left"},
        xaxis_title=xlabel,
        yaxis_title=yaxis,
        template="plotly",
    )
    fig.update_yaxes(rangemode="tozero")

    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.write_image(str(out), format="png", width=1000, height=600, scale=2)
        backend = "plotly"
    except Exception as e:
        print(f"plotly failed ({e}); fallback matplotlib")
        write_png_with_matplotlib(xs, ys, out, title, yaxis, xlabel)
        backend = "matplotlib"
    print(f"Wrote {out.resolve()} ({backend}), n={len(ys)}")


def plot_one(txt: Path, output: Path | None, metric: str):
    """Plot one input file; ``output=None`` means <stem>_<metric>.png next to it."""
    key = resolve_metric(txt, metric)
    xs, ys, used_labels = load_metric(txt, key)
    out = output if output is not None else txt.with_name(f"{txt.stem}_{key}.png")
    xlabel = "SDF_RES" if used_labels else "index"
    write_png(xs, ys, out, title=f"{key}: {txt.name}", yaxis=key, xlabel=xlabel)
    print(f"{key}:", ", ".join(f"{v:.6g}" for v in ys))


def main():
    """Plot every input file given on the command line."""
    args = parse_args()
    for txt in args.txt:
        plot_one(txt, args.output, args.metric)


if __name__ == "__main__":
    main()
