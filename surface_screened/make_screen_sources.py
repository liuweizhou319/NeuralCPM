#!/usr/bin/env python3
r"""
make_screen_sources.py - write a source-spec (constraint) JSON from the
command line.

Role in the pipeline
--------------------
The source spec is the JSON list of Dirichlet sources that train_screen.py,
fem_screen.py and the evaluation tools read through --source-spec (the
schema is documented in the module docstring of train_screen.py).  This
script composes such a file from command-line flags, so that it does not have
to be edited by hand.  It needs only the Python standard library and does not
read the mesh: vertex ids are zero-based vertex indices and are checked
later, by the tools that load the mesh.

Flags
-----
Every --vertex-ids, --disk-seed, --curve-npy and --curve-points adds one
source.  The companion flags (--vertex-value, --disk-radius, ...) are
repeatable as well and are matched to their source by order of appearance;
missing ones take the defaults value 1.0, weight 1.0, patch_radius 0.0,
radius 0.1, generalize false, closed false.  The sources are written in this
order: --merge entries, vertex sources, disk sources, --curve-npy sources,
--curve-points sources.

--vertex-*, --disk-*, --curve-npy, --curve-value, --curve-weight,
--curve-closed, --r-min, --r-max, --value-generalize-source, --value-min and
--value-max have the same names and meaning as the inline shortcuts of
train_screen.py.  The following flags exist only here:

    -o, --out             output path (required; its directory must exist)
    --merge JSON          start from an existing source spec and append to it
    --indent N            JSON indentation (default: 2)
    --curve-points SPEC   inline polyline "x,y,z;x,y,z;...", with the
                          companions --curve-points-value,
                          --curve-points-weight and --curve-points-closed

Conversely, --export-r, --export-value and --train-codes-out exist only in
train_screen.py.

--curve-npy writes the path as it is given into "points_npy"; the file is not
opened here.  train_screen.py and fem_screen.py load it relative to their
working directory, evaluate_dirichlet_generalization.py resolves a relative
path against the directory of the JSON file, and export_split_samples.py /
export_worst_rel_l2.py do not resolve "points_npy" at all.  For a spec that
goes through the generalization pipeline, prefer --curve-points (inline
points).

--value-generalize-source INDEX writes "value_generalize": true and
"value_range": [--value-min, --value-max] into source INDEX of the final
list; in training every source marked this way gets its own independent code
component.  --disk-generalize true writes "generalize": true and
"r_range": [--r-min, --r-max] into a disk source (legacy disk-radius
generalization).  The two kinds cannot be mixed in one file.  The source
specs shipped in data/ carry no such markers: the generalized sources can
equally be selected with --value-generalize-source on the train_screen.py
command line.

Usage (from the repository root)
--------------------------------
Four vertex sources (the same sources as data/Bunny_2.json):

    python make_screen_sources.py -o data/my_sources.json \
        --vertex-ids 1101 --vertex-value 1.0 --vertex-patch-radius 0.02 \
        --vertex-ids 1589 --vertex-value -1.0 --vertex-patch-radius 0.02 \
        --vertex-ids 1842 --vertex-value 0.0 --vertex-patch-radius 0.02 \
        --vertex-ids 2234 --vertex-value -2.0 --vertex-patch-radius 0.02

Append an inline polyline to an existing spec (the coordinates are only an
illustration; real curve points must lie on or near the surface, in the
normalized mesh frame):

    python make_screen_sources.py -o data/my_sources_curve.json \
        --merge data/my_sources.json \
        --curve-points "0.1,0.0,0.3;0.2,0.0,0.3;0.3,0.1,0.3" \
        --curve-points-value 0.5

Train with the result (fem_screen.py takes the same file through
--source-spec):

    python train_screen.py data/Bunny.obj --source-spec data/my_sources.json \
        --out-dir results/single/my_run
"""

from __future__ import annotations

import argparse
import json
import sys


def _parse_int_list(s: str) -> list[int]:
    """Parse a comma-separated list of integers ("100,250" -> [100, 250])."""
    return [int(x) for x in s.split(",") if x.strip()]


def _str_to_bool(x: str | None, default: bool = False) -> bool:
    """Interpret "1" / "true" / "t" / "yes" / "y" (any case) as True."""
    if x is None:
        return default
    return str(x).lower() in ("1", "true", "t", "yes", "y")


def _parse_curve_points_inline(spec: str) -> list[list[float]]:
    """Parse an inline polyline: "x,y,z;x,y,z;..." or "x y z; x y z"."""
    out: list[list[float]] = []
    for block in spec.split(";"):
        block = block.strip()
        if not block:
            continue
        nums = [float(t) for t in block.replace(",", " ").split()]
        if len(nums) != 3:
            raise ValueError(
                f"each curve point needs 3 numbers, got {len(nums)}: {block!r}"
            )
        out.append(nums)
    if len(out) < 2:
        raise ValueError("a curve needs at least 2 points")
    return out


def build_source_list(args: argparse.Namespace) -> list[dict]:
    """Compose the list of source dicts that is written to the JSON file.

    Order: --merge entries, vertex sources, disk sources, --curve-npy
    sources, --curve-points sources.  The sources selected with
    --value-generalize-source are then marked as value-generalized.
    """
    sources: list[dict] = []

    # existing spec to append to
    if args.merge:
        with open(args.merge, "r", encoding="utf-8") as f:
            merged = json.load(f)
        if not isinstance(merged, list):
            sys.exit("--merge file must contain a JSON array")
        for s in merged:
            if not isinstance(s, dict):
                sys.exit("every item of the --merge file must be a JSON object")
            sources.append(dict(s))

    # vertex sources
    n_v = len(args.vertex_ids)
    for k in range(n_v):
        ids = _parse_int_list(args.vertex_ids[k])
        val = args.vertex_value[k] if k < len(args.vertex_value) else 1.0
        wt = args.vertex_weight[k] if k < len(args.vertex_weight) else 1.0
        pr = (
            args.vertex_patch_radius[k]
            if k < len(args.vertex_patch_radius)
            else 0.0
        )
        sources.append(
            {
                "type": "vertex",
                "ids": ids,
                "value": float(val),
                "weight": float(wt),
                "patch_radius": float(pr),
            }
        )

    # disk_curve sources
    n_d = len(args.disk_seed)
    for k in range(n_d):
        seed = args.disk_seed[k]
        r = args.disk_radius[k] if k < len(args.disk_radius) else 0.1
        val = args.disk_value[k] if k < len(args.disk_value) else 1.0
        wt = args.disk_weight[k] if k < len(args.disk_weight) else 1.0
        gen = _str_to_bool(
            args.disk_generalize[k] if k < len(args.disk_generalize) else None
        )
        d: dict = {
            "type": "disk_curve",
            "seed": int(seed),
            "radius": float(r),
            "value": float(val),
            "weight": float(wt),
            "generalize": gen,
        }
        if gen:
            if args.r_min is None or args.r_max is None:
                sys.exit("--disk-generalize true requires --r-min and --r-max")
            d["r_range"] = [float(args.r_min), float(args.r_max)]
        sources.append(d)

    # curve sources referencing a .npy file (the path is stored, not opened)
    n_cn = len(args.curve_npy)
    for k in range(n_cn):
        npy = args.curve_npy[k]
        val = args.curve_value[k] if k < len(args.curve_value) else 1.0
        wt = args.curve_weight[k] if k < len(args.curve_weight) else 1.0
        clo = _str_to_bool(
            args.curve_closed[k] if k < len(args.curve_closed) else None
        )
        sources.append(
            {
                "type": "curve",
                "points_npy": npy,
                "value": float(val),
                "weight": float(wt),
                "closed": clo,
            }
        )

    # curve sources with inline points
    n_cp = len(args.curve_points)
    for k in range(n_cp):
        pts = _parse_curve_points_inline(args.curve_points[k])
        val = args.curve_points_value[k] if k < len(args.curve_points_value) else 1.0
        wt = (
            args.curve_points_weight[k]
            if k < len(args.curve_points_weight)
            else 1.0
        )
        clo = _str_to_bool(
            args.curve_points_closed[k]
            if k < len(args.curve_points_closed)
            else None
        )
        sources.append(
            {
                "type": "curve",
                "points": pts,
                "value": float(val),
                "weight": float(wt),
                "closed": clo,
            }
        )

    if not sources:
        sys.exit("No sources defined: use --vertex-ids / --disk-seed / "
                 "--curve-npy / --curve-points, or --merge")

    # Dirichlet-value generalization markers (indices into the list above)
    selected = list(args.value_generalize_source)
    duplicates = sorted({i for i in selected if selected.count(i) > 1})
    if duplicates:
        sys.exit("--value-generalize-source contains duplicate index(es): "
                 + ", ".join(map(str, duplicates)))
    invalid = [i for i in selected if i < 0 or i >= len(sources)]
    if invalid:
        sys.exit("--value-generalize-source index out of range: "
                 + ", ".join(map(str, invalid))
                 + f"; valid range is 0..{len(sources) - 1}")
    # training requires min < max, so an empty or reversed range is rejected
    if selected and args.value_min >= args.value_max:
        sys.exit(f"invalid value range [{args.value_min}, {args.value_max}]: "
                 "--value-min must be < --value-max")
    for index in selected:
        sources[index]["value_generalize"] = True
        sources[index]["value_range"] = [
            float(args.value_min), float(args.value_max)]

    # radius mode (one shared scalar code) and value mode (one code component
    # per marked source) condition the network differently and are exclusive
    radius_on = any(s.get("type") == "disk_curve"
                    and bool(s.get("generalize", False)) for s in sources)
    value_on = any(bool(s.get("value_generalize", False)) for s in sources)
    if radius_on and value_on:
        sys.exit("disk-radius and Dirichlet-value generalization cannot be "
                 "mixed: they use distinct conditioning semantics")
    return sources


def main() -> None:
    """Parse the command line and write the source-spec JSON."""
    p = argparse.ArgumentParser(
        description="Build a screened-diffusion source-spec JSON from the "
                    "command line",
    )
    p.add_argument(
        "-o", "--out", required=True,
        help="output .json path (its directory must exist)",
    )
    p.add_argument(
        "--merge",
        metavar="JSON",
        help="load this existing JSON array first, then append the sources "
             "defined on the command line",
    )
    p.add_argument(
        "--indent", type=int, default=2,
        help="JSON indentation; 0 = most compact (default: 2)",
    )

    p.add_argument(
        "--vertex-ids",
        action="append",
        default=[],
        help="comma-separated vertex ids of one vertex source; repeatable",
    )
    p.add_argument(
        "--vertex-value",
        action="append",
        default=[],
        type=float,
        help="value for the matching --vertex-ids (default: 1.0)",
    )
    p.add_argument(
        "--vertex-weight",
        action="append",
        default=[],
        type=float,
        help="weight for the matching --vertex-ids (default: 1.0)",
    )
    p.add_argument(
        "--vertex-patch-radius",
        action="append",
        default=[],
        type=float,
        help="Dijkstra patch radius for the matching --vertex-ids; "
             "0 = the listed vertices only (default: 0.0)",
    )

    p.add_argument(
        "--disk-seed",
        action="append",
        default=[],
        type=int,
        help="seed vertex id of one disk_curve source; repeatable",
    )
    p.add_argument(
        "--disk-radius",
        action="append",
        default=[],
        type=float,
        help="radius for the matching --disk-seed (default: 0.1)",
    )
    p.add_argument(
        "--disk-value",
        action="append",
        default=[],
        type=float,
        help="value for the matching --disk-seed (default: 1.0)",
    )
    p.add_argument(
        "--disk-weight",
        action="append",
        default=[],
        type=float,
        help="weight for the matching --disk-seed (default: 1.0)",
    )
    p.add_argument(
        "--disk-generalize",
        action="append",
        default=[],
        help='"true"/"false" for the matching --disk-seed (default: false); '
             "true requires --r-min and --r-max",
    )

    p.add_argument(
        "--curve-npy",
        action="append",
        default=[],
        help="path of a (K,3) .npy polyline for one curve source; stored "
             "verbatim in the JSON as points_npy (not opened here) and "
             "loaded later by train_screen.py / fem_screen.py relative to "
             "their working directory; repeatable",
    )
    p.add_argument(
        "--curve-value",
        action="append",
        default=[],
        type=float,
        help="value for the matching --curve-npy (default: 1.0)",
    )
    p.add_argument(
        "--curve-weight",
        action="append",
        default=[],
        type=float,
        help="weight for the matching --curve-npy (default: 1.0)",
    )
    p.add_argument(
        "--curve-closed",
        action="append",
        default=[],
        help='"true"/"false" for the matching --curve-npy (default: false)',
    )

    p.add_argument(
        "--curve-points",
        action="append",
        default=[],
        metavar="SPEC",
        help='inline polyline "x,y,z;x,y,z;..." for one curve source; '
             "repeat to define several curves",
    )
    p.add_argument(
        "--curve-points-value",
        action="append",
        default=[],
        type=float,
        help="value for the matching --curve-points (default: 1.0)",
    )
    p.add_argument(
        "--curve-points-weight",
        action="append",
        default=[],
        type=float,
        help="weight for the matching --curve-points (default: 1.0)",
    )
    p.add_argument(
        "--curve-points-closed",
        action="append",
        default=[],
        help='"true"/"false" for the matching --curve-points '
             "(default: false)",
    )

    p.add_argument(
        "--r-min",
        type=float,
        default=None,
        help="lower end of r_range; required when any --disk-generalize "
             "is true",
    )
    p.add_argument(
        "--r-max",
        type=float,
        default=None,
        help="upper end of r_range; required when any --disk-generalize "
             "is true",
    )

    p.add_argument(
        "--value-generalize-source",
        action="append",
        default=[],
        type=int,
        metavar="INDEX",
        help="zero-based index in the final source list; repeat to mark "
             "several sources, each of which gets its own independent code "
             "component",
    )
    p.add_argument(
        "--value-min",
        type=float,
        default=-2.0,
        help="minimum generalized Dirichlet value (default: -2)",
    )
    p.add_argument(
        "--value-max",
        type=float,
        default=2.0,
        help="maximum generalized Dirichlet value (default: 2)",
    )

    args = p.parse_args()
    sources = build_source_list(args)

    indent = None if args.indent <= 0 else args.indent
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(sources, f, indent=indent)
        f.write("\n")
    print(f"Wrote {len(sources)} source(s) -> {args.out}")


if __name__ == "__main__":
    main()
