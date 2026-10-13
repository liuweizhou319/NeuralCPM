#!/usr/bin/env python3
"""Report the voxel spacing of the sparse SDF grid for a mesh and SDF resolution.

The narrow-band SDF built by ``SparseGridFDM_SDF.from_mesh`` lives on a sparse
lattice whose spacing follows from ``--sdf-res`` and ``--sdf-padding``:

    voxel_size = (1 + 2 * sdf_padding) * L_max / (sdf_res - 1)

where ``L_max`` is the longest bounding-box edge of the *normalised* mesh
(bounding-box centre moved to the origin, then divided by the maximum vertex
norm, exactly as ``MeshGeometry`` does).  ``--fd-halo`` adds that many voxel
layers on every side of the voxels that intersect the surface (at least 2), so
the band reaches roughly ``fd_halo * voxel_size`` away from the surface.

This script only evaluates those formulas; it does not build the grid and
needs neither a GPU nor fvdb.  It is used by ``run_sdf_sweep.sh`` to
annotate each run with its spacing, and is handy for choosing a resolution:

    python sdf_spacing.py data/fandisk.obj --sdf-res 200 --fd-halo 6
    python sdf_spacing.py data/fandisk.obj --pairs 20,3 60,3 100,4 200,6 300,8
    python sdf_spacing.py data/fandisk_5w.obj --pairs 200,6 --tsv

Columns
-------
sdf_res, fd_halo   the grid settings (fd_halo after the minimum of 2 is applied)
voxel_size         lattice spacing in normalised mesh units
band_halfwidth     fd_halo * voxel_size, the approximate reach of the band
voxel_size_input   the same spacing in the units of the input OBJ file
voxel_over_edge    voxel_size divided by the mean mesh edge length
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from Mesh_geometry import load_obj


def normalised_vertices(vertices: np.ndarray) -> tuple[np.ndarray, float]:
    """Normalise like ``MeshGeometry.__init__``; return (vertices, scale)."""
    center = (vertices.max(axis=0) + vertices.min(axis=0)) / 2
    vertices = vertices - center
    scale = np.sqrt((vertices ** 2).sum(axis=1)).max()
    return vertices / scale, float(scale)


def voxel_size(vertices: np.ndarray, resolution: int, padding: float) -> float:
    """Lattice spacing, mirroring ``SparseGridFDM_SDF.from_mesh``."""
    V = vertices.astype(np.float32)
    mn, mx = V.min(0), V.max(0)
    pad = padding * (mx - mn).max()
    bb_min = mn - pad
    bb_max = mx + pad
    return float((bb_max - bb_min).max() / (resolution - 1))


def mean_edge_length(vertices: np.ndarray, faces: np.ndarray) -> float:
    """Mean length of the unique mesh edges."""
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    diff = vertices[edges[:, 0]] - vertices[edges[:, 1]]
    return float(np.sqrt((diff.astype(np.float64) ** 2).sum(axis=1)).mean())


def parse_pair(text: str) -> tuple[int, int]:
    """Parse 'SDF_RES,FD_HALO' (the halo defaults to 2 when omitted)."""
    parts = [p for p in text.replace(":", ",").split(",") if p.strip()]
    if len(parts) == 1:
        return int(parts[0]), 2
    if len(parts) == 2:
        return int(parts[0]), int(parts[1])
    raise argparse.ArgumentTypeError(f"expected SDF_RES,FD_HALO but got {text!r}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Voxel spacing of the sparse SDF grid for given SDF settings.",
    )
    p.add_argument("mesh", help="mesh .obj path")
    p.add_argument("--sdf-res", type=int, default=None,
                   help="SDF resolution (lattice nodes along the longest padded "
                        "bounding-box axis)")
    p.add_argument("--fd-halo", type=int, default=2,
                   help="voxel layers added around the surface voxels (default: 2)")
    p.add_argument("--pairs", nargs="+", type=parse_pair, default=None,
                   metavar="SDF_RES,FD_HALO",
                   help="several settings at once, e.g. --pairs 100,4 200,6")
    p.add_argument("--sdf-padding", type=float, default=0.15,
                   help="bounding-box padding as a fraction of L_max (default: 0.15)")
    p.add_argument("--tsv", action="store_true",
                   help="print a tab-separated table without the header comments")
    args = p.parse_args()
    if (args.sdf_res is None) == (args.pairs is None):
        p.error("give either --sdf-res (with optional --fd-halo) or --pairs")
    return args


def main():
    args = parse_args()
    pairs = args.pairs if args.pairs is not None else [(args.sdf_res, args.fd_halo)]
    for res, _ in pairs:
        if res < 2:
            sys.exit(f"SDF resolution must be >= 2, got {res}")

    raw_vertices, faces = load_obj(args.mesh)
    if raw_vertices.size == 0 or faces.size == 0:
        sys.exit(f"no vertices or faces loaded from {args.mesh!r}")
    vertices, scale = normalised_vertices(raw_vertices)
    extent = vertices.max(axis=0) - vertices.min(axis=0)
    l_max = float(extent.max())
    edge = mean_edge_length(vertices, faces)

    columns = ("sdf_res", "fd_halo", "voxel_size", "band_halfwidth",
               "voxel_size_input", "voxel_over_edge")
    rows = []
    for res, halo in pairs:
        halo = max(2, int(halo))
        vs = voxel_size(vertices, res, args.sdf_padding)
        rows.append((str(res), str(halo), f"{vs:.6g}", f"{halo * vs:.6g}",
                     f"{vs * scale:.6g}", f"{vs / edge:.4g}"))

    if args.tsv:
        print("\t".join(columns))
        for row in rows:
            print("\t".join(row))
        return

    print(f"# mesh: {args.mesh}  vertices={len(vertices)}  faces={len(faces)}")
    print(f"# normalisation scale (input units per normalised unit): {scale:.6g}")
    print(f"# normalised bounding box: L_max={l_max:.6g}  "
          f"mean edge length={edge:.6g}")
    print(f"# sdf_padding={args.sdf_padding:g}  "
          f"voxel_size = (1 + 2*padding) * L_max / (sdf_res - 1)")
    widths = [max(len(c), *(len(r[i]) for r in rows)) for i, c in enumerate(columns)]
    print("  ".join(c.ljust(w) for c, w in zip(columns, widths)))
    for row in rows:
        print("  ".join(v.ljust(w) for v, w in zip(row, widths)))


if __name__ == "__main__":
    main()
