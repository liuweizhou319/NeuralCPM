#!/usr/bin/env python3
"""
Write a constraint (anchor) .npz for train_vecfield.py.

Output format (the one --constraints expects):
  face_ids : (N,) int64     triangle indices (0-based, as in Mesh_geometry.load_obj)
  vectors  : (N,3) float32  target vectors; training projects them onto the
                            tangent plane of each face

Examples
--------
  # faces 0, 5, 100: a random tangent vector on each face (same magnitude as
  # the built-in random constraints)
  python make_vecfield_constraints.py data/37384.obj -o my_c.npz --faces 0,5,100 --random

  # the same faces, all with the world-space vector (1,0,0) (projected onto
  # each tangent plane)
  python make_vecfield_constraints.py data/37384.obj -o my_c.npz --faces 0,5,100 --vector 1 0 0

  # one vector per face: "x,y,z;x,y,z;..." separated by semicolons
  python make_vecfield_constraints.py data/37384.obj -o my_c.npz --faces 0,5,100 \\
      --vectors-list "1,0,0;0,1,0;0,0,1"

  # without the mesh normalisation (must match --no-normalize of train_vecfield.py)
  python make_vecfield_constraints.py data/37384.obj -o my_c.npz --faces 10 --random --no-normalize
"""

from __future__ import annotations

import argparse
import re
import sys

import numpy as np

from Mesh_geometry import MeshGeometry
from SurfaceVectorFieldDesign import project_to_tangent


def _parse_face_ids(spec: str) -> np.ndarray:
    parts = re.split(r"[\s,]+", spec.strip())
    ids = [int(p) for p in parts if p]
    if not ids:
        raise ValueError("no face index found")
    return np.array(ids, dtype=np.int64)


def _random_tangents(mesh: MeshGeometry, face_ids: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    normals = mesh.face_normals_np[face_ids]
    raw = rng.randn(len(face_ids), 3).astype(np.float32)
    dot = (raw * normals).sum(axis=1, keepdims=True)
    tangent = raw - dot * normals
    norms = np.sqrt((tangent**2).sum(axis=1, keepdims=True) + 1e-30)
    tangent = tangent / norms
    tangent *= mesh.avg_edge * 2.0
    return tangent.astype(np.float32)


def _parse_vectors_list(s: str, n: int) -> np.ndarray:
    blocks = [b.strip() for b in s.split(";") if b.strip()]
    if len(blocks) != n:
        raise ValueError(
            f"--vectors-list needs {n} vectors (separated by semicolons), got {len(blocks)}"
        )
    out = np.zeros((n, 3), dtype=np.float32)
    for i, b in enumerate(blocks):
        nums = [float(x) for x in b.replace(",", " ").split()]
        if len(nums) != 3:
            raise ValueError(f"vector {i} must have 3 numbers, got: {b!r}")
        out[i] = nums
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description="Write a vector-field constraint npz (face_ids + vectors)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("mesh", help="the .obj used for training")
    p.add_argument("-o", "--out", required=True, help="output .npz path")
    p.add_argument(
        "--faces",
        required=True,
        help='face indices separated by commas or spaces, e.g. "0,5,10" or "0 5 10" (0-based)',
    )
    p.add_argument(
        "--no-normalize",
        action="store_true",
        help="do not normalise the mesh (must match train_vecfield.py)",
    )
    p.add_argument("--seed", type=int, default=42, help="random seed for --random")

    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--random",
        action="store_true",
        help="a random unit tangent on each face (times 2*avg_edge, as the built-in random constraints)",
    )
    g.add_argument(
        "--vector",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        help="one 3D vector for all faces (projected onto each tangent plane)",
    )
    g.add_argument(
        "--vectors-list",
        metavar="STR",
        help='one vector per face, separated by semicolons, e.g. "1,0,0;0,1,0;0,0,1"',
    )

    args = p.parse_args()

    mesh = MeshGeometry(args.mesh, normalize=not args.no_normalize, device="cpu")
    n_faces = len(mesh.faces_np)
    face_ids = _parse_face_ids(args.faces)
    if (face_ids < 0).any() or (face_ids >= n_faces).any():
        bad = face_ids[(face_ids < 0) | (face_ids >= n_faces)]
        print(f"Error: face index out of range [0, {n_faces - 1}], invalid: {bad}", file=sys.stderr)
        sys.exit(1)

    n = len(face_ids)
    if args.random:
        vectors = _random_tangents(mesh, face_ids, args.seed)
    elif args.vector is not None:
        v = np.array(args.vector, dtype=np.float32).reshape(1, 3)
        normals = mesh.face_normals_np[face_ids]
        vectors = project_to_tangent(
            np.broadcast_to(v, (n, 3)), normals, preserve_norm=False
        ).astype(np.float32)
    else:
        vectors = _parse_vectors_list(args.vectors_list, n)
        normals = mesh.face_normals_np[face_ids]
        vectors = project_to_tangent(vectors, normals, preserve_norm=False).astype(
            np.float32
        )

    np.savez(args.out, face_ids=face_ids, vectors=vectors)
    print(
        f"Wrote {args.out}: {n} constraints, "
        f"avg |v|={np.linalg.norm(vectors, axis=1).mean():.6f}"
    )


if __name__ == "__main__":
    main()
