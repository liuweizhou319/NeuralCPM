#!/usr/bin/env python3
"""Visualize a frame field result npz with polyscope, and export it for the
parameterization.

Reads the face-center file written by run_train_frame_field.sh
(<mesh name>.npz) and shows the frame field as a cross at every face, its
singularities, and the feature curves of the mesh.  Also in the polyscope
window (switched off at the start): the two axes e1 and e2 of the frame, the
direction of minimal curvature t_feat that the field is aligned with, the
singularity index at the vertices, and the per-face scalars of the file
(rotation angle, alignment and smoothness terms).

With --export, the mesh of the file is written as an OBJ file and the frame
field as a text file with one line "x1 y1 z1 x2 y2 z2" (e1 and e2) per face,
in the face order of the OBJ: the input of parameterization_libigl.

Example:
  python vis_framefield_npz.py
  python vis_framefield_npz.py ../surface_framefield/results/Bunny/Bunny.npz
  python vis_framefield_npz.py ../surface_framefield/results/Bunny/Bunny.npz --length 0.008
  python vis_framefield_npz.py ../surface_framefield/results/Bunny/Bunny.npz --export --no-vis
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# Default: the result of surface_framefield/run_train_frame_field.sh
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_NPZ = ROOT / "surface_framefield/results/Bunny/Bunny.npz"

# Per-face scalars of the file
SCALARS = ("theta", "w_curv", "feat_align", "feat_align_weighted",
           "smooth_transport_edge_adj", "smooth_transport_vertex_ring",
           "smooth_knn_edge_adj", "smooth_knn_vertex_ring")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("npz", type=Path, nargs="?", default=DEFAULT_NPZ,
                   help="<mesh name>.npz written by run_train_frame_field.sh")
    p.add_argument("--length", type=float, default=0.004,
                   help="length of the vectors, relative to the size of the scene")
    p.add_argument("--export", action="store_true",
                   help="write the mesh (.obj) and the frame field (.txt) for parameterization_libigl")
    p.add_argument("--obj-out", type=Path, help="OBJ file of --export (default: <npz name>.obj)")
    p.add_argument("--field-out", type=Path,
                   help="frame field file of --export (default: <npz name>_crossfield.txt)")
    p.add_argument("--no-vis", action="store_true", help="do not open the polyscope window")
    return p.parse_args()


def export(V, F, e1, e2, obj_path, field_path):
    """Write the mesh as OBJ and the frame field as one line "e1 e2" per face."""
    obj_path.parent.mkdir(parents=True, exist_ok=True)
    field_path.parent.mkdir(parents=True, exist_ok=True)
    with open(obj_path, "w", encoding="utf-8") as f:
        for x, y, z in V:
            f.write(f"v {x} {y} {z}\n")
        for i, j, k in F + 1:
            f.write(f"f {i} {j} {k}\n")
    np.savetxt(field_path, np.concatenate([e1, e2], axis=1), fmt="%.8f")
    print(f"  exported mesh        -> {obj_path}")
    print(f"  exported frame field -> {field_path}")


def singularity_index(V, F, fc, fn, direction):
    """Index of the cross field around every interior vertex (0: regular).

    The faces around a vertex are ordered by the angle of their centers in the
    tangent plane of the vertex; the index is the winding number of
    4 x (angle of direction) along this loop, divided by 4.
    """
    # Vertex normals (sum of the face normals) and a tangent basis (bx, by)
    vn = np.zeros_like(V, dtype=np.float32)
    for c in range(3):
        np.add.at(vn, F[:, c], fn)
    vn /= np.linalg.norm(vn, axis=1, keepdims=True) + 1e-12
    ref = np.tile(np.array([[1.0, 0.0, 0.0]], dtype=np.float32), (len(V), 1))
    ref[np.abs(vn[:, 0]) > 0.9] = (0.0, 1.0, 0.0)
    bx = ref - (ref * vn).sum(axis=1, keepdims=True) * vn
    bx /= np.linalg.norm(bx, axis=1, keepdims=True) + 1e-12
    by = np.cross(vn, bx)
    by /= np.linalg.norm(by, axis=1, keepdims=True) + 1e-12

    # Boundary vertices (on an edge with a single face) are skipped
    edges, counts = np.unique(np.sort(F[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1),
                              axis=0, return_counts=True)
    boundary = np.zeros(len(V), dtype=bool)
    boundary[edges[counts == 1].ravel()] = True

    vertex_faces = [[] for _ in range(len(V))]
    for f, face in enumerate(F):
        for v in face:
            vertex_faces[v].append(f)

    index = np.zeros(len(V), dtype=np.float32)
    for v, faces in enumerate(vertex_faces):
        if boundary[v] or len(faces) < 3:
            continue
        n, tx, ty = vn[v], bx[v], by[v]

        # Faces around the vertex, ordered by angle
        c = fc[faces] - V[v]
        c -= (c @ n)[:, None] * n
        keep = np.linalg.norm(c, axis=1) > 1e-12
        if keep.sum() < 3:
            continue
        c = c[keep]
        faces = np.asarray(faces, dtype=np.int64)[keep]
        faces = faces[np.argsort(np.arctan2(c @ ty, c @ tx))]

        # Winding of the field direction along this loop
        d = direction[faces]
        d -= (d @ n)[:, None] * n
        keep = np.linalg.norm(d, axis=1) > 1e-12
        if keep.sum() < 3:
            continue
        d = d[keep]
        d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-12
        phase = np.exp(4j * np.arctan2(d @ ty, d @ tx))
        index[v] = np.rint(np.angle(np.roll(phase, -1) * np.conj(phase)).sum() / (2 * np.pi)) / 4
    return index


def main():
    args = parse_args()
    data = np.load(args.npz)

    V = data["mesh_vertices"]
    F = data["mesh_faces"]
    e1 = data["e1"]
    e2 = data["e2"]
    radius = args.length / 8

    print(f"{args.npz}")
    print(f"  mesh: {len(V)} vertices, {len(F)} faces")
    print(f"  reference frame: {data['ref_mode']}")

    if args.export:
        export(V, F, e1, e2,
               args.obj_out or args.npz.with_suffix(".obj"),
               args.field_out or args.npz.with_name(args.npz.stem + "_crossfield.txt"))
    if args.no_vis:
        return

    # Singularities of the cross field
    index = singularity_index(V, F, data["face_centers"], data["face_normals_mesh"], e1)
    singular = index != 0
    print(f"  singular vertices: {int(singular.sum())} (sum of the indices {index.sum():+.2f})")

    import polyscope as ps

    ps.init()
    ps.set_up_dir("z_up")
    ps.set_ground_plane_mode("shadow_only")

    sm = ps.register_surface_mesh("mesh", V, F, smooth_shade=True)

    # Frame field: (1, 0) in the basis (e1, e2) with n_sym=4 is drawn as the
    # four directions e1, e2, -e1, -e2 of the cross.
    sm.add_tangent_vector_quantity("cross", np.tile([1.0, 0.0], (len(F), 1)), e1, e2, n_sym=4,
                                   defined_on="faces", enabled=True, length=args.length,
                                   radius=radius, color=(0.2, 0.5, 0.9))
    for name, color in (("e1", (0.9, 0.5, 0.1)), ("e2", (0.1, 0.7, 0.3)), ("t_feat", (0.1, 0.1, 0.1))):
        sm.add_vector_quantity(name, data[name], defined_on="faces", enabled=False,
                               length=args.length, radius=radius, color=color)
    for name in SCALARS:
        sm.add_scalar_quantity(name, data[name], defined_on="faces", cmap="viridis", enabled=False)

    # Singularities: index at the vertices, and the singular vertices as
    # spheres (red: positive index, blue: negative)
    sm.add_scalar_quantity("singularity index", index, defined_on="vertices",
                           cmap="coolwarm", vminmax=(-0.25, 0.25), enabled=False)
    if singular.any():
        cloud = ps.register_point_cloud("singularities", V[singular], radius=0.01)
        cloud.add_scalar_quantity("index", index[singular], cmap="coolwarm",
                                  vminmax=(-0.25, 0.25), enabled=True)

    # Feature curves (in the file when the training extracted them): the
    # midpoints of the sharp edges and the directions of these edges
    if "feat_curve_pts" in data.files and len(data["feat_curve_pts"]) > 0:
        print(f"  feature curves: {len(data['feat_curve_pts'])} sharp edges")
        curves = ps.register_point_cloud("feature curves", data["feat_curve_pts"],
                                         radius=0.002, color=(1.0, 0.2, 0.2))
        curves.add_vector_quantity("tangent", data["feat_curve_tangent"], enabled=False,
                                   length=args.length, radius=radius, color=(0.9, 0.1, 0.1))

    ps.show()


if __name__ == "__main__":
    main()
