"""
fem_screen.py -- FEM reference solver for screened-Dirichlet diffusion on a mesh.

Solves the same screened-Dirichlet variational problem as
Surfacescreendiffusion.py, but discretized with P1 cotangent FEM on the input
triangle mesh and solved by a single sparse linear system -- no neural
network, no SDF.  Its output is the reference field the network is compared
against.

Role in the pipeline
--------------------
* Command line: step 2 of a single run.  Mesh + source spec -> an npz with the
  per-vertex FEM solution, which compare_nn_fem_mse.py (error metrics) and
  vis_compare_nn_fem.py (viewer) compare with the diffusion.npz written by
  train_screen.py.
* Library: solve_fem_screened() is imported by
  evaluate_dirichlet_generalization.py, export_split_samples.py and
  export_worst_rel_l2.py; dijkstra_from() is imported by
  evaluate_dirichlet_generalization.py.

Continuous energy
-----------------
With |M| the surface area, m the number of sources, a_k the target value of
source k ("value") and weight_k its penalty weight ("weight"):

    E(u) =  (lambda_dir  / 2|M|) * int_M |grad_M u|^2 dA
          + (lambda_mass / 2|M|) * int_M  u^2          dA
          + (1 / 2m) * sum_k weight_k * mean_{S_k} (u - a_k)^2

Up to the overall factor 1/2 (which does not change the minimizer) these are
the Dirichlet, mass and source terms of the network loss in
Surfacescreendiffusion.py: area means over the surface, and a source term
averaged over the sources.  The source constraints are soft penalties: u
approaches a_k on S_k only as weight_k grows.

P1 FEM discretization (u in R^|V|, one value per vertex)
--------------------------------------------------------
    u^T K u   ~  int_M |grad_M u|^2 dA   (cotangent stiffness K = -igl.cotmatrix)
    u^T M u   ~  int_M  u^2          dA   (diagonal Voronoi mass matrix)
    |M|       =  sum of the entries of M (total surface area)
    mean_{S_k} (u - a_k)^2 = (B_k u - a_k 1)^T W_k (B_k u - a_k 1)

where B_k evaluates u at the sample points of source k (barycentric
interpolation, or vertex selection) and W_k is a diagonal matrix of weights
normalized to sum to one, so the source-wise mean matches
Surfacescreendiffusion.py.

After multiplying the whole energy by |M|, the stationarity condition is the
assembled system

    ( lambda_dir*K + lambda_mass*M
      + (|M|/m) * sum_k weight_k B_k^T W_k B_k ) u
      = (|M|/m) * sum_k weight_k B_k^T W_k (a_k 1).

The factor |M|/m (source_scale in the code) comes from the area-mean
normalization of the first two terms and from the average over the m sources;
m is the number of entries in the source list (the network divides by the
number of sources that produced samples in the epoch, which is the same
unless a source is empty, e.g. a disk_curve whose iso-contour has no segments:
such a source adds nothing here but is still counted in m).

Source spec and how each source type becomes a constraint
---------------------------------------------------------
The source spec is a JSON list of dicts, the same file train_screen.py takes
(keys that only steer the network generalization are ignored here):

    [
      {"type": "vertex",     "value": 1.0,  "weight": 100.0,
       "ids": [100, 250],    "patch_radius": 0.0},
      {"type": "disk_curve", "value": -1.0, "weight": 100.0,
       "seed": 9876, "radius": 0.2},
      {"type": "curve",      "value": 0.5, "weight": 100.0,
       "points_npy": "my_curve.npy", "closed": false}
    ]

"value" (a_k) is required for every source here; "weight" (weight_k) defaults
to 1.

* vertex: S_k is the set of listed vertices "ids".  If patch_radius > 0 it is
  replaced by all vertices whose graph distance to one of the ids is
  <= patch_radius.  W_k is uniform (1/n for n vertices), i.e. the source adds
  the diagonal penalty weight_k * (|M|/m) / n at each of these vertices.
  The ids are used as listed when patch_radius <= 0: they are not
  deduplicated, so a repeated id is counted with its multiplicity (n is the
  length of the list and the penalty is added once per occurrence).  With
  patch_radius > 0 the expanded set is unique.
* disk_curve: S_k is the iso-contour {graph distance from vertex "seed"
  = "radius"}; both keys are required here.  The vertex distances are
  interpolated linearly on each face, and every face in which exactly two
  edges cross the level contributes one straight segment.  The penalty is the
  arclength average over these segments, integrated with two Gauss points per
  segment (exact, because u is linear on a face) and barycentric
  interpolation.
* curve: S_k is a polyline given inline as "points" (K x 3) or as
  "points_npy" (a .npy file, path relative to the working directory, resolved
  by load_sources), closed if "closed" is true.  Two Gauss points are placed
  on every straight segment of the polyline, each is projected to the closest
  point of the mesh, and the penalty is the average weighted by the lengths of
  the unprojected segments.  The points are used as given: they must already
  be in the normalized mesh coordinates described below.

Graph-distance convention (radius, patch_radius)
------------------------------------------------
"radius" and "patch_radius" are Dijkstra distances on the mesh edge graph
built by _mesh_edge_graph_csr().  That graph lists every edge once per
adjacent face and per direction, and duplicate entries are summed, so an
interior edge (shared by two faces) has weight 2 * |edge| and a boundary edge
1 * |edge|.  On a closed mesh the graph distance is therefore twice the
shortest edge-path length: "radius": 0.2 is the contour at edge-path distance
0.1 from the seed.  Surfacescreendiffusion.py builds its graph in the same
way, so a source spec means the same thing for the network and for the FEM
reference.  The convention is kept as it is because the shipped source specs
and all reported numbers depend on it.

Mesh normalization
------------------
Unless --no-normalize is given, load_mesh() translates the mesh so that the
center of its axis-aligned bounding box is the origin and then divides by the
largest vertex norm (the farthest vertex ends up at distance 1 from the
origin).  This is the normalization of Mesh_geometry.MeshGeometry used for
training; it is computed in float64 here and in float32 on the network side,
so the two vertex arrays agree to about 1e-7 only.  Radii and curve points
refer to these normalized coordinates.

Output
------
An npz file with mesh_vertices (V x 3, float32: the vertices used by the
solve, normalized unless --no-normalize is given), mesh_faces (F x 3, int64),
u_vertex (V, float32: the FEM solution at the vertices), value_min /
value_max (range of the source values), lambda_dir, lambda_mass, and the
source geometry for visualization: src_pts / src_pts_value (vertex sources)
and src_seg_p1 / src_seg_p2 / src_seg_value (contour and polyline segments).

Requires numpy, scipy and the libigl Python bindings (igl); polyscope is
needed only for --vis.

Usage (from the repository root)
--------------------------------
The paths below continue the train_screen.py usage example (--out-dir
results/single/fandisk).  run_single.sh writes by default to
results/single/fandisk_res200_halo6 instead.

    # FEM reference for a single run.  Use the mesh, source spec and lambdas of
    # the training run (run_single.sh does this).
    python fem_screen.py data/fandisk.obj --source-spec data/fandisk_4v.json --out results/single/fandisk/fem_diffusion.npz

    # interactive view of a result
    python fem_screen.py --vis results/single/fandisk/fem_diffusion.npz

    # optional: write a source spec from the command line with
    # make_screen_sources.py first
    python make_screen_sources.py -o my_sources.json --vertex-ids 100,250 --vertex-value 1.0 --vertex-weight 100
"""

import os
import sys
import json
import argparse
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse.csgraph import dijkstra as _scipy_dijkstra
import igl


# ============================================================
# 1. Mesh I/O & FEM operators
# ============================================================

def load_mesh(path, normalize=True):
    """Load mesh. Normalization matches MeshGeometry (NN training).

    Returns float64 vertices and int64 faces.  With normalize=True the mesh
    is centered at its bounding-box center and scaled so that the farthest
    vertex has norm 1.
    """
    V, F = igl.read_triangle_mesh(path)
    V = V.astype(np.float64)
    F = F.astype(np.int64)
    if normalize:
        # Same as MeshGeometry: bbox center + unit-sphere scale
        center = 0.5 * (V.max(axis=0) + V.min(axis=0))
        V = V - center
        scale = float(np.sqrt((V ** 2).sum(axis=1)).max())
        if scale > 1e-14:
            V = V / scale
        print(f"  Normalized (MeshGeometry-compatible): center={center}, scale={scale:.4f}")
    print(f"Mesh: {len(V)} verts, {len(F)} faces"
          f"{'  (normalized)' if normalize else ''}")
    return V, F


def fem_operators(V, F):
    """Cotangent stiffness K (positive semidefinite) and Voronoi mass M."""
    L = igl.cotmatrix(V, F)              # NSD in libigl
    K = (-L).tocsr()
    M = igl.massmatrix(V, F, igl.MASSMATRIX_TYPE_VORONOI).tocsr()
    return K, M


# ============================================================
# 2. Geodesic-disk iso-contour extraction
# ============================================================

def _mesh_edge_graph_csr(V, F):
    """Mesh edge graph as a sparse matrix of edge lengths.

    Every face contributes its three edges in both directions and csr_matrix
    sums duplicate entries, so an interior edge (two adjacent faces) gets
    weight 2 * length and a boundary edge 1 * length.  Graph distances on a
    closed mesh are therefore twice the edge-path length.  This doubled
    convention is the same as in Surfacescreendiffusion.py and defines what
    'radius' and 'patch_radius' mean; it must not be changed.
    """
    edges = np.concatenate(
        [F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]],
         F[:, [1, 0]], F[:, [2, 1]], F[:, [0, 2]]], axis=0)
    w = np.linalg.norm(V[edges[:, 0]] - V[edges[:, 1]], axis=1)
    return sp.csr_matrix((w, (edges[:, 0], edges[:, 1])),
                         shape=(len(V), len(V)))


def dijkstra_from(V, F, seed_id, _cache={}):
    """Per-vertex geodesic-edge distance from seed_id (Dijkstra, cached).

    Distances are measured on the graph of _mesh_edge_graph_csr, i.e. in the
    doubled convention described there.  The graph and the result of every
    seed are memoized in the mutable default argument `_cache`, keyed by
    id(V): keep one vertex array alive and unmodified per mesh, and do not
    modify the returned array.
    """
    key = (id(V), int(seed_id))
    if key in _cache:
        return _cache[key]
    G = _cache.setdefault(('graph', id(V)), _mesh_edge_graph_csr(V, F))
    d = _scipy_dijkstra(G, directed=False, indices=int(seed_id))
    _cache[key] = d.astype(np.float64)
    return _cache[key]


def face_iso_segments(V, F, dist_vertex, r):
    """For iso-value r, return one segment per crossing face plus its face id.

    A face contributes a segment only if exactly two of its edges cross the
    level r (strict inequalities at both edge endpoints).

    Returns
    -------
    p1, p2 : (S, 3) segment endpoints (on the surface)
    face_id: (S,) face index for each segment
    """
    pts = V[F]                    # (F, 3, 3)
    dv  = dist_vertex[F]          # (F, 3)
    a_idx = np.array([0, 1, 2])
    b_idx = np.array([1, 2, 0])
    da = dv[:, a_idx]; db = dv[:, b_idx]
    crosses = ((da < r) & (db > r)) | ((da > r) & (db < r))

    denom = db - da
    denom = np.where(np.abs(denom) < 1e-20, 1e-20, denom)
    t = (r - da) / denom
    pa = pts[:, a_idx, :]; pb = pts[:, b_idx, :]
    edge_pts = pa + t[..., None] * (pb - pa)

    valid = crosses.sum(axis=1) == 2
    if not valid.any():
        z = np.zeros((0, 3), dtype=np.float64)
        return z, z.copy(), np.zeros(0, dtype=np.int64)

    vf = np.where(valid)[0]
    vc = crosses[vf]; vp = edge_pts[vf]
    order = np.argsort(-vc.astype(np.int8), axis=1)
    row = np.arange(len(vf))
    p1 = vp[row, order[:, 0]]
    p2 = vp[row, order[:, 1]]
    return p1, p2, vf.astype(np.int64)


# ============================================================
# 3. Barycentric interpolation matrix B
# ============================================================

def barycentric_coords_batch(P, A, B, C, eps=1e-14):
    """Barycentric (u, v, w) of P w.r.t. triangles ABC; u + v + w = 1."""
    v0 = B - A; v1 = C - A; v2 = P - A
    d00 = np.einsum("ij,ij->i", v0, v0)
    d01 = np.einsum("ij,ij->i", v0, v1)
    d11 = np.einsum("ij,ij->i", v1, v1)
    d20 = np.einsum("ij,ij->i", v2, v0)
    d21 = np.einsum("ij,ij->i", v2, v1)
    den = d00 * d11 - d01 * d01
    den = np.where(np.abs(den) < eps, eps, den)
    v = (d11 * d20 - d01 * d21) / den
    w = (d00 * d21 - d01 * d20) / den
    u = 1.0 - v - w
    return np.stack([u, v, w], axis=1)


def build_interp_matrix(V, F, face_ids, pts):
    """B such that (B @ u)[i] is the barycentric value at pts[i] in face_ids[i]."""
    m = len(pts)
    n = len(V)
    if m == 0:
        return sp.csr_matrix((0, n))
    tri = F[face_ids]
    bary = barycentric_coords_batch(
        pts.astype(np.float64),
        V[tri[:, 0]], V[tri[:, 1]], V[tri[:, 2]])
    rows = np.repeat(np.arange(m), 3)
    cols = tri.reshape(-1)
    vals = bary.reshape(-1)
    return sp.csr_matrix((vals, (rows, cols)), shape=(m, n))


def project_points_to_mesh(V, F, pts):
    """Snap each point to the nearest mesh face; returns (face_ids, projected_pts)."""
    pts = np.asarray(pts, dtype=np.float64)
    _sqd, face_ids, projected = igl.point_mesh_squared_distance(pts, V, F)
    return face_ids.astype(np.int64), projected.astype(np.float64)


# ============================================================
# 4. Source contributions to the linear system
# ============================================================

def add_vertex_source(A, b, V, F, src, source_scale=1.0, verbose=True):
    """Soft point penalty averaged over the selected vertices.

    Optional `patch_radius` fattens the source to all vertices within that
    Dijkstra distance (doubled graph convention, see _mesh_edge_graph_csr).
    Uniform counting weights match the NN source sampler.
    """
    n = A.shape[0]
    ids = np.asarray(src['ids'], dtype=np.int64)
    if len(ids) == 0:
        return A, b
    val = float(src['value'])
    lam = float(src.get('weight', 1.0))
    pr  = float(src.get('patch_radius', 0.0))

    if pr > 0:
        expanded = set()
        for vid in ids:
            d = dijkstra_from(V, F, vid)
            expanded.update(np.where(d <= pr)[0].tolist())
        ids = np.asarray(sorted(expanded), dtype=np.int64)

    # The NN takes a mean within each source. source_scale is |M|/m, the
    # factor obtained after scaling the surface-average loss by |M|.
    w = np.full(len(ids), float(source_scale) / len(ids), dtype=np.float64)

    diag = np.zeros(n, dtype=np.float64)
    rhs  = np.zeros(n, dtype=np.float64)
    np.add.at(diag, ids, lam * w)
    np.add.at(rhs,  ids, lam * w * val)
    A = A + sp.diags(diag, format='csr')
    b = b + rhs
    if verbose:
        print(f"    vertex     : value={val:+.3f}  weight={lam:.2f}  "
              f"|ids|={len(ids)}  patch_radius={pr:.4f}")
    return A, b


def add_curve_source_segments(A, b, V, F, p1, p2, face_ids, value, lam,
                              source_scale=1.0):
    """Average a curve penalty using exact two-point quadrature per P1 segment."""
    if len(p1) == 0:
        return A, b
    lengths = np.linalg.norm(p2 - p1, axis=1)
    total_length = float(lengths.sum())
    if total_length <= 1e-20:
        return A, b

    offset = 1.0 / (2.0 * np.sqrt(3.0))
    t = np.array([0.5 - offset, 0.5 + offset], dtype=np.float64)
    quad_pts = (p1[:, None, :] + t[None, :, None]
                * (p2 - p1)[:, None, :]).reshape(-1, 3)
    quad_faces = np.repeat(face_ids, 2)
    quad_weights = np.repeat(
        0.5 * float(source_scale) * lengths / total_length, 2)

    B = build_interp_matrix(V, F, quad_faces, quad_pts)
    W = sp.diags(quad_weights, format='csr')
    a = np.full(len(quad_pts), float(value), dtype=np.float64)
    A = A + lam * (B.T @ W @ B)
    b = b + lam * (B.T @ W @ a)
    return A, b


def add_disk_curve_source(A, b, V, F, src, source_scale=1.0, verbose=True):
    """Geodesic-disk iso-contour source.  Uses Dijkstra on the edge graph
    to compute distances from the seed vertex, then extracts segments."""
    seed = int(src['seed'])
    r    = float(src['radius'])
    val  = float(src['value'])
    lam  = float(src.get('weight', 1.0))
    d = dijkstra_from(V, F, seed)
    p1, p2, face_ids = face_iso_segments(V, F, d, r)
    L = np.linalg.norm(p2 - p1, axis=1).sum() if len(p1) > 0 else 0.0
    if verbose:
        print(f"    disk_curve : value={val:+.3f}  weight={lam:.2f}  "
              f"seed={seed}  r={r:.4f}  {len(p1)} segs  perimeter~{L:.4f}")
    return add_curve_source_segments(
        A, b, V, F, p1, p2, face_ids, val, lam,
        source_scale=source_scale)


def add_curve_source(A, b, V, F, src, source_scale=1.0, verbose=True):
    """Polyline source using two-point quadrature projected pointwise to M."""
    pts = np.asarray(src['points'], dtype=np.float64)
    val = float(src['value'])
    lam = float(src.get('weight', 1.0))
    closed = bool(src.get('closed', False))
    if len(pts) < 2:
        if verbose:
            print(f"    curve      : skipped (need >=2 points)")
        return A, b
    if closed:
        p1 = pts
        p2 = np.roll(pts, -1, axis=0)
    else:
        p1 = pts[:-1]
        p2 = pts[1:]
    lengths = np.linalg.norm(p2 - p1, axis=1)
    L = float(lengths.sum())
    if L <= 1e-20:
        return A, b

    offset = 1.0 / (2.0 * np.sqrt(3.0))
    t = np.array([0.5 - offset, 0.5 + offset], dtype=np.float64)
    quad_raw = (p1[:, None, :] + t[None, :, None]
                * (p2 - p1)[:, None, :]).reshape(-1, 3)
    quad_faces, quad_pts = project_points_to_mesh(V, F, quad_raw)
    quad_weights = np.repeat(
        0.5 * float(source_scale) * lengths / L, 2)
    B = build_interp_matrix(V, F, quad_faces, quad_pts)
    W = sp.diags(quad_weights, format='csr')
    a = np.full(len(quad_pts), val, dtype=np.float64)
    A = A + lam * (B.T @ W @ B)
    b = b + lam * (B.T @ W @ a)
    if verbose:
        print(f"    curve      : value={val:+.3f}  weight={lam:.2f}  "
              f"K={len(pts)}  length~{L:.4f}  closed={closed}")
    return A, b


# ============================================================
# 5. Top-level solver
# ============================================================

def solve_fem_screened(V, F, sources, lambda_dir=1.0, lambda_mass=1.0,
                      verbose=True):
    """Assemble and solve the FEM screened-Dirichlet system.

    Solves (lambda_dir*K + lambda_mass*M + (|M|/m) * sum_k weight_k
    B_k^T W_k B_k) u = (|M|/m) * sum_k weight_k B_k^T W_k (a_k 1) with one
    sparse direct solve (see the module docstring).

    Parameters
    ----------
    V, F    : (|V|, 3) float64 vertices and (|F|, 3) int64 faces; pass the
              normalized mesh when comparing with the network
    sources : list of source dicts ('vertex', 'disk_curve' or 'curve'); a
              curve must carry its points under 'points'
    lambda_dir, lambda_mass : weights of the Dirichlet and mass terms
    verbose : print the assembly log, the relative residual and the mean of
              u over every source

    Returns
    -------
    u   : (|V|,) per-vertex solution
    info: dict with assembled segments / source data, useful for viz.
          Keys: src_pts, src_pts_value (vertex sources), src_seg_p1,
          src_seg_p2, src_seg_value (contour and polyline segments).
    """
    n = len(V)
    K, M = fem_operators(V, F)
    A = lambda_dir * K + lambda_mass * M
    b = np.zeros(n, dtype=np.float64)
    surface_area = float(M.sum())
    if not np.isfinite(surface_area) or surface_area <= 0.0:
        raise ValueError("Mesh must have positive finite area")
    source_scale = surface_area / len(sources) if sources else 1.0

    if verbose:
        print(f"\nAssembly  lambda_dir={lambda_dir}  lambda_mass={lambda_mass}  "
              f"ell ~ sqrt(lambda_dir/lambda_mass) = "
              f"{(lambda_dir / max(lambda_mass, 1e-12)) ** 0.5:.4f}")
        print(f"Sources ({len(sources)}):")

    # collect viz segments / vertex-source markers per source for export
    viz_seg_p1, viz_seg_p2, viz_seg_v = [], [], []
    viz_pts, viz_pts_v = [], []

    for src in sources:
        stype = src.get('type')
        if stype == 'vertex':
            A, b = add_vertex_source(
                A, b, V, F, src, source_scale=source_scale,
                verbose=verbose)
            ids = np.asarray(src['ids'], dtype=np.int64)
            pr  = float(src.get('patch_radius', 0.0))
            if pr > 0:
                expanded = set()
                for vid in ids:
                    d = dijkstra_from(V, F, vid)
                    expanded.update(np.where(d <= pr)[0].tolist())
                ids = np.asarray(sorted(expanded), dtype=np.int64)
            viz_pts.append(V[ids])
            viz_pts_v.append(np.full(len(ids), src['value'], dtype=np.float64))

        elif stype == 'disk_curve':
            A, b = add_disk_curve_source(
                A, b, V, F, src, source_scale=source_scale,
                verbose=verbose)
            seed = int(src['seed']); r = float(src['radius'])
            d = dijkstra_from(V, F, seed)
            p1, p2, _ = face_iso_segments(V, F, d, r)
            if len(p1) > 0:
                viz_seg_p1.append(p1); viz_seg_p2.append(p2)
                viz_seg_v.append(np.full(len(p1), src['value'], dtype=np.float64))

        elif stype == 'curve':
            A, b = add_curve_source(
                A, b, V, F, src, source_scale=source_scale,
                verbose=verbose)
            pts = np.asarray(src['points'], dtype=np.float64)
            if len(pts) >= 2:
                _, pp = project_points_to_mesh(V, F, pts)
                if bool(src.get('closed', False)):
                    p1 = pp; p2 = np.roll(pp, -1, axis=0)
                else:
                    p1 = pp[:-1]; p2 = pp[1:]
                viz_seg_p1.append(p1); viz_seg_p2.append(p2)
                viz_seg_v.append(np.full(len(p1), src['value'], dtype=np.float64))

        else:
            raise ValueError(f"Unknown source type: {stype!r}")

    # ---- Solve ----
    if verbose:
        print(f"\nSolving {n}x{n} sparse system ...")
    A = A.tocsc()
    u = spla.spsolve(A, b)

    if verbose:
        # quick residual + per-source predicted means
        resid = float(np.linalg.norm(A @ u - b) / max(np.linalg.norm(b), 1e-30))
        print(f"  rel. residual = {resid:.3e}")
        print(f"  u in [{u.min():.4f}, {u.max():.4f}]   <u>={u.mean():.4f}")
        for k, src in enumerate(sources):
            stype = src['type']; val = float(src['value'])
            if stype == 'vertex':
                ids = np.asarray(src['ids'], dtype=np.int64)
                pr  = float(src.get('patch_radius', 0.0))
                if pr > 0:
                    expanded = set()
                    for vid in ids:
                        d = dijkstra_from(V, F, vid)
                        expanded.update(np.where(d <= pr)[0].tolist())
                    ids = np.asarray(sorted(expanded), dtype=np.int64)
                m = float(u[ids].mean())
            elif stype == 'disk_curve':
                d = dijkstra_from(V, F, int(src['seed']))
                p1, p2, fids = face_iso_segments(V, F, d, float(src['radius']))
                if len(p1) == 0:
                    m = float('nan')
                else:
                    mid = 0.5 * (p1 + p2)
                    Bm = build_interp_matrix(V, F, fids, mid)
                    m = float((Bm @ u).mean())
            elif stype == 'curve':
                pts = np.asarray(src['points'], dtype=np.float64)
                if len(pts) < 2:
                    m = float('nan')
                else:
                    fids, pp = project_points_to_mesh(V, F, pts)
                    Bm = build_interp_matrix(V, F, fids, pp)
                    m = float((Bm @ u).mean())
            print(f"  source[{k}] {stype:>10s}  target={val:+.3f}  "
                  f"<u_S>={m:+.4f}")

    info = {
        'src_seg_p1':  np.concatenate(viz_seg_p1, axis=0) if viz_seg_p1
                       else np.zeros((0, 3), dtype=np.float64),
        'src_seg_p2':  np.concatenate(viz_seg_p2, axis=0) if viz_seg_p2
                       else np.zeros((0, 3), dtype=np.float64),
        'src_seg_value': np.concatenate(viz_seg_v, axis=0) if viz_seg_v
                       else np.zeros((0,),  dtype=np.float64),
        'src_pts':       np.concatenate(viz_pts, axis=0) if viz_pts
                       else np.zeros((0, 3), dtype=np.float64),
        'src_pts_value': np.concatenate(viz_pts_v, axis=0) if viz_pts_v
                       else np.zeros((0,),  dtype=np.float64),
    }
    return u, info


# ============================================================
# 6. Source-spec loader (compatible with Surfacescreendiffusion JSON)
# ============================================================

def load_sources(path):
    """Read a source-spec JSON (a list of dicts) for the command-line solver.

    A 'points_npy' entry is loaded with np.load (path relative to the working
    directory) and stored under 'points'.
    """
    with open(path, 'r') as f:
        sources = json.load(f)
    if not isinstance(sources, list):
        raise SystemExit("source-spec must be a JSON list of dicts")
    out = []
    for s in sources:
        s = dict(s)
        if 'points_npy' in s:
            s['points'] = np.load(s.pop('points_npy'))
        out.append(s)
    return out


# ============================================================
# 7. Export & visualize
# ============================================================

def export_npz(save_path, V, F, u, info, lambda_dir, lambda_mass, sources):
    """Save mesh, solution and source geometry to an npz (keys: module docstring)."""
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    vals = [float(s['value']) for s in sources] if sources else [0.0, 1.0]
    v_min, v_max = float(min(vals)), float(max(vals))
    np.savez(save_path,
             mesh_vertices=V.astype(np.float32),
             mesh_faces=F.astype(np.int64),
             u_vertex=u.astype(np.float32),
             value_min=np.float32(v_min),
             value_max=np.float32(v_max),
             lambda_dir=np.float32(lambda_dir),
             lambda_mass=np.float32(lambda_mass),
             src_pts=info['src_pts'].astype(np.float32),
             src_pts_value=info['src_pts_value'].astype(np.float32),
             src_seg_p1=info['src_seg_p1'].astype(np.float32),
             src_seg_p2=info['src_seg_p2'].astype(np.float32),
             src_seg_value=info['src_seg_value'].astype(np.float32))
    print(f"\nExported -> {save_path}")


def visualize_polyscope(npz_path):
    """Show an exported npz in polyscope: the field, source points and curves."""
    import polyscope as ps
    d = np.load(npz_path)
    V = d['mesh_vertices']; F = d['mesh_faces']; u = d['u_vertex']
    v_min = float(d['value_min']) if 'value_min' in d.files else float(u.min())
    v_max = float(d['value_max']) if 'value_max' in d.files else float(u.max())

    ps.init()
    ps.set_up_dir("z_up")
    ps.set_ground_plane_mode("shadow_only")
    sm = ps.register_surface_mesh("mesh", V, F, smooth_shade=True)

    if v_min < 0 < v_max:
        vabs = max(abs(v_min), abs(v_max))
        sm.add_scalar_quantity("u (FEM diffusion)", u, cmap='coolwarm',
                               vminmax=(-vabs, vabs), enabled=True)
    else:
        sm.add_scalar_quantity("u (FEM diffusion)", u, cmap='viridis',
                               vminmax=(v_min, v_max), enabled=True)
    sm.add_scalar_quantity("u (autoscale)", u, cmap='viridis', enabled=False)

    if len(d['src_pts']) > 0:
        pc = ps.register_point_cloud("source_points", d['src_pts'], radius=0.006)
        if v_min < 0 < v_max:
            vabs = max(abs(v_min), abs(v_max))
            pc.add_scalar_quantity("value", d['src_pts_value'], cmap='coolwarm',
                                   vminmax=(-vabs, vabs), enabled=True)
        else:
            pc.add_scalar_quantity("value", d['src_pts_value'], cmap='viridis',
                                   vminmax=(v_min, v_max), enabled=True)

    if len(d['src_seg_p1']) > 0:
        p1 = d['src_seg_p1']; p2 = d['src_seg_p2']
        nodes = np.concatenate([p1, p2], axis=0)
        S = len(p1)
        edges = np.stack([np.arange(S), np.arange(S, 2 * S)], axis=1)
        cn = ps.register_curve_network("source_curves", nodes, edges, radius=0.003)
        if v_min < 0 < v_max:
            vabs = max(abs(v_min), abs(v_max))
            cn.add_scalar_quantity("value", d['src_seg_value'], defined_on='edges',
                                   cmap='coolwarm', vminmax=(-vabs, vabs),
                                   enabled=True)
        else:
            cn.add_scalar_quantity("value", d['src_seg_value'], defined_on='edges',
                                   cmap='viridis', vminmax=(v_min, v_max),
                                   enabled=True)
    ps.show()


# ============================================================
# 8. CLI
# ============================================================

def parse_args():
    """Command-line arguments of the solver / viewer."""
    p = argparse.ArgumentParser()
    p.add_argument('mesh', nargs='?', type=str,
                   help='input mesh (.obj/.ply/.off/...)')
    p.add_argument('--source-spec', type=str, default=None,
                   help='JSON file with the list of sources')
    p.add_argument('--lambda-dir',  type=float, default=1.0,
                   help='weight lambda_dir of the Dirichlet term (default: 1.0)')
    p.add_argument('--lambda-mass', type=float, default=1.0,
                   help='weight lambda_mass of the mass term (default: 1.0)')
    p.add_argument('--no-normalize', action='store_true',
                   help='do NOT center the mesh at its bounding-box center and '
                        'rescale it so that the farthest vertex has norm 1')
    p.add_argument('--out', type=str, default='fem_results/diffusion.npz',
                   help='output .npz file (default: fem_results/diffusion.npz)')
    p.add_argument('--vis', type=str, default=None,
                   help='visualize the given .npz and exit (skip solve)')
    return p.parse_args()


def main():
    """Entry point: view an npz (--vis), or solve and export."""
    args = parse_args()

    if args.vis is not None:
        visualize_polyscope(args.vis)
        return

    if args.mesh is None:
        raise SystemExit("Need a mesh path (or use --vis <npz>).")
    if args.source_spec is None:
        raise SystemExit("Need --source-spec <json>.")

    V, F = load_mesh(args.mesh, normalize=not args.no_normalize)
    sources = load_sources(args.source_spec)

    u, info = solve_fem_screened(
        V, F, sources,
        lambda_dir=args.lambda_dir,
        lambda_mass=args.lambda_mass,
    )

    export_npz(args.out, V, F, u, info,
               args.lambda_dir, args.lambda_mass, sources)
    print(f"  Visualize with:  python {sys.argv[0]} --vis {args.out}")


if __name__ == '__main__':
    main()
