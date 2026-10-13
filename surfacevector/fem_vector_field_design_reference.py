#!/usr/bin/env python3
"""
FEM reference for surface vector field design.

Energy:
    min_U  int_M ||grad U||_F^2 dA  + lambda_tangent int_M (U.n)^2 dA
    s.t.   U(face_center_i) = target_i

Discretization:
    - P1 vertex-based vector-valued FEM in ambient R^3.
    - Cotangent stiffness matrix for each Cartesian component.
    - Face-center constraints are barycentric averages of the three vertices.
    - Tangency is handled by a soft FEM penalty and optional output projection.

Constraint format is compatible with make_vecfield_constraints.py:
    npz with arrays:
      face_ids : (K,) int64
      vectors  : (K,3) float32

Output npz is compatible with visualize_polyscope() in SurfaceVectorFieldDesign.py.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


# -----------------------------------------------------------------------------
# Geometry helpers
# -----------------------------------------------------------------------------

def _normalize_mesh_unit_box(V: np.ndarray) -> np.ndarray:
    V = np.asarray(V, dtype=np.float64)
    mn = V.min(axis=0)
    mx = V.max(axis=0)
    center = 0.5 * (mn + mx)
    scale = float(np.max(mx - mn))
    if scale <= 1e-30:
        return V - center
    return (V - center) / scale


def load_mesh(mesh_path: str, normalize: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Load mesh. Prefer the user's MeshGeometry loader to match training."""
    # First try the project loader, because it should match train_vecfield.py.
    try:
        from Mesh_geometry import MeshGeometry  # type: ignore
        mesh = MeshGeometry(mesh_path, normalize=normalize, device="cpu")
        V = np.asarray(mesh.vertices_np, dtype=np.float64)
        F = np.asarray(mesh.faces_np, dtype=np.int64)
        return V, F
    except Exception:
        pass

    # Fallback: trimesh.
    try:
        import trimesh  # type: ignore
        tm = trimesh.load(mesh_path, process=False)
        if isinstance(tm, trimesh.Scene):
            meshes = [g for g in tm.geometry.values() if hasattr(g, "vertices")]
            if len(meshes) == 0:
                raise ValueError(f"No mesh geometry found in scene: {mesh_path}")
            tm = trimesh.util.concatenate(meshes)
        V = np.asarray(tm.vertices, dtype=np.float64)
        F = np.asarray(tm.faces, dtype=np.int64)
    except Exception as e:
        raise RuntimeError(
            "Failed to load mesh. Install trimesh or make sure Mesh_geometry.py is importable."
        ) from e

    if normalize:
        V = _normalize_mesh_unit_box(V)
    return V, F


def face_geometry(V: np.ndarray, F: np.ndarray):
    v0 = V[F[:, 0]]
    v1 = V[F[:, 1]]
    v2 = V[F[:, 2]]
    centers = (v0 + v1 + v2) / 3.0
    raw = np.cross(v1 - v0, v2 - v0)
    dbl_area = np.linalg.norm(raw, axis=1)
    normals = raw / np.maximum(dbl_area[:, None], 1e-30)
    areas = 0.5 * dbl_area
    return centers, normals, areas


def vertex_area_lumped(nv: int, F: np.ndarray, face_areas: np.ndarray) -> np.ndarray:
    m = np.zeros(nv, dtype=np.float64)
    for c in range(3):
        np.add.at(m, F[:, c], face_areas / 3.0)
    return m


def vertex_normals_area_weighted(
    nv: int, F: np.ndarray, face_normals: np.ndarray, face_areas: np.ndarray
) -> np.ndarray:
    N = np.zeros((nv, 3), dtype=np.float64)
    weighted = face_normals * face_areas[:, None]
    for c in range(3):
        np.add.at(N, F[:, c], weighted)
    norm = np.linalg.norm(N, axis=1, keepdims=True)
    return N / np.maximum(norm, 1e-30)


def project_to_tangent(v: np.ndarray, n: np.ndarray, preserve_norm: bool = False) -> np.ndarray:
    dot = np.sum(v * n, axis=-1, keepdims=True)
    t = v - dot * n
    if preserve_norm:
        vnorm = np.linalg.norm(v, axis=-1, keepdims=True)
        tnorm = np.linalg.norm(t, axis=-1, keepdims=True)
        t = t * np.where(tnorm > 1e-30, vnorm / np.maximum(tnorm, 1e-30), 1.0)
    return t


# -----------------------------------------------------------------------------
# FEM matrices
# -----------------------------------------------------------------------------

def cotangent_laplacian_positive(V: np.ndarray, F: np.ndarray) -> sp.csr_matrix:
    """Positive semi-definite cotangent stiffness matrix.

    Tries gpytoolbox/libigl first, then falls back to a small internal cotan builder.
    """
    # gpytoolbox returns a positive cotangent Laplacian.
    try:
        from gpytoolbox import cotangent_laplacian  # type: ignore
        L = cotangent_laplacian(V, F).tocsr().astype(np.float64)
        L = 0.5 * (L + L.T)
        if np.mean(L.diagonal()) < 0:
            L = -L
        return L.tocsr()
    except Exception:
        pass

    # libigl's cotmatrix convention is usually negative semi-definite, so negate.
    try:
        import igl  # type: ignore
        L = (-igl.cotmatrix(V, F)).tocsr().astype(np.float64)
        L = 0.5 * (L + L.T)
        if np.mean(L.diagonal()) < 0:
            L = -L
        return L.tocsr()
    except Exception:
        pass

    # Fallback implementation.
    nv = V.shape[0]
    I, J, W = [], [], []

    def cot(a: np.ndarray, b: np.ndarray) -> float:
        cr = np.linalg.norm(np.cross(a, b))
        return float(np.dot(a, b) / max(cr, 1e-30))

    def add_edge(i: int, j: int, w: float):
        # Positive stiffness: w * ||u_i - u_j||^2 contribution.
        I.extend([i, j, i, j])
        J.extend([i, j, j, i])
        W.extend([w, w, -w, -w])

    for tri in F:
        i, j, k = map(int, tri)
        vi, vj, vk = V[i], V[j], V[k]
        coti = cot(vj - vi, vk - vi)  # opposite edge jk
        cotj = cot(vk - vj, vi - vj)  # opposite edge ki
        cotk = cot(vi - vk, vj - vk)  # opposite edge ij
        add_edge(j, k, 0.5 * coti)
        add_edge(k, i, 0.5 * cotj)
        add_edge(i, j, 0.5 * cotk)

    L = sp.coo_matrix((W, (I, J)), shape=(nv, nv)).tocsr()
    L = 0.5 * (L + L.T)
    return L.tocsr()


def build_vector_energy(
    L: sp.csr_matrix,
    vertex_mass: np.ndarray,
    vertex_normals: np.ndarray,
    lambda_tangent: float = 100.0,
    epsilon_mass: float = 1e-10,
) -> sp.csr_matrix:
    """Build Q for component-major unknown z=[Ux_all, Uy_all, Uz_all]."""
    nv = L.shape[0]
    M = sp.diags(vertex_mass, format="csr")
    Q = sp.kron(sp.eye(3, format="csr"), L + epsilon_mass * M, format="csr")

    if lambda_tangent <= 0:
        return Q.tocsr()

    rows, cols, vals = [], [], []
    N = vertex_normals
    wv = lambda_tangent * vertex_mass
    for a in range(3):
        for b in range(3):
            vals_ab = wv * N[:, a] * N[:, b]
            nz = np.abs(vals_ab) > 0
            ids = np.nonzero(nz)[0]
            rows.extend((a * nv + ids).tolist())
            cols.extend((b * nv + ids).tolist())
            vals.extend(vals_ab[ids].tolist())
    Qtan = sp.coo_matrix((vals, (rows, cols)), shape=(3 * nv, 3 * nv)).tocsr()
    return (Q + Qtan).tocsr()


def deduplicate_face_constraints(
    face_ids: np.ndarray, vectors: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Average duplicate constraints on the same face to avoid rank problems."""
    face_ids = np.asarray(face_ids, dtype=np.int64).reshape(-1)
    vectors = np.asarray(vectors, dtype=np.float64).reshape(-1, 3)
    order = np.argsort(face_ids)
    f_sorted = face_ids[order]
    v_sorted = vectors[order]

    out_f, out_v = [], []
    start = 0
    while start < len(f_sorted):
        end = start + 1
        while end < len(f_sorted) and f_sorted[end] == f_sorted[start]:
            end += 1
        out_f.append(f_sorted[start])
        out_v.append(v_sorted[start:end].mean(axis=0))
        start = end
    return np.asarray(out_f, dtype=np.int64), np.asarray(out_v, dtype=np.float64)


def build_face_center_constraints(
    F: np.ndarray, face_ids: np.ndarray, targets: np.ndarray, nv: int
) -> Tuple[sp.csr_matrix, np.ndarray]:
    """Aeq z = beq, with z=[Ux_all, Uy_all, Uz_all]."""
    k = len(face_ids)
    rows, cols, vals = [], [], []
    beq = targets.reshape(-1).astype(np.float64)  # row order: face0 xyz, face1 xyz, ...

    for local_i, fid in enumerate(face_ids):
        tri = F[int(fid)]
        for comp in range(3):
            row = 3 * local_i + comp
            for v in tri:
                rows.append(row)
                cols.append(comp * nv + int(v))
                vals.append(1.0 / 3.0)

    Aeq = sp.coo_matrix((vals, (rows, cols)), shape=(3 * k, 3 * nv)).tocsr()
    return Aeq, beq


def solve_kkt(Q: sp.csr_matrix, Aeq: sp.csr_matrix, beq: np.ndarray) -> np.ndarray:
    n = Q.shape[0]
    m = Aeq.shape[0]
    K = sp.bmat([[Q, Aeq.T], [Aeq, None]], format="csc")
    rhs = np.concatenate([np.zeros(n, dtype=np.float64), beq.astype(np.float64)])
    sol = spla.spsolve(K, rhs)
    if not np.all(np.isfinite(sol[:n])):
        raise FloatingPointError("KKT solve produced NaN/Inf. Try larger --epsilon-mass.")
    return sol[:n]


@dataclass
class FEMResult:
    V: np.ndarray
    F: np.ndarray
    U_vertex_raw: np.ndarray
    U_vertex: np.ndarray
    U_face_raw: np.ndarray
    U_face: np.ndarray
    face_centers: np.ndarray
    face_normals: np.ndarray
    constraint_face_ids: np.ndarray
    constraint_vectors: np.ndarray
    constraint_error_raw: float
    constraint_error_output: float
    tangent_rms_face: float
    energy_dirichlet: float


def fem_vector_field_design(
    V: np.ndarray,
    F: np.ndarray,
    constraint_face_ids: np.ndarray,
    constraint_vectors: np.ndarray,
    lambda_tangent: float = 100.0,
    epsilon_mass: float = 1e-10,
    project_output: bool = True,
) -> FEMResult:
    nv = V.shape[0]
    nf = F.shape[0]
    if len(constraint_face_ids) == 0:
        raise ValueError("Need at least one vector constraint.")
    if (constraint_face_ids < 0).any() or (constraint_face_ids >= nf).any():
        bad = constraint_face_ids[(constraint_face_ids < 0) | (constraint_face_ids >= nf)]
        raise ValueError(f"Constraint face id out of range [0,{nf-1}]: {bad[:10]}")

    centers, fn, fa = face_geometry(V, F)
    vmass = vertex_area_lumped(nv, F, fa)
    vn = vertex_normals_area_weighted(nv, F, fn, fa)

    face_ids, targets = deduplicate_face_constraints(constraint_face_ids, constraint_vectors)
    # Match the neural pipeline: target vectors are projected to the corresponding face tangent plane.
    targets = project_to_tangent(targets, fn[face_ids], preserve_norm=False)

    L = cotangent_laplacian_positive(V, F)
    Q = build_vector_energy(
        L=L,
        vertex_mass=vmass,
        vertex_normals=vn,
        lambda_tangent=lambda_tangent,
        epsilon_mass=epsilon_mass,
    )
    Aeq, beq = build_face_center_constraints(F, face_ids, targets, nv)
    z = solve_kkt(Q, Aeq, beq)

    Uv_raw = np.column_stack([z[0:nv], z[nv:2 * nv], z[2 * nv:3 * nv]])
    Uf_raw = Uv_raw[F].mean(axis=1)

    if project_output:
        Uv = project_to_tangent(Uv_raw, vn, preserve_norm=False)
        Uf = project_to_tangent(Uf_raw, fn, preserve_norm=False)
    else:
        Uv = Uv_raw
        Uf = Uf_raw

    # Diagnostics.
    pred_raw = Uf_raw[face_ids]
    pred_out = Uf[face_ids]
    err_raw = float(np.sqrt(np.mean(np.sum((pred_raw - targets) ** 2, axis=1))))
    err_out = float(np.sqrt(np.mean(np.sum((pred_out - targets) ** 2, axis=1))))
    tangent_rms = float(np.sqrt(np.mean(np.sum((Uf * fn) ** 2, axis=1))))
    energy = float(
        sum(Uv_raw[:, c].T @ (L @ Uv_raw[:, c]) for c in range(3))
    )

    return FEMResult(
        V=V,
        F=F,
        U_vertex_raw=Uv_raw,
        U_vertex=Uv,
        U_face_raw=Uf_raw,
        U_face=Uf,
        face_centers=centers,
        face_normals=fn,
        constraint_face_ids=face_ids,
        constraint_vectors=targets.astype(np.float32),
        constraint_error_raw=err_raw,
        constraint_error_output=err_out,
        tangent_rms_face=tangent_rms,
        energy_dirichlet=energy,
    )


def save_result_npz(result: FEMResult, out_path: str):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    nf = result.F.shape[0]
    zeros = np.zeros(nf, dtype=np.float32)
    zero_vec = np.zeros_like(result.U_face, dtype=np.float32)

    # This schema is intentionally compatible with visualize_polyscope() in SurfaceVectorFieldDesign.py.
    np.savez(
        out_path,
        mesh_vertices=result.V.astype(np.float32),
        mesh_faces=result.F.astype(np.int64),
        theta=np.float32(0.0),
        constraint_face_ids=result.constraint_face_ids.astype(np.int64),
        constraint_vectors_original=result.constraint_vectors.astype(np.float32),
        constraint_vectors_rotated=result.constraint_vectors.astype(np.float32),
        face_centers=result.face_centers.astype(np.float32),
        face_normals=result.face_normals.astype(np.float32),
        face_normals_mesh=result.face_normals.astype(np.float32),
        phi=zeros,
        psi=zeros,
        u_pot=zero_vec,
        u_stream=result.U_face.astype(np.float32),
        u=result.U_face.astype(np.float32),
        u_raw=result.U_face_raw.astype(np.float32),
        u_vertex=result.U_vertex.astype(np.float32),
        u_vertex_raw=result.U_vertex_raw.astype(np.float32),
        constraint_rmse_raw=np.float32(result.constraint_error_raw),
        constraint_rmse_output=np.float32(result.constraint_error_output),
        tangent_rms_face=np.float32(result.tangent_rms_face),
        energy_dirichlet=np.float32(result.energy_dirichlet),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="FEM reference for surface vector field design: min int||grad u||^2 s.t. face-center vector constraints.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("mesh", help="Path to .obj/.ply mesh")
    p.add_argument(
        "--constraints",
        required=True,
        help="npz with arrays face_ids (K,) and vectors (K,3), same as make_vecfield_constraints.py",
    )
    p.add_argument("-o", "--out", default="fem_vecfield_ref.npz", help="Output npz path")
    p.add_argument(
        "--no-normalize",
        action="store_true",
        help="Use this iff train_vecfield.py also uses --no-normalize",
    )
    p.add_argument(
        "--lambda-tangent",
        type=float,
        default=100.0,
        help="Soft penalty weight for int(u.n)^2 dA. Set 0 to disable.",
    )
    p.add_argument(
        "--epsilon-mass",
        type=float,
        default=1e-10,
        help="Tiny mass regularizer to stabilize the KKT system.",
    )
    p.add_argument(
        "--no-project-output",
        action="store_true",
        help="Do not project the exported face/vertex vector field to tangent planes after solving.",
    )
    return p


def main():
    args = build_arg_parser().parse_args()
    V, F = load_mesh(args.mesh, normalize=not args.no_normalize)
    c = np.load(args.constraints)
    face_ids = np.asarray(c["face_ids"], dtype=np.int64)
    vectors = np.asarray(c["vectors"], dtype=np.float64)

    result = fem_vector_field_design(
        V=V,
        F=F,
        constraint_face_ids=face_ids,
        constraint_vectors=vectors,
        lambda_tangent=args.lambda_tangent,
        epsilon_mass=args.epsilon_mass,
        project_output=not args.no_project_output,
    )
    save_result_npz(result, args.out)

    print(f"Saved FEM reference -> {args.out}")
    print(f"  vertices={V.shape[0]} faces={F.shape[0]} constraints={len(result.constraint_face_ids)}")
    print(f"  Dirichlet energy     : {result.energy_dirichlet:.8e}")
    print(f"  constraint RMSE raw  : {result.constraint_error_raw:.8e}")
    print(f"  constraint RMSE out  : {result.constraint_error_output:.8e}")
    print(f"  face tangency RMS    : {result.tangent_rms_face:.8e}")
    print("Visualize:")
    print(f"  python SurfaceVectorFieldDesign.py --vis {args.out}")


if __name__ == "__main__":
    main()
