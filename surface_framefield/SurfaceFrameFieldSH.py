"""
SurfaceFrameFieldSH - theta-Field Frame Field via Phase Code
===========================================================

Pipeline:
    Step 0: Sparse-grid SDF oracle (fixed, precomputed)
    Step 1: Extract feature curves via libigl
    Step 2: Siren on ``(x, n) in R^6`` outputs scalar **theta**
    Step 3: Build deterministic reference frame ``(r1, r2)`` from oracle normal ``n`` (Rodrigues),
            or optionally use Hessian principal tangents as ``(r1, r2)`` when ``ref_mode='principal'``.
    Step 4: Rotate that tangent frame by **theta** -> ``(e1, e2, n)`` -> SH ``q``.

Losses:
    L_smooth : kNN Dirichlet on SH coefficients at random surface samples (weight constant in time).
               Optional ``dual_surface_smooth``: add face-center samples with Dirichlet energy on
               SH ``q`` along **vertex-ring** face pairs (faces sharing a vertex; see
               ``undirected_face_vertex_ring_edges``).
    L_feat   : |grad_r F_T(q, t_i) - 4t_i|^2 - weight lambda_feat follows piecewise ramp/plateau/cosine decay in s=t/T
    L_cpm    : |J_q . n|^2 (SH field constant along normal)
    L_amp    : softplus(tau^2 - |a_raw|^2) code anti-collapse

Key advantage: theta has 1 DOF -> normal-alignment and octahedral variety
are satisfied BY CONSTRUCTION. No need for normal-align / variety losses.
"""

import math
import os
import numpy as np
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import grad

from model_pytorch import Siren
from sh_pytorch import R3_to_sh4_zonal, grad_oct_polynomial_sh4


# One-shot stats for debugging cuSOLVER / NaN issues in compute_feature_direction_mincurv
_MINCURV_DEBUG_FIRST = True


def _tensor_finite_report(tag: str, t: torch.Tensor, max_show_bad: int = 8) -> bool:
    """Print NaN/Inf counts and extrema; return True if all finite."""
    with torch.no_grad():
        t = t.detach()
        n_elem = t.numel()
        n_nan = torch.isnan(t).sum().item()
        n_inf = torch.isinf(t).sum().item()
        finite = torch.isfinite(t)
        ok = bool(finite.all().item())
        print(f"    [{tag}] shape={tuple(t.shape)} dtype={t.dtype} device={t.device}")
        print(f"        nan={n_nan}  inf={n_inf}  finite_frac={finite.float().mean().item():.6f}")
        if ok and n_elem > 0:
            print(f"        abs: min={t.abs().min().item():.6g} max={t.abs().max().item():.6g} "
                  f"mean={t.abs().mean().item():.6g}")
        elif not ok:
            bad_batch = (~finite.reshape(t.shape[0], -1).all(dim=1)).nonzero(as_tuple=False).flatten()
            nb = min(max_show_bad, bad_batch.numel())
            if nb > 0:
                idxs = bad_batch[:nb].tolist()
                print(f"        first batch indices with non-finite: {idxs}")
        return ok


def lambda_feat_schedule_ratio(
    s: float,
    plateau_start: float = 0.2,
    plateau_end: float = 0.8,
) -> float:
    """
    Multiplier applied to lambda_feat as a function of training progress s = t/T in [0, 1].
    lambda_smooth is not scaled. Requires 0 < plateau_start < plateau_end < 1.

    - 0 <= s < plateau_start : linear ramp from 0.2x to 1.0x
    - plateau_start <= s < plateau_end : plateau at 1.0x
    - plateau_end <= s <= 1 : cosine decay from 1.0x down to 0.2x
    """
    ps = float(plateau_start)
    pe = float(plateau_end)
    if not (0.0 < ps < pe < 1.0):
        raise ValueError(
            "lambda_feat_schedule_ratio needs 0 < plateau_start < plateau_end < 1 "
            f"(got plateau_start={ps}, plateau_end={pe})")
    s = float(np.clip(s, 0.0, 1.0))
    if s < ps:
        return 0.2 + 0.8 * (s / ps)
    if s < pe:
        return 1.0
    span = 1.0 - pe
    return 0.2 + 0.8 * 0.5 * (1.0 + math.cos(math.pi * (s - pe) / span))


# ============================================================
# Debug check
# ============================================================

def _debug_check_q_vs_frame(num_tests=64):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    errs = []
    for _ in range(num_tests):
        A = torch.randn(3, 3, device=device)
        Q, _ = torch.linalg.qr(A)
        if torch.det(Q) < 0:
            Q[:, 2] = -Q[:, 2]
        e1, e2, e3 = Q[:, 0:1].T, Q[:, 1:2].T, Q[:, 2:3].T
        t = torch.randn(1, 3, device=device)
        t = t / (t.norm(dim=-1, keepdim=True) + 1e-8)
        d1 = (e1 * t).sum(-1, keepdim=True)
        d2 = (e2 * t).sum(-1, keepdim=True)
        d3 = (e3 * t).sum(-1, keepdim=True)
        grad_explicit = 4.0 * (d1**3 * e1 + d2**3 * e2 + d3**3 * e3)
        R = Q.T.unsqueeze(0)  # rows = frame dirs
        q = R3_to_sh4_zonal(R)
        q = q / (q.norm(dim=-1, keepdim=True) + 1e-8)
        grad_q = grad_oct_polynomial_sh4(t, q)
        errs.append((grad_q - grad_explicit).norm(dim=-1).mean().item())
    print(f"[debug] row-frame mean err = {sum(errs)/len(errs):.6e}  "
          f"max = {max(errs):.6e}")


# ============================================================
# Feature curve extraction (libigl)
# ============================================================

def extract_feature_curves(V, F, face_normals, angle_deg=30.0):
    import igl
    F64 = np.ascontiguousarray(F.astype(np.int64))
    uE, _, EF, _ = igl.edge_flaps(F64)
    cos_thresh = np.cos(np.deg2rad(angle_deg))
    n = face_normals.astype(np.float64)
    sharp_mask = np.zeros(len(uE), dtype=bool)
    for i in range(len(uE)):
        f0, f1 = EF[i]
        if f0 < 0 or f1 < 0:
            continue
        if (n[f0] * n[f1]).sum() < cos_thresh:
            sharp_mask[i] = True
    sharp_edges = uE[sharp_mask]
    if len(sharp_edges) == 0:
        print("  Warning: no feature edges, lowering threshold")
        cos_angles = np.array([(n[EF[i,0]]*n[EF[i,1]]).sum()
                               if EF[i,0]>=0 and EF[i,1]>=0 else 1.0
                               for i in range(len(uE))])
        k = min(100, len(uE)//10)
        idx = np.argsort(cos_angles)[:k]
        sharp_edges = uE[idx]
        sharp_mask[:] = False; sharp_mask[idx] = True
    Vf = V.astype(np.float64)
    mid = (Vf[sharp_edges[:,0]] + Vf[sharp_edges[:,1]]) / 2.0
    tan = Vf[sharp_edges[:,1]] - Vf[sharp_edges[:,0]]
    tan = tan / (np.linalg.norm(tan, axis=1, keepdims=True) + 1e-12)

    sharp_face_pairs = np.asarray([EF[i] for i in np.where(sharp_mask)[0]],
                                   dtype=np.int64)
    print(f"  Feature edges: {len(sharp_edges)} (threshold={angle_deg} deg)")
    return (mid.astype(np.float32), tan.astype(np.float32),
            np.where(sharp_mask)[0].astype(np.int64),
            sharp_face_pairs)


# ============================================================
# kNN graph
# ============================================================

def knn_graph_fast(pts, k=8, k_scale=None):
    """
    kNN directed edges (i -> j) with **local-scaling heat kernel** weights
    (Zelnik-Manor & Perona style):

        sigma_i = d(x_i, k_scale-th NN),   w_ij = exp( -||x_i-x_j||^2 / (sigma_i sigma_j + eps) ).

    ``k_scale`` defaults to ``k``. One ``topk(max(k,k_scale))`` pass on squared distances.
    """
    N = pts.shape[0]
    if N <= 1:
        z = torch.empty(0, 2, device=pts.device, dtype=torch.long)
        w = torch.empty(0, device=pts.device, dtype=pts.dtype)
        return z, w
    if k_scale is None:
        k_scale = k
    k = int(max(1, min(int(k), N - 1)))
    k_scale = int(max(1, min(int(k_scale), N - 1)))
    km = max(k, k_scale)

    d2 = torch.cdist(pts, pts, p=2).pow(2)
    d2.fill_diagonal_(float('inf'))
    dkm, ikm = d2.topk(km, dim=-1, largest=False)

    dk = dkm[:, :k]
    ik = ikm[:, :k]
    sigma = torch.sqrt(dkm[:, k_scale - 1].clamp(min=1e-12))

    src = torch.arange(N, device=pts.device).unsqueeze(1).expand(-1, k)
    edges = torch.stack([src.reshape(-1), ik.reshape(-1)], dim=1)

    dij2 = dk.reshape(-1)
    si = sigma[edges[:, 0]]
    sj = sigma[edges[:, 1]]
    eps = 1e-12
    w = torch.exp(-dij2 / (si * sj + eps))
    return edges, w

def shortest_rotation_apply(n_src, n_tgt, v):
    """
    Apply shortest rotation U: n_src -> n_tgt to vector(s) v, batched.
    All inputs (N, 3); returns (N, 3).

    Uses Rodrigues formula. Handles parallel and anti-parallel cases.
    n_src, n_tgt assumed unit; v can be any vector.
    """
    cos_a = (n_src * n_tgt).sum(-1, keepdim=True).clamp(-1.0, 1.0)
    axis_raw = torch.cross(n_src, n_tgt, dim=-1)
    # NOTE: use sqrt(|x|^2 + eps) instead of .norm() so backward is finite at
    # axis_raw ~ 0 (n_src ~ n_tgt, common on smooth surfaces). Plain `.norm()`
    # has d|x|/dx = x/|x| = 0/0 at zero, and torch.where-masking in forward
    # does NOT prevent NaN from leaking back through the masked branch
    # (0 * NaN = NaN in float), which eventually destroys the optimizer state.
    axis_norm_sq = (axis_raw * axis_raw).sum(-1, keepdim=True)
    axis_norm = (axis_norm_sq + 1e-24).sqrt()

    anti = (axis_norm < 1e-6) & (cos_a < 0.0)
    par  = (axis_norm < 1e-6) & (cos_a >= 0.0)
    use_dummy = anti | par

    # For anti-parallel, rotate pi around any axis perpendicular to n_src;
    # take any vector not collinear with n_src - pick world x or y.
    x_axis = torch.tensor([1.0, 0.0, 0.0], device=v.device, dtype=v.dtype).expand_as(v)
    y_axis = torch.tensor([0.0, 1.0, 0.0], device=v.device, dtype=v.dtype).expand_as(v)
    pick_x = (n_src[:, :1].abs() < 0.9)
    dummy_axis = torch.where(pick_x, x_axis, y_axis)
    dummy_axis = dummy_axis - (dummy_axis * n_src).sum(-1, keepdim=True) * n_src
    dummy_axis = F.normalize(dummy_axis, dim=-1, eps=1e-8)

    axis = torch.where(use_dummy, dummy_axis, axis_raw)
    axis_norm_safe = torch.where(use_dummy, torch.ones_like(axis_norm), axis_norm)

    cos_a_fixed = torch.where(anti, -torch.ones_like(cos_a), cos_a)
    cos_a_fixed = torch.where(par,  torch.ones_like(cos_a_fixed), cos_a_fixed)
    sin_a = torch.where(use_dummy,
                        torch.where(anti, torch.zeros_like(axis_norm),
                                          torch.zeros_like(axis_norm)),
                        axis_norm)

    k = axis / (axis_norm_safe + 1e-12)
    kxv = torch.cross(k, v, dim=-1)
    kdv = (k * v).sum(-1, keepdim=True)
    return v * cos_a_fixed + kxv * sin_a + k * kdv * (1.0 - cos_a_fixed)


def undirected_face_adjacency_edges(F: np.ndarray) -> np.ndarray:
    """Interior mesh edges -> rows (f0, f1) with f0 < f1 face indices sharing an edge."""
    from collections import defaultdict

    F = np.asarray(F, dtype=np.int64)
    e2f = defaultdict(list)
    for fi, tri in enumerate(F):
        for a, b in ((0, 1), (1, 2), (2, 0)):
            u, v = int(tri[a]), int(tri[b])
            key = (u, v) if u < v else (v, u)
            e2f[key].append(fi)
    pairs = [tuple(fl) for fl in e2f.values() if len(fl) == 2]
    if not pairs:
        return np.zeros((0, 2), dtype=np.int64)
    return np.array(pairs, dtype=np.int64)


def undirected_face_vertex_ring_edges(F: np.ndarray) -> np.ndarray:
    """
    Unique undirected face pairs ``(f0, f1)`` with ``f0 < f1`` that share **at least one**
    mesh vertex (the usual **one-ring** of faces around each triangle). Strictly contains
    ``undirected_face_adjacency_edges`` (edge-adjacent pairs are a subset).
    """
    from collections import defaultdict

    F = np.asarray(F, dtype=np.int64)
    v2f = defaultdict(list)
    for fi, tri in enumerate(F):
        for k in range(3):
            v2f[int(tri[k])].append(fi)
    pair_set = set()
    for flist in v2f.values():
        m = len(flist)
        if m < 2:
            continue
        for ii in range(m):
            for jj in range(ii + 1, m):
                a, b = int(flist[ii]), int(flist[jj])
                if a == b:
                    continue
                if a > b:
                    a, b = b, a
                pair_set.add((a, b))
    if not pair_set:
        return np.zeros((0, 2), dtype=np.int64)
    return np.array(sorted(pair_set), dtype=np.int64)


# ============================================================
# Hessian principal directions
# ============================================================

def compute_hessian_principal_dirs(oracle, pts):
    n = oracle.normal(pts).detach()
    H = oracle.hessian(pts).detach()
    n = n / (n.norm(dim=-1, keepdim=True) + 1e-8)
    P = torch.eye(3, device=pts.device, dtype=n.dtype).unsqueeze(0) - \
        torch.bmm(n.unsqueeze(2), n.unsqueeze(1))
    M = torch.bmm(P, torch.bmm(H, P))
    M = 0.5 * (M + M.transpose(1, 2))
    B = M.shape[0]
    ex = torch.zeros(B, 3, device=pts.device, dtype=n.dtype)
    ex[:, 0] = 1.0
    use_y = (ex * n).abs().sum(-1, keepdim=True) > 0.9
    ey = torch.zeros(B, 3, device=pts.device, dtype=n.dtype)
    ey[:, 1] = 1.0
    e = torch.where(use_y, ey, ex)
    u1 = e - (e * n).sum(-1, keepdim=True) * n
    u1 = u1 / (u1.norm(dim=-1, keepdim=True) + 1e-8)
    u2 = torch.cross(n, u1, dim=-1)
    u2 = u2 / (u2.norm(dim=-1, keepdim=True) + 1e-8)
    a1 = torch.bmm(M, u1.unsqueeze(2)).squeeze(2)
    a2 = torch.bmm(M, u2.unsqueeze(2)).squeeze(2)
    g00 = (u1 * a1).sum(-1, keepdim=True)
    g01 = (u1 * a2).sum(-1, keepdim=True)
    g11 = (u2 * a2).sum(-1, keepdim=True)
    G = torch.cat(
        (torch.cat((g00, g01), dim=-1).unsqueeze(-2),
         torch.cat((g01, g11), dim=-1).unsqueeze(-2)),
        dim=-2)
    # Chunked eigh: very large B (e.g. |F| with per_face_adj) can trigger cuSOLVER batched errors.
    _eigh_chunk = 8192
    if B <= _eigh_chunk:
        w2, q2 = torch.linalg.eigh(G)
    else:
        w2_l, q2_l = [], []
        for s in range(0, B, _eigh_chunk):
            sl = slice(s, s + _eigh_chunk)
            w, v = torch.linalg.eigh(G[sl])
            w2_l.append(w)
            q2_l.append(v)
        w2 = torch.cat(w2_l, dim=0)
        q2 = torch.cat(q2_l, dim=0)
    lam2 = w2[:, 0]
    lam1 = w2[:, 1]
    c10 = q2[:, 0, 1].unsqueeze(-1)
    c11 = q2[:, 1, 1].unsqueeze(-1)
    c20 = q2[:, 0, 0].unsqueeze(-1)
    c21 = q2[:, 1, 0].unsqueeze(-1)
    t1 = c10 * u1 + c11 * u2
    t2 = c20 * u1 + c21 * u2
    t1 = t1 / (t1.norm(dim=-1, keepdim=True) + 1e-8)
    t2 = t2 / (t2.norm(dim=-1, keepdim=True) + 1e-8)
    eps = 1e-8
    w_curv = (lam1.abs() - lam2.abs()).abs() / (lam1.abs() + lam2.abs() + eps)
    return t1, t2, w_curv


def _curv_weight_scale01_torch(s: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Min-max ``s`` to ``[0, 1]`` (per batch)."""
    lo, hi = s.min(), s.max()
    span = hi - lo
    if float(span.item()) < eps:
        return torch.ones_like(s)
    return ((s - lo) / (span + eps)).clamp(0.0, 1.0)


def _curv_weight_bottom_floor_torch(
    s: torch.Tensor, floor: float, bottom_q: float, eps: float = 1e-12
) -> torch.Tensor:
    """Lowest ``bottom_q`` quantile of ``s`` -> ``floor``; remainder linear in ``[floor, 1]``."""
    tau = torch.quantile(s.detach().float(), bottom_q).to(dtype=s.dtype, device=s.device)
    s_max = s.max()
    sm = s_max - tau
    hi = torch.where(
        sm < eps,
        torch.ones_like(s),
        floor + (1.0 - floor) * (s - tau) / (sm + eps))
    return torch.where(s <= tau, torch.full_like(s, floor), hi).clamp(float(floor), 1.0)


def compute_feature_direction_mincurv(
    oracle,
    pts,
    feat_align_weight: str = "k1_minus_k2",
    curv_weight_mode: Optional[str] = None,
    curv_weight_floor: float = 0.001,
    curv_bottom_quantile: float = 0.2,
):
    """
    Min-principal-curvature direction ``t_feat`` and curvature weight ``w_curv``.

    Same **tangent 2x2** construction as ``compute_hessian_principal_dirs``: restrict
    ``M = P H P`` to ``span{u1,u2} _|_ n`` and ``eigh(G)`` - not full 3x3 ``|lambda|`` ordering.

    Parameters
    ----------
    feat_align_weight : str
        ``k1_minus_k2`` - ``(kappa1 - kappa2)`` with kappa1>=kappa2 from ``G``.

    curv_weight_mode : None, ``"scale01"``, or ``"bottom_floor"``
        If set, **requires** ``feat_align_weight="abs_abs_diff"``; replaces raw
        ``|||kappa1|-|kappa2|||`` with weights in ``[0, 1]``: ``scale01`` = min-max over the batch;
        ``bottom_floor`` = lowest ``curv_bottom_quantile`` fraction set to ``curv_weight_floor``,
        then linear to ``1`` (same as ``debug/test_crossfield.py``).
    curv_weight_floor : float
        Used only for ``bottom_floor`` (default ``0.001``).
    curv_bottom_quantile : float
        Used only for ``bottom_floor`` (default ``0.2``).

    Prefer ``oracle.normal_jacobian``; else ``oracle.hessian``.

    Returns
    -------
    t_feat : (N, 3) unit - **smaller** eigenvalue of ``G`` (kappa2 direction).
    w_curv : (N,) detached weight from kappa1, kappa2.
    """
    global _MINCURV_DEBUG_FIRST
    verbose_once = _MINCURV_DEBUG_FIRST

    if verbose_once:
        print("[compute_feature_direction_mincurv] first-call diagnostics:")

    n = oracle.normal(pts).detach()
    n = n / (n.norm(dim=-1, keepdim=True) + 1e-8)

    if hasattr(oracle, "normal_jacobian"):
        Hc = oracle.normal_jacobian(pts).detach()   # gradient-corrected Hessian proxy
        _hc_src = "normal_jacobian"
    else:
        Hc = oracle.hessian(pts).detach()
        _hc_src = "hessian"

    P = torch.eye(3, device=pts.device).unsqueeze(0) - \
        torch.bmm(n.unsqueeze(2), n.unsqueeze(1))

    M = torch.bmm(P, torch.bmm(Hc, P))
    M = 0.5 * (M + M.transpose(1, 2))

    if verbose_once:
        _tensor_finite_report("pts (input)", pts.detach())
        _tensor_finite_report("n (oracle.normal)", n)
        _tensor_finite_report(f"Hc ({_hc_src})", Hc)
        _tensor_finite_report("M (projected symm)", M)
        _MINCURV_DEBUG_FIRST = False

    ok = (
        torch.isfinite(n).all()
        and torch.isfinite(Hc).all()
        and torch.isfinite(M).all()
    )
    if not ok:
        print("[compute_feature_direction_mincurv] NON-FINITE before eigh(G):")
        _tensor_finite_report("pts (input)", pts.detach())
        _tensor_finite_report("n (oracle.normal)", n)
        _tensor_finite_report(f"Hc ({_hc_src})", Hc)
        _tensor_finite_report("M (projected symm)", M)
        raise RuntimeError(
            "compute_feature_direction_mincurv: NaN/Inf in n, Hc, or M (see batch indices above)."
        )

    B = M.shape[0]
    ex = torch.zeros(B, 3, device=pts.device, dtype=n.dtype)
    ex[:, 0] = 1.0
    use_y = (ex * n).abs().sum(-1, keepdim=True) > 0.9
    ey = torch.zeros(B, 3, device=pts.device, dtype=n.dtype)
    ey[:, 1] = 1.0
    e = torch.where(use_y, ey, ex)
    u1 = e - (e * n).sum(-1, keepdim=True) * n
    u1 = u1 / (u1.norm(dim=-1, keepdim=True) + 1e-8)
    u2 = torch.cross(n, u1, dim=-1)
    u2 = u2 / (u2.norm(dim=-1, keepdim=True) + 1e-8)
    a1 = torch.bmm(M, u1.unsqueeze(2)).squeeze(2)
    a2 = torch.bmm(M, u2.unsqueeze(2)).squeeze(2)
    g00 = (u1 * a1).sum(-1, keepdim=True)
    g01 = (u1 * a2).sum(-1, keepdim=True)
    g11 = (u2 * a2).sum(-1, keepdim=True)
    G = torch.cat(
        (torch.cat((g00, g01), dim=-1).unsqueeze(-2),
         torch.cat((g01, g11), dim=-1).unsqueeze(-2)),
        dim=-2)

    _eigh_chunk = 8192
    if B <= _eigh_chunk:
        w2, q2 = torch.linalg.eigh(G)
    else:
        w2_l, q2_l = [], []
        for s in range(0, B, _eigh_chunk):
            sl = slice(s, s + _eigh_chunk)
            w, v = torch.linalg.eigh(G[sl])
            w2_l.append(w)
            q2_l.append(v)
        w2 = torch.cat(w2_l, dim=0)
        q2 = torch.cat(q2_l, dim=0)

    k1 = w2[:, 1]
    k2 = w2[:, 0]

    c20 = q2[:, 0, 0].unsqueeze(-1)
    c21 = q2[:, 1, 0].unsqueeze(-1)
    t_feat = c20 * u1 + c21 * u2
    t_feat = t_feat / (t_feat.norm(dim=-1, keepdim=True) + 1e-8)

    if feat_align_weight == "k1_minus_k2":
        w_curv = (k1 - k2)
    elif feat_align_weight == "abs_abs_diff":
        w_curv = (k1.abs() - k2.abs()).abs()
    else:
        raise ValueError(
            "feat_align_weight must be 'k1_minus_k2' or 'abs_abs_diff', "
            f"got {feat_align_weight!r}"
        )

    if curv_weight_mode is not None:
        if feat_align_weight != "abs_abs_diff":
            raise ValueError(
                "curv_weight_mode requires feat_align_weight='abs_abs_diff' "
                f"(got {feat_align_weight!r})")
        if curv_weight_mode not in ("scale01", "bottom_floor"):
            raise ValueError(
                "curv_weight_mode must be None, 'scale01', or 'bottom_floor' "
                f"(got {curv_weight_mode!r})")
        if curv_weight_mode == "bottom_floor":
            if not (0.0 < curv_bottom_quantile < 1.0):
                raise ValueError(
                    "curv_bottom_quantile must be in (0, 1) "
                    f"(got {curv_bottom_quantile})")
            if not (0.0 < curv_weight_floor < 1.0):
                raise ValueError(
                    "curv_weight_floor must be in (0, 1) "
                    f"(got {curv_weight_floor})")
        s = (k1.abs() - k2.abs()).abs()
        if curv_weight_mode == "scale01":
            w_curv = _curv_weight_scale01_torch(s)
        else:
            w_curv = _curv_weight_bottom_floor_torch(
                s, floor=curv_weight_floor, bottom_q=curv_bottom_quantile)
        w_curv = w_curv.detach()

    return t_feat, w_curv

# ============================================================
# Reference frame from normal
# ============================================================

def build_reference_frame_old(n):
    """
    Deterministic orthonormal tangent frame (r1, r2) from unit normal n.

    Uses "project a seed axis" trick:
        seed = (0,0,1) if |n_z| < 0.9 else (1,0,0)
        r1 = normalize(seed - (seed.n)n)
        r2 = n x r1
    """
    seed = torch.zeros_like(n)
    mask = n[:, 2].abs() < 0.9
    seed[mask, 2] = 1.0
    seed[~mask, 0] = 1.0

    r1 = seed - (seed * n).sum(-1, keepdim=True) * n
    r1 = r1 / (r1.norm(dim=-1, keepdim=True) + 1e-8)
    r2 = torch.cross(n, r1, dim=-1)
    return r1, r2

def build_reference_frame(n):
    """
    Deterministic orthonormal tangent frame from unit normal n
    via minimal rotation from canonical z-hat to n.

    R_n maps z-hat -> n, then r1 = R_n x-hat, r2 = R_n y-hat.
    Uses Rodrigues formula. Special-cases n ~ -z-hat.
    No in-place ops on tensors depending on n, so safer for higher-order autograd.
    """
    n = F.normalize(n, dim=-1, eps=1e-8)

    z = torch.tensor([0.0, 0.0, 1.0], device=n.device, dtype=n.dtype).expand_as(n)
    x_hat = torch.tensor([1.0, 0.0, 0.0], device=n.device, dtype=n.dtype).expand_as(n)
    y_hat = torch.tensor([0.0, 1.0, 0.0], device=n.device, dtype=n.dtype).expand_as(n)

    cos_a = (n * z).sum(-1, keepdim=True).clamp(-1.0, 1.0)   # (N,1)
    axis_raw = torch.cross(z, n, dim=-1)                     # (N,3)
    axis_norm_raw = axis_raw.norm(dim=-1, keepdim=True)      # (N,1)

    anti = (axis_norm_raw < 1e-6) & (cos_a < 0.0)            # n ~ -z
    par  = (axis_norm_raw < 1e-6) & (cos_a >= 0.0)           # n ~ +z

    x_axis = torch.tensor([1.0, 0.0, 0.0], device=n.device, dtype=n.dtype).expand_as(n)

    # If anti-parallel or parallel, use x-axis as a dummy rotation axis
    use_x_axis = anti | par
    axis = torch.where(use_x_axis, x_axis, axis_raw)

    # avoid division by zero
    axis_norm = torch.where(use_x_axis, torch.ones_like(axis_norm_raw), axis_norm_raw)

    # for anti-parallel, angle = pi => cos=-1, sin=0
    # for parallel, angle = 0   => cos=+1, sin=0
    cos_a_fixed = torch.where(anti, -torch.ones_like(cos_a), cos_a)
    cos_a_fixed = torch.where(par,  torch.ones_like(cos_a_fixed), cos_a_fixed)

    sin_a = torch.where(use_x_axis, torch.zeros_like(axis_norm), axis_norm)

    k = axis / (axis_norm + 1e-12)   # unit rotation axis

    def rodrigues_apply(v):
        kxv = torch.cross(k, v, dim=-1)
        kdv = (k * v).sum(-1, keepdim=True)
        return v * cos_a_fixed + kxv * sin_a + k * kdv * (1.0 - cos_a_fixed)

    r1 = rodrigues_apply(x_hat)
    r2 = rodrigues_apply(y_hat)

    # re-orthonormalize for numerical safety
    r1 = r1 - (r1 * n).sum(-1, keepdim=True) * n
    r1 = F.normalize(r1, dim=-1, eps=1e-8)

    r2 = r2 - (r2 * n).sum(-1, keepdim=True) * n
    r2 = r2 - (r2 * r1).sum(-1, keepdim=True) * r1
    r2 = F.normalize(r2, dim=-1, eps=1e-8)

    return r1, r2


def build_reference_frame_axis_swap(n):
    """
    Axis-swap reference frame (no fixed global pole): pick the axis with the smallest
    ``|n_i|``, zero that component, and swap/negate the other two to get ``u _|_ n``;
    then ``v = n x u``. Direct torch port of the numpy reference:

        for each row:
            i = argmin(|n_x|, |n_y|, |n_z|)     (the "skip" axis)
            (j, k) = the other two indices in ascending order
            u_j =  n_k,  u_k = -n_j,  u_i = 0
            v   =  n x u

    Avoids the Rodrigues "-z_hat" singularity; discontinuities only on the 3 planes
    ``|n_i| = |n_j|`` (zero-measure), which is acceptable for a gauge.
    """
    n = F.normalize(n, dim=-1, eps=1e-8)
    nx = n[..., 0]
    ny = n[..., 1]
    nz = n[..., 2]
    zeros = torch.zeros_like(nx)

    # three candidate u's, one per "skip" axis choice
    u0 = torch.stack([zeros,    nz,  -ny], dim=-1)  # skip x  -> (j,k) = (1,2)
    u1 = torch.stack([   nz, zeros,  -nx], dim=-1)  # skip y  -> (j,k) = (0,2)
    u2 = torch.stack([   ny,  -nx, zeros], dim=-1)  # skip z  -> (j,k) = (0,1)

    min_ind = n.abs().argmin(dim=-1)
    m0 = (min_ind == 0).unsqueeze(-1)
    m1 = (min_ind == 1).unsqueeze(-1)
    u = torch.where(m0, u0, torch.where(m1, u1, u2))
    v = torch.cross(n, u, dim=-1)

    u = F.normalize(u, dim=-1, eps=1e-8)
    v = F.normalize(v, dim=-1, eps=1e-8)
    return u, v


# ============================================================
# Main class
# ============================================================

class SurfaceFrameFieldSH:
    """
    theta-field frame field learner.

    Network on ``(x, n)`` outputs theta; rotate a tangent reference ``(r1, r2)`` (Rodrigues from ``n``,
    or principal directions if ``ref_mode='principal'``) to get ``(e1, e2, n)`` -> SH ``q``.

    Normal-alignment and octahedral variety satisfied by construction.
    """

    def __init__(self, sdf_oracle, hidden=128, n_layers=4,
                 omega_0=30.0, lr=1e-3, device='cpu'):
        self.device = torch.device(device)
        self.oracle = sdf_oracle
        self.hidden = hidden
        self.n_layers = n_layers
        self.lr = lr

        # Siren: (x, n) in R^6
        self.model = Siren(
            in_features=6,
            hidden_features=hidden,
            hidden_layers=n_layers,
            out_features=1,          # directly learn theta
            outermost_linear=True,
            first_omega_0=omega_0,
            hidden_omega_0=omega_0,
        ).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"SurfaceFrameFieldSH (angle-field): Siren {n_params} params, "
            f"hidden={hidden}, layers={n_layers}, omega_0={omega_0}")

    # ================================================================
    # Forward: theta -> frame -> SH
    # ================================================================

    def _forward_from_x_and_n(self, x_leaf, n_fixed, r_frame=None):
        """
        Low-level forward with:
        - x_leaf: leaf tensor requiring grad
        - n_fixed: normal in third column of R (detached label or n_leaf for Jacobian)
        - r_frame: optional ``(r1, r2)`` tangent reference; if ``None``, use Rodrigues frame from ``n_fixed``.

        IMPORTANT:
        Do NOT detach / recreate x_leaf here.
        This is used by CPM so that grad(q, x_leaf) is valid.
        """
        xn = torch.cat([x_leaf, n_fixed], dim=1)  # (N, 6)
        theta = self.model(xn).squeeze(-1)  # (N,)

        if r_frame is None:
            r1, r2 = build_reference_frame(n_fixed)
        else:
            r1, r2 = r_frame

        cos_t = torch.cos(theta).unsqueeze(-1)
        sin_t = torch.sin(theta).unsqueeze(-1)

        e1 = cos_t * r1 + sin_t * r2
        e2 = -sin_t * r1 + cos_t * r2

        R = torch.stack([e1, e2, n_fixed], dim=1)             # rows = frame dirs
        q_raw = R3_to_sh4_zonal(R)
        q = q_raw / (q_raw.norm(dim=-1, keepdim=True) + 1e-8)

        return {
            'theta': theta,
            'e1': e1, 'e2': e2, 'n': n_fixed,
            'q': q, 'q_raw': q_raw,
            'pts_leaf': x_leaf,
        }

    def forward_surface(self, pts, create_graph=True, ref_mode='normal'):
        """
        Surface forward:
        - ``ref_mode='normal'``: Rodrigues minimal rotation z_hat -> ``n`` for ``(r1, r2)``.
        - ``ref_mode='principal'``: ``(r1, r2)`` from ``compute_hessian_principal_dirs`` (detached).
        Normal label is ``oracle.normal(x)``, detached.
        """
        pts_leaf = pts.detach().requires_grad_(True)
        n = self.oracle.normal(pts_leaf).detach()
        r_frame = None
        if ref_mode == 'principal':
            k1, k2, _ = compute_hessian_principal_dirs(self.oracle, pts_leaf)
            r_frame = (k1.detach(), k2.detach())
        elif ref_mode == 'axis_swap':
            r1, r2 = build_reference_frame_axis_swap(n)
            r_frame = (r1.detach(), r2.detach())
        return self._forward_from_x_and_n(pts_leaf, n, r_frame=r_frame)

    # ================================================================
    # Loss: SH Dirichlet smoothness (kNN)
    # ================================================================

    def loss_smooth_knn(self, q, pts, knn_k=12, p=2.0, knn_k_scale=None):
        """kNN Dirichlet with local-scaling heat kernel weights on ``pts`` (see ``knn_graph_fast``)."""
        with torch.no_grad():
            edges, weights = knn_graph_fast(
                pts.detach(), k=knn_k, k_scale=knn_k_scale)
        if edges.numel() == 0:
            return (q * 0).sum()
        diff = q[edges[:, 0]] - q[edges[:, 1]]
        dist = (diff * diff).sum(-1)
        wsum = weights.sum().clamp(min=1e-12)
        if p == 2.0:
            return (weights * dist).sum() / wsum
        return (weights * dist.pow(p / 2.0)).sum() / wsum

    def loss_smooth_knn_transport(self, e1, e2, n, pts, knn_k=12, p=2.0,
                                  knn_k_scale=None, symmetric=True):
        """
        Transport-aware kNN Dirichlet on SH q.

        For each directed edge (i -> j):
          1) shortest rotation U_{j -> i} that maps n_j to n_i (n_i detached as anchor)
          2) transport j's frame: e1_j' = U.e1_j, e2_j' = U.e2_j
          3) build R = (e1_j', e2_j', n_i), lift to SH q_{j -> i}
          4) compare against q_i (recomputed locally from (e1_i, e2_i, n_i))

        (Previously ``symmetric=True`` also averaged the (j->i) direction; training now uses
        only the directed ``j->i`` term ``dist_ij`` for minimal cost / simpler graph signal.)

        e1, e2, n: (N, 3) on the network graph (gradients flow through e1, e2).
                   n is used only as orientation anchor; we detach it for the rotation.
        """
        _ = symmetric  # API compat; symmetric (i<->j) average disabled - use ``dist_ij`` only.
        with torch.no_grad():
            edges, weights = knn_graph_fast(pts.detach(), k=knn_k, k_scale=knn_k_scale)
        if edges.numel() == 0:
            return (e1 * 0).sum()

        i = edges[:, 0]
        j = edges[:, 1]

        n_i_anchor = n[i].detach()
        n_j_anchor = n[j].detach()

        # transport j -> i: rotate (e1_j, e2_j) by U taking n_j -> n_i
        e1_j_t = shortest_rotation_apply(n_j_anchor, n_i_anchor, e1[j])
        e2_j_t = shortest_rotation_apply(n_j_anchor, n_i_anchor, e2[j])
        R_ji = torch.stack([e1_j_t, e2_j_t, n_i_anchor], dim=1)
        q_ji = R3_to_sh4_zonal(R_ji)
        q_ji = q_ji / (q_ji.norm(dim=-1, keepdim=True) + 1e-8)

        # q_i in same anchor (just (e1_i, e2_i, n_i_anchor))
        R_i = torch.stack([e1[i], e2[i], n_i_anchor], dim=1)
        q_i = R3_to_sh4_zonal(R_i)
        q_i = q_i / (q_i.norm(dim=-1, keepdim=True) + 1e-8)

        diff_ij = q_i - q_ji
        dist_ij = (diff_ij * diff_ij).sum(-1)

        # if symmetric:
        #     e1_i_t = shortest_rotation_apply(n_i_anchor, n_j_anchor, e1[i])
        #     e2_i_t = shortest_rotation_apply(n_i_anchor, n_j_anchor, e2[i])
        #     R_ij = torch.stack([e1_i_t, e2_i_t, n_j_anchor], dim=1)
        #     q_ij = R3_to_sh4_zonal(R_ij)
        #     q_ij = q_ij / (q_ij.norm(dim=-1, keepdim=True) + 1e-8)
        #     R_j = torch.stack([e1[j], e2[j], n_j_anchor], dim=1)
        #     q_j = R3_to_sh4_zonal(R_j)
        #     q_j = q_j / (q_j.norm(dim=-1, keepdim=True) + 1e-8)
        #     diff_ji = q_j - q_ij
        #     dist_ji = (diff_ji * diff_ji).sum(-1)
        #     dist = 0.5 * (dist_ij + dist_ji)
        # else:
        #     dist = dist_ij
        dist = dist_ij

        wsum = weights.sum().clamp(min=1e-12)
        if p == 2.0:
            return (weights * dist).sum() / wsum
        return (weights * dist.pow(p / 2.0)).sum() / wsum

    @staticmethod
    def loss_smooth_face_adjacency(q: torch.Tensor, face_edge_idx: torch.Tensor, p: float = 2.0):
        r"""
        Dirichlet energy on SH ``q`` over a **symmetric face neighbor graph** given by
        ``face_edge_idx`` (undirected rows ``(i,j)``). Training uses **vertex one-ring**:
        faces sharing a mesh vertex (see ``undirected_face_vertex_ring_edges``), not only
        edge-adjacent faces.

        For each face ``f`` let ``deg(f)`` be the number of graph edges incident to ``f``.
        Accumulate ``d_{ij} = \|q_i-q_j\|_2^p`` (``p=2`` -> squared L2) on each edge to both
        endpoints, then **per-face mean** ``\bar L_f = (\sum_{j\sim f} d_{fj}) / \deg(f)``,
        and return ``(1/|F|) \sum_f \bar L_f``.
        """
        if face_edge_idx.numel() == 0:
            return torch.tensor(0.0, device=q.device, dtype=q.dtype)
        i0 = face_edge_idx[:, 0]
        i1 = face_edge_idx[:, 1]
        diff = q[i0] - q[i1]
        dist = (diff * diff).sum(-1)
        if p != 2.0:
            dist = dist.pow(p / 2.0)
        nf = int(q.shape[0])
        acc = torch.zeros(nf, device=q.device, dtype=dist.dtype)
        deg = torch.zeros(nf, device=q.device, dtype=dist.dtype)
        acc.index_add_(0, i0, dist)
        acc.index_add_(0, i1, dist)
        one = torch.ones(dist.shape[0], device=q.device, dtype=deg.dtype)
        deg.index_add_(0, i0, one)
        deg.index_add_(0, i1, one)
        per_face = acc / deg.clamp(min=1.0)
        return per_face.mean()

    @staticmethod
    def loss_smooth_face_adjacency_transport(
        e1: torch.Tensor,
        e2: torch.Tensor,
        n: torch.Tensor,
        face_edge_idx: torch.Tensor,
        p: float = 2.0,
    ):
        r"""
        Transport-aware counterpart of :meth:`loss_smooth_face_adjacency`.

        Same undirected face graph (``face_edge_idx``, rows ``(i, j)``) and same per-face
        deg-normalized aggregation. Each edge contributes the **directed** transported SH
        Dirichlet (rotate neighbor ``j`` into anchor ``n_i``, compare to ``q_i``; normals
        detached as rotation anchors only):

            d_{ij} = ||q_i - q_{j->i}||^2

        For ``p != 2``, ``d_{ij}`` is raised to ``p/2`` (same convention as the original).
        """
        if face_edge_idx.numel() == 0:
            return torch.tensor(0.0, device=e1.device, dtype=e1.dtype)
        i0 = face_edge_idx[:, 0]
        i1 = face_edge_idx[:, 1]

        n_i = n[i0].detach()
        n_j = n[i1].detach()

        e1_j_t = shortest_rotation_apply(n_j, n_i, e1[i1])
        e2_j_t = shortest_rotation_apply(n_j, n_i, e2[i1])
        R_ji = torch.stack([e1_j_t, e2_j_t, n_i], dim=1)
        q_ji = R3_to_sh4_zonal(R_ji)
        q_ji = q_ji / (q_ji.norm(dim=-1, keepdim=True) + 1e-8)
        R_i = torch.stack([e1[i0], e2[i0], n_i], dim=1)
        q_i = R3_to_sh4_zonal(R_i)
        q_i = q_i / (q_i.norm(dim=-1, keepdim=True) + 1e-8)
        dist_ij = ((q_i - q_ji) ** 2).sum(-1)

        # e1_i_t = shortest_rotation_apply(n_i, n_j, e1[i0])
        # e2_i_t = shortest_rotation_apply(n_i, n_j, e2[i0])
        # R_ij = torch.stack([e1_i_t, e2_i_t, n_j], dim=1)
        # q_ij = R3_to_sh4_zonal(R_ij)
        # q_ij = q_ij / (q_ij.norm(dim=-1, keepdim=True) + 1e-8)
        # R_j = torch.stack([e1[i1], e2[i1], n_j], dim=1)
        # q_j = R3_to_sh4_zonal(R_j)
        # q_j = q_j / (q_j.norm(dim=-1, keepdim=True) + 1e-8)
        # dist_ji = ((q_j - q_ij) ** 2).sum(-1)
        # dist = 0.5 * (dist_ij + dist_ji)
        dist = dist_ij
        if p != 2.0:
            dist = dist.pow(p / 2.0)

        nf = int(e1.shape[0])
        acc = torch.zeros(nf, device=e1.device, dtype=dist.dtype)
        deg = torch.zeros(nf, device=e1.device, dtype=dist.dtype)
        acc.index_add_(0, i0, dist)
        acc.index_add_(0, i1, dist)
        one = torch.ones(dist.shape[0], device=e1.device, dtype=deg.dtype)
        deg.index_add_(0, i0, one)
        deg.index_add_(0, i1, one)
        per_face = acc / deg.clamp(min=1.0)
        return per_face.mean()

    # ================================================================
    # Loss: Feature alignment (Hessian principal dirs)
    # ================================================================

    @staticmethod
    def loss_feat(q, t_feat, w_curv=None):
        """
        Single-direction feature alignment:
            |grad_r F_T(q, t_feat) - 4 t_feat|^2
        where t_feat is the selected principal direction.
        """
        q = q / (q.norm(dim=-1, keepdim=True) + 1e-8)

        t_feat = t_feat / (t_feat.norm(dim=-1, keepdim=True) + 1e-8)
        gF = grad_oct_polynomial_sh4(t_feat, q)
        per_pt = ((gF - 4.0 * t_feat) ** 2).sum(-1)

        if w_curv is not None:
            return (w_curv * per_pt).mean()
        return per_pt.mean()

    @staticmethod
    def loss_feat_edge_transport(e1_f0, e2_f0, n_f0,
                                 e1_f1, e2_f1, n_f1,
                                 n_mid, k1_mid):
        """
        Feature-edge alignment via transport.

        Each side's frame (e1, e2, n_side) is minimally rotated so that
        n_side -> n_mid. The transported frame is lifted to SH q, and we
        penalize the octahedral-alignment residual against the mid-edge
        min-|kappa| principal direction k1_mid:

            |grad_{k1} F_T(q, k1) - 4.k1|^2

        n_f0, n_f1, n_mid, k1_mid are detached supervisory signals.
        Gradients flow through e1_*, e2_*.
        """
        def _side_residual(e1, e2, n_side):
            e1_t = shortest_rotation_apply(n_side, n_mid, e1)
            e2_t = shortest_rotation_apply(n_side, n_mid, e2)
            R = torch.stack([e1_t, e2_t, n_mid], dim=1)
            q = R3_to_sh4_zonal(R)
            q = q / (q.norm(dim=-1, keepdim=True) + 1e-8)
            gF = grad_oct_polynomial_sh4(k1_mid, q)
            res = gF - 4.0 * k1_mid
            return (res * res).sum(-1).mean()

        return 0.5 * (_side_residual(e1_f0, e2_f0, n_f0)
                      + _side_residual(e1_f1, e2_f1, n_f1))

    @staticmethod
    def loss_feat_edge_transport_per_edge(
        e1_f0, e2_f0, n_f0,
        e1_f1, e2_f1, n_f1,
        n_mid, k1_mid,
    ):
        """Same as ``loss_feat_edge_transport`` but one scalar per edge row (no final mean)."""
        def _side_sq(e1, e2, n_side):
            e1_t = shortest_rotation_apply(n_side, n_mid, e1)
            e2_t = shortest_rotation_apply(n_side, n_mid, e2)
            R = torch.stack([e1_t, e2_t, n_mid], dim=1)
            q = R3_to_sh4_zonal(R)
            q = q / (q.norm(dim=-1, keepdim=True) + 1e-8)
            gF = grad_oct_polynomial_sh4(k1_mid, q)
            res = gF - 4.0 * k1_mid
            return (res * res).sum(-1)

        return 0.5 * (
            _side_sq(e1_f0, e2_f0, n_f0) + _side_sq(e1_f1, e2_f1, n_f1))

    # ================================================================
    # Loss: CPM on q  (|J_q . n|^2)
    # ================================================================

    def loss_cpm_SH(self, nb_pts, ref_mode='normal'):
        """
        CPM on q:
            || d_x q(x, n_cp(x)) . n_cp(x) ||^2

        Here:
        - x is the differentiation variable
        - n_cp(x) is treated as a FIXED normal label
        - this matches the closest-point pullback semantics
        """
        x_leaf = nb_pts.detach().requires_grad_(True)

        cp = self.oracle.closest_point(x_leaf).detach()
        n_cp = self.oracle.normal(cp).detach()

        r_frame = None
        if ref_mode == 'principal':
            k1, k2, _ = compute_hessian_principal_dirs(self.oracle, x_leaf)
            r_frame = (k1.detach(), k2.detach())
        elif ref_mode == 'axis_swap':
            r1, r2 = build_reference_frame_axis_swap(n_cp)
            r_frame = (r1.detach(), r2.detach())

        out = self._forward_from_x_and_n(x_leaf, n_cp, r_frame=r_frame)
        q = out['q']   # (N, 9)

        Jq_n = torch.zeros_like(q)
        for k in range(q.shape[1]):
            dqk_dx = grad(
                q[:, k].sum(),
                x_leaf,
                create_graph=True,
                retain_graph=True
            )[0]   # (N, 3)
            Jq_n[:, k] = (dqk_dx * n_cp).sum(-1)

        return (Jq_n ** 2).sum(-1).mean()

    def loss_cpm(self, nb_pts, ref_mode='normal'):
        """
        CPM on the scalar theta field:
            || grad_x theta(x, n_cp(x)) . n_cp(x) ||^2

        Forces theta to be constant along the normal fiber of the closest-point
        map - i.e., theta is an intrinsic surface scalar after extension.

        Here:
        - x is the differentiation variable (narrow-band points)
        - n_cp(x) is the oracle normal at the closest surface point, detached
        - network input is (x, n_cp(x)); ref_mode only affects forward_frame
          semantics elsewhere - theta itself comes straight from the network.
        """
        x_leaf = nb_pts.detach().requires_grad_(True)

        cp = self.oracle.closest_point(x_leaf).detach()
        n_cp = self.oracle.normal(cp).detach()

        xn = torch.cat([x_leaf, n_cp], dim=1)       # (N, 6)
        theta = self.model(xn).squeeze(-1)           # (N,)

        dtheta_dx = grad(
            theta.sum(), x_leaf,
            create_graph=True, retain_graph=True,
        )[0]                                         # (N, 3)

        return ((dtheta_dx * n_cp).sum(-1) ** 2).mean()

    def jacobian_q_total(self, pts, ref_mode='normal'):
        """
        Total Jacobian on the surface:

            D_x q(x, n(x)) = d_x q + d_n q . J_n
        Here:
          - x is the surface point
          - n(x) is the oracle normal at x
          - J_n = normal_jacobian(x)

        Returns
        -------
        out        : dict from _forward_from_x_and_n
        Jq_total   : (N, 9, 3) total Jacobian wrt x
        n_oracle   : (N, 3) surface normal
        Jq_x       : (N, 9, 3) explicit d_x q
        Jq_nx      : (N, 9, 3) explicit d_n q . J_n
        """
        x_leaf = pts.detach().requires_grad_(True)

        n_oracle = self.oracle.normal(x_leaf).detach()
        if not hasattr(self.oracle, "normal_jacobian"):
            raise AttributeError(
                "oracle.normal_jacobian is required for total surface derivative."
            )
        J_n = self.oracle.normal_jacobian(x_leaf).detach()   # (N, 3, 3)

        # n is treated as an independent leaf for partial derivative extraction
        n_leaf = n_oracle.clone().requires_grad_(True)
        r_frame = None
        if ref_mode == 'principal':
            k1, k2, _ = compute_hessian_principal_dirs(self.oracle, x_leaf)
            r_frame = (k1.detach(), k2.detach())
        elif ref_mode == 'axis_swap':
            # Use n_oracle (detached) so the piecewise argmin gauge does NOT
            # propagate gradients into n_leaf. Discontinuities in (r1, r2)
            # across |n_i|=|n_j| surfaces are NOT octahedral symmetries
            # (~60 deg tangent-plane rotations), so any gradient through them
            # blows up quickly (unlike 'normal' where Rodrigues is smooth).
            r1, r2 = build_reference_frame_axis_swap(n_oracle)
            r_frame = (r1.detach(), r2.detach())

        out = self._forward_from_x_and_n(x_leaf, n_leaf, r_frame=r_frame)
        q = out['q']   # (N, 9)

        Jq_x = torch.zeros(q.shape[0], q.shape[1], 3,
                        device=q.device, dtype=q.dtype)
        Jq_nx = torch.zeros_like(Jq_x)
        Jq_total = torch.zeros_like(Jq_x)

        for k in range(q.shape[1]):
            dqk_dx = grad(
                q[:, k].sum(),
                x_leaf,
                create_graph=True,
                retain_graph=True
            )[0]   # (N, 3)

            dqk_dn = grad(
                q[:, k].sum(),
                n_leaf,
                create_graph=True,
                retain_graph=True
            )[0]   # (N, 3)

            # d_{n(x)} q = d_n q . J_n
            dqk_nx = torch.bmm(dqk_dn.unsqueeze(1), J_n).squeeze(1)   # (N, 3)

            Jq_x[:, k, :] = dqk_dx
            Jq_nx[:, k, :] = dqk_nx
            Jq_total[:, k, :] = dqk_dx + dqk_nx

            # # chain rule: dq/dx = dq/dx + dq/dn . J_n
            # corr = torch.bmm(dqk_dn.unsqueeze(1), J_n).squeeze(1)   # (N, 3)
            # Jq[:, k, :] = dqk_dx + corr

        return out, Jq_total, n_oracle, Jq_x, Jq_nx


    def loss_smooth_analytic(self, surf_pts, p=2.0, ref_mode='normal'):
        """
        Analytic surface smoothness:

            ||grad_M q||^p,   with
            D_x q = d_x q + d_n q . J_n

        Note:
            Jq_normal_dir = (D_x q) n
        is the directional derivative along the ambient normal direction.
        It is NOT d_n q and NOT d_{n(x)} q.
        """
        out, Jq_total, n, Jq_x, Jq_nx = self.jacobian_q_total(surf_pts, ref_mode=ref_mode)

        # project total Jacobian to tangent plane:
        # Jq_tan = Jq_total - ((Jq_total n) (x) n)
        Jq_normal_dir = (Jq_total * n[:, None, :]).sum(-1, keepdim=True)   # (N, 9, 1)
        Jq_tan = Jq_total - Jq_normal_dir * n[:, None, :]                   # (N, 9, 3)

        energy = (Jq_tan ** 2).sum(dim=(1, 2))   # per-point ||grad_M q||^2

        if p == 2.0:
            return energy.mean(), out, Jq_total, n, Jq_x, Jq_nx
        return energy.pow(p / 2.0).mean(), out, Jq_total, n, Jq_x, Jq_nx

    # ================================================================
    # Loss: code anti-collapse
    # ================================================================

    @staticmethod
    def loss_amp(code_raw, tau=0.5):
        """softplus(tau^2 - |a_raw|^2) per point, prevents code from vanishing."""
        return F.softplus(tau**2 - (code_raw**2).sum(-1)).mean()

    # ================================================================
    # Training
    # ================================================================

    def train(
        self,
        mesh,
        feat_pts_np,
        feat_tangents_np,
        feat_face_pairs_np: Optional[np.ndarray] = None,
        lambda_feat_edge: float = 100.0,
        num_epochs: int = 5000,
        num_surface: int = 5000,
        num_narrowband: int = 5000,
        # loss weights
        lambda_smooth: float = 1.0,
        lambda_feat: float = 5.0,
        lambda_cpm: float = 5.0,
        cpm_schedule: str = "fixed",
        lambda_cpm_ramp_start: float = 0.0,
        cpm_ramp_hold_frac: float = 0.2,
        lambda_amp: float = 0.1,
        # hyperparams
        tau_amp: float = 0.5,
        band_width: float = 0.02,
        knn_k: int = 12,
        knn_k_scale: Optional[int] = None,
        smooth_p: float = 2.0,
        clip_grad: float = 1.0,
        use_w_curv: bool = True,
        feat_align_weight: str = "k1_minus_k2",
        curv_weight_mode: Optional[str] = None,
        curv_weight_floor: float = 0.001,
        curv_bottom_quantile: float = 0.2,
        # logging
        log_every: int = 100,
        save_every: int = 2000,
        save_dir: Optional[str] = None,
        dual_surface_smooth: bool = False,
        surface_placement: str = "random",
        lr_schedule: str = "cosine",
        fixed_lr: float = 5e-5,
        ref_mode: str = 'normal',
        feat_weight_schedule: str = "anneal",
        feat_edge_target: str = "mincurv",
        feat_schedule_plateau_start: float = 0.2,
        feat_schedule_plateau_end: float = 0.8,
    ):
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        if surface_placement not in ("random", "per_face_adj"):
            raise ValueError(
                "surface_placement must be 'random' or 'per_face_adj' "
                f"(got {surface_placement!r})")
        if feat_edge_target not in ("mincurv", "tangent"):
            raise ValueError(
                "feat_edge_target must be 'mincurv' or 'tangent' "
                f"(got {feat_edge_target!r})")
        ps = float(feat_schedule_plateau_start)
        pe = float(feat_schedule_plateau_end)
        if not (0.0 < ps < pe < 1.0):
            raise ValueError(
                "feat_schedule_plateau_start / feat_schedule_plateau_end must satisfy "
                f"0 < start < end < 1 (got {ps}, {pe})")
        if curv_weight_mode is not None:
            if feat_align_weight != "abs_abs_diff":
                raise ValueError(
                    "curv_weight_mode requires feat_align_weight='abs_abs_diff' "
                    f"(got {feat_align_weight!r})")
            if curv_weight_mode not in ("scale01", "bottom_floor"):
                raise ValueError(
                    "curv_weight_mode must be None, 'scale01', or 'bottom_floor' "
                    f"(got {curv_weight_mode!r})")
            if curv_weight_mode == "bottom_floor":
                if not (0.0 < curv_bottom_quantile < 1.0):
                    raise ValueError(
                        "curv_bottom_quantile must be in (0, 1) "
                        f"(got {curv_bottom_quantile})")
                if not (0.0 < curv_weight_floor < 1.0):
                    raise ValueError(
                        "curv_weight_floor must be in (0, 1) "
                        f"(got {curv_weight_floor})")
        if lr_schedule not in ("cosine", "fixed"):
            raise ValueError(
                "lr_schedule must be 'cosine' or 'fixed' "
                f"(got {lr_schedule!r})")
        if cpm_schedule not in ("fixed", "ramp"):
            raise ValueError(
                "cpm_schedule must be 'fixed' or 'ramp' "
                f"(got {cpm_schedule!r})")
        if feat_weight_schedule not in ("fixed", "anneal"):
            raise ValueError(
                "feat_weight_schedule must be 'fixed' or 'anneal' "
                f"(got {feat_weight_schedule!r})")
        if not (0.0 <= cpm_ramp_hold_frac < 1.0):
            raise ValueError(
                "cpm_ramp_hold_frac must be in [0, 1) "
                f"(got {cpm_ramp_hold_frac})")

        dev = self.device
        self.model.train()

        _fixed_lr = fixed_lr
        if lr_schedule == "cosine":
            for g in self.optimizer.param_groups:
                g["lr"] = self.lr
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=num_epochs, eta_min=1e-6)
        else:
            for g in self.optimizer.param_groups:
                g["lr"] = _fixed_lr
            scheduler = None

        n_feat = len(feat_pts_np)

        fc_tensor = None
        face_edge_idx = None
        Fm = mesh.faces_np.astype(np.int64)
        if dual_surface_smooth or surface_placement == "per_face_adj":
            # adj = undirected_face_vertex_ring_edges(Fm)
            adj = undirected_face_adjacency_edges(Fm)
            if adj.size == 0:
                if surface_placement == "per_face_adj":
                    raise ValueError(
                        "surface_placement='per_face_adj' needs at least one undirected "
                        "edge-adjacent face pair; got none (check mesh).")
                face_edge_idx = torch.zeros(0, 2, dtype=torch.long, device=dev)
            else:
                face_edge_idx = torch.from_numpy(adj).long().to(dev)
        if dual_surface_smooth:
            V = mesh.vertices_np.astype(np.float32)
            fc = (V[Fm[:, 0]] + V[Fm[:, 1]] + V[Fm[:, 2]]) * (1.0 / 3.0)
            fc_tensor = torch.from_numpy(fc).float().to(dev)

        # --- Feature-edge supervision targets (fixed across epochs) ---
        feat_edge_ready = (
            feat_face_pairs_np is not None
            and len(feat_face_pairs_np) > 0
            and surface_placement == "per_face_adj"
        )
        if feat_edge_ready:
            fep = torch.from_numpy(
                np.asarray(feat_face_pairs_np, dtype=np.int64)).to(dev)
            fmid = torch.from_numpy(
                np.asarray(feat_pts_np, dtype=np.float32)).to(dev)

            with torch.no_grad():
                n_mid = self.oracle.normal(fmid).detach()
                n_mid = F.normalize(n_mid, dim=-1, eps=1e-8)
                if feat_edge_target == "tangent":
                    t_mid = torch.from_numpy(
                        np.asarray(feat_tangents_np, dtype=np.float32)).to(dev)
                else:
                    t_mid, _ = compute_feature_direction_mincurv(
                        self.oracle, fmid,
                        feat_align_weight=feat_align_weight,
                        curv_weight_mode=None)
                    t_mid = t_mid.detach()
                # Project onto tangent plane of n_mid for orthogonality, then normalize
                t_mid = t_mid - (t_mid * n_mid).sum(-1, keepdim=True) * n_mid
                t_mid = F.normalize(t_mid, dim=-1, eps=1e-8)
            print(f"  Feature-edge supervision: {len(fep)} edges "
                  f"(lambda_feat_edge={lambda_feat_edge}, target={feat_edge_target!r})")
        else:
            fep = None
            n_mid = t_mid = None
            print("  Feature-edge supervision: disabled")

        print(f"\nTraining theta-field frame field: {num_epochs} epochs")
        if dual_surface_smooth and fc_tensor is not None:
            print(f"  dual_surface_smooth: ON  |faces|={mesh.faces_np.shape[0]}  "
                  f"|adj_edges|={face_edge_idx.shape[0]}")
        else:
            print("  dual_surface_smooth: OFF")
        print(f"  smooth={lambda_smooth} feat={lambda_feat} "
              f"cpm={lambda_cpm} (schedule={cpm_schedule!r}"
              f"{'' if cpm_schedule == 'fixed' else f', hold_frac={cpm_ramp_hold_frac}, ramp_start={lambda_cpm_ramp_start}'}) "
              f"amp={lambda_amp}")
        print(f"  Feature points (mesh extract only): {n_feat}")
        print(f"  L_feat: random surface samples; L_feat_fc: face centers (if dual_surface_smooth); "
              f"align weight = {feat_align_weight!r}")
        if curv_weight_mode is not None:
            print(f"  curv_weight_mode={curv_weight_mode!r}  "
                  f"(floor={curv_weight_floor}, bottom_q={curv_bottom_quantile})")
        if feat_weight_schedule == "fixed":
            print("  lambda_feat: fixed (lambda_feat_eff = lambda_feat for all epochs)")
        else:
            print(
                "  lambda_feat: anneal - s=t/T: "
                f"0-{100 * ps:.0f}% ramp 0.2->1x, "
                f"{100 * ps:.0f}-{100 * pe:.0f}% 1x, "
                f"{100 * pe:.0f}-100% cosine->0.2x")
        print("  lambda_smooth: constant (no schedule)")
        if lr_schedule == "cosine":
            print(f"  lr_schedule=cosine  (Adam {self.lr:g} -> 1e-6, CosineAnnealingLR T_max={num_epochs})")
        else:
            print(f"  lr_schedule=fixed  (Adam lr={_fixed_lr:g}, no scheduler)")
        print(f"  surface_placement={surface_placement!r}  (per_face_adj -> |F| pts + vertex-ring "
              f"face graph; num_surface ignored for surf batch)")
        if surface_placement == "random":
            _kns = knn_k_scale if knn_k_scale is not None else knn_k
            print(f"  L_smooth_knn: k={knn_k}, local-scale sigma from {_kns}-th NN, heat-kernel weights")
        else:
            ne = int(face_edge_idx.shape[0]) if face_edge_idx is not None else 0
            print(f"  L_smooth_knn: face **vertex-ring** Dirichlet ({ne} undirected face pairs)")

        for epoch in range(1, num_epochs + 1):

            # ---- Sample surface & narrow band ----
            if surface_placement == "per_face_adj":
                surf_pts, _ = mesh.sample_surface_per_face(device=dev)
            else:
                surf_pts, _ = mesh.sample_surface(N=num_surface, device=dev)
            # Off-surface band: surface samples + isotropic Gaussian noise (Var=band_width per axis)
            nb_base, _ = mesh.sample_surface(N=num_narrowband, device=dev)
            nb_pts = nb_base + torch.randn_like(nb_base) * math.sqrt(band_width)

            # ---- Analytic smoothness (total Jacobian) ----
            # L_smooth, out, Jq_total_surf, n_surf, Jq_x_surf, Jq_nx_surf = self.loss_smooth_analytic(
            #     surf_pts, p=smooth_p, ref_mode=ref_mode
            # )
            out, _, n_surf, Jq_x_surf, _ = self.jacobian_q_total(
                surf_pts, ref_mode=ref_mode)
            # ---- Surface smoothness: Euclidean kNN (random pts) or vertex-ring Dirichlet (per-face pts) ----
            if surface_placement == "per_face_adj":
                # L_smooth_knn = self.loss_smooth_face_adjacency(
                #     out['q'], face_edge_idx, p=smooth_p)
                L_smooth_knn = self.loss_smooth_face_adjacency_transport(
                    out['e1'], out['e2'], out['n'], face_edge_idx, p=smooth_p)
            else:
                # L_smooth_knn = self.loss_smooth_knn(
                #     out['q'], surf_pts, knn_k=knn_k, p=smooth_p,
                #     knn_k_scale=knn_k_scale)
                L_smooth_knn = self.loss_smooth_knn_transport(
                    out['e1'], out['e2'], out['n'], surf_pts,
                    knn_k=knn_k, p=smooth_p, knn_k_scale=knn_k_scale)

            L_smooth_fc = torch.tensor(0.0, device=dev)
            L_feat_fc = torch.tensor(0.0, device=dev)
            if dual_surface_smooth and fc_tensor is not None and face_edge_idx is not None:
                out_fc = self.forward_surface(
                    fc_tensor, create_graph=True, ref_mode=ref_mode)
                # L_smooth_fc = self.loss_smooth_face_adjacency(
                #     out_fc['q'], face_edge_idx, p=smooth_p)
                L_smooth_fc = self.loss_smooth_face_adjacency_transport(
                    out_fc['e1'], out_fc['e2'], out_fc['n'], face_edge_idx, p=smooth_p)
                t_fc, w_fc = compute_feature_direction_mincurv(
                    self.oracle, fc_tensor,
                    feat_align_weight=feat_align_weight,
                    curv_weight_mode=curv_weight_mode,
                    curv_weight_floor=curv_weight_floor,
                    curv_bottom_quantile=curv_bottom_quantile)
                L_feat_fc = self.loss_feat(
                    out_fc['q'], t_fc,
                    w_curv=(w_fc if use_w_curv else None))
            L_smooth = L_smooth_knn + L_smooth_fc if dual_surface_smooth else L_smooth_knn

            # ---- CPM on surface: (D_x q).n = (d_x q).n since J_n n = 0 ----
            # Jq_normal_dir_surf = (Jq_x_surf * n_surf[:, None, :]).sum(-1)   # (N, 9)
            # L_cpm_surf = (Jq_normal_dir_surf ** 2).sum(-1).mean()

            # ---- Amp ----
            L_amp = torch.tensor(0.0, device=dev)

            # ---- Principal-direction alignment (all surface samples, same loss_feat) ----
            t_surf, w_surf = compute_feature_direction_mincurv(
                self.oracle, surf_pts,
                feat_align_weight=feat_align_weight,
                curv_weight_mode=curv_weight_mode,
                curv_weight_floor=curv_weight_floor,
                curv_bottom_quantile=curv_bottom_quantile)
            L_feat = self.loss_feat(
                out['q'], t_surf,
                w_curv=(w_surf if use_w_curv else None))

            # ---- Feature-edge alignment: two face-side samples transported to mid-point ----
            L_feat_edge = torch.tensor(0.0, device=dev)
            if feat_edge_ready:
                f0, f1 = fep[:, 0], fep[:, 1]
                L_feat_edge = self.loss_feat_edge_transport(
                    out['e1'][f0], out['e2'][f0], out['n'][f0].detach(),
                    out['e1'][f1], out['e2'][f1], out['n'][f1].detach(),
                    n_mid, t_mid)

            s_prog = epoch / float(num_epochs)
            if feat_weight_schedule == "fixed":
                lambda_feat_eff = lambda_feat
                lambda_feat_edge_eff = lambda_feat_edge
            else:
                r = lambda_feat_schedule_ratio(
                    s_prog, plateau_start=ps, plateau_end=pe)
                lambda_feat_eff = lambda_feat * r
                lambda_feat_edge_eff = lambda_feat_edge * r

            if cpm_schedule == "fixed":
                lambda_cpm_eff = lambda_cpm
            else:
                hold = cpm_ramp_hold_frac
                if s_prog <= hold:
                    lambda_cpm_eff = 0.0
                else:
                    t = (s_prog - hold) / (1.0 - hold)
                    lambda_cpm_eff = (
                        lambda_cpm_ramp_start
                        + (lambda_cpm - lambda_cpm_ramp_start) * t)

            # ---- CPM on narrow band ----
            # NOTE: only build the L_cpm graph when it will actually contribute
            # to the loss. Otherwise autograd still traverses L_cpm's subgraph
            # during loss.backward(); any NaN there (e.g. from the second-order
            # grad(q, x_leaf) path under certain ref_modes / narrow-band points)
            # is multiplied by 0 -> `0 * NaN = NaN` and contaminates params.
            if lambda_cpm_eff > 0.0:
                L_cpm = self.loss_cpm(nb_pts, ref_mode=ref_mode)
            else:
                L_cpm = torch.tensor(0.0, device=dev)

            # ---- Total ----
            loss = (
                    # lambda_smooth * (L_smooth_knn + L_smooth_fc)
                    lambda_smooth * (  L_smooth_knn)
                    # + lambda_feat_eff * (L_feat + L_feat_2 + L_feat_fc)
                    + lambda_feat_eff * (L_feat)
                    + lambda_feat_edge_eff * L_feat_edge
                    + lambda_cpm_eff * L_cpm
                    + lambda_amp * L_amp)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad)
            self.optimizer.step()
            if scheduler is not None:
                scheduler.step()

            # ---- Logging ----
            if epoch % log_every == 0 or epoch == 1:
                with torch.no_grad():
                    theta_abs = out['theta'].abs().mean().item()
                    theta_std = out['theta'].std().item()
                    q_var = out['q'].var(dim=0).sum().item()

                    # debug: kNN smooth
                    # L_smooth_knn_dbg = self.loss_smooth_knn(
                    #     out['q'], surf_pts, knn_k=knn_k, p=smooth_p).item()

                    # debug: normal alignment residual
                    # q_dbg = out['q'] / (out['q'].norm(dim=-1, keepdim=True) + 1e-8)
                    # n_dbg = out['n'] / (out['n'].norm(dim=-1, keepdim=True) + 1e-8)
                    # gF_n = grad_oct_polynomial_sh4(n_dbg, q_dbg)
                    # nalign = ((gF_n - 4.0 * n_dbg) ** 2).sum(-1).mean().item()

                print(
                    f"  [{epoch}/{num_epochs}]  "
                    f"loss={loss.item():.4f}  "
                    # f"smooth={L_smooth.item():.4f}  "
                    f"L_smooth_knn={L_smooth_knn:.4f}  "
                    f"L_smooth_fc={L_smooth_fc.item():.4f}  "
                    f"feat={L_feat.item():.6f}  feat_fc={L_feat_fc.item():.6f}  "
                    f"feat_edge={L_feat_edge.item():.6f}  "
                    f"lambda_feat_eff={lambda_feat_eff:.5f}  "
                    f"lambda_feat_edge_eff={lambda_feat_edge_eff:.5f}  "
                    f"lambda_cpm_eff={lambda_cpm_eff:.5f}  "
                    f"cpm={L_cpm.item():.6f}  "
                    # f"nalign={nalign:.6f}  "
                    f"|theta|={theta_abs:.4f}  theta_std={theta_std:.4f}  "
                    f"q_var={q_var:.4f}  "
                    f"lr={self.optimizer.param_groups[0]['lr']:.6f}")

            if save_dir and save_every and epoch % save_every == 0:
                self.save(os.path.join(save_dir, f"epoch_{epoch}.pt"))

    # ================================================================
    # Save / Load
    # ================================================================

    def save(self, path):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'hidden': self.hidden,
            'n_layers': self.n_layers,
            'lr': self.lr,
        }, path)
        print(f"Saved -> {path}")

    @classmethod
    def load(cls, path, sdf_oracle, device='cpu'):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        obj = cls(sdf_oracle,
                  hidden=ckpt.get('hidden', 128),
                  n_layers=ckpt.get('n_layers', 4),
                  lr=ckpt.get('lr', 1e-3), device=device)
        obj.model.load_state_dict(ckpt['model'])
        obj.optimizer.load_state_dict(ckpt['optimizer'])
        print(f"Loaded <- {path}")
        return obj

    def set_eval(self):
        self.model.eval()

    def set_train(self):
        self.model.train()

    # ================================================================
    # Evaluation
    # ================================================================

    def evaluate(self, sample_surface_fn, N=5000):
        self.set_eval()
        pts, _ = sample_surface_fn(N=N, device=self.device)
        with torch.enable_grad():
            out = self.forward_surface(pts, create_graph=False)
        return {k: out[k].detach().cpu().numpy()
                if isinstance(out[k], torch.Tensor) else out[k]
                for k in ['theta', 'e1', 'e2', 'n', 'q']
               } | {'pts': pts.cpu().numpy()}

    def evaluate_mesh_face_centers(
        self,
        mesh,
        save_path=None,
        bs=4096,
        ref_mode='normal',
        feat_align_weight: str = 'k1_minus_k2',
        curv_weight_mode: Optional[str] = None,
        curv_weight_floor: float = 0.001,
        curv_bottom_quantile: float = 0.2,
        include_feat_curves: bool = False,
        feat_curve_angle_deg: float = 30.0,
        feat_edge_target: str = 'mincurv',
    ):
        if feat_edge_target not in ('mincurv', 'tangent'):
            raise ValueError(
                "feat_edge_target must be 'mincurv' or 'tangent' "
                f"(got {feat_edge_target!r})")
        self.set_eval()
        dev = self.device
        V = mesh.vertices_np.astype(np.float32)
        Faces = mesh.faces_np.astype(np.int64)
        v0, v1, v2 = V[Faces[:, 0]], V[Faces[:, 1]], V[Faces[:, 2]]
        fc = ((v0 + v1 + v2) / 3.0).astype(np.float32)

        keys = ['theta', 'e1', 'e2', 'q']
        keys_ref = [
            'e1_ref', 'e2_ref', 'n_ref', 't_feat',
            'feat_align', 'w_curv', 'feat_align_weighted',
        ]
        acc = {k: [] for k in keys + keys_ref}

        with torch.enable_grad():
            for s in range(0, len(fc), bs):
                e = min(s + bs, len(fc))
                pts = torch.from_numpy(fc[s:e]).to(dev)
                o = self.forward_surface(pts, create_graph=False, ref_mode=ref_mode)

                n_face = torch.from_numpy(mesh.face_normals_np[s:e]).to(dev)
                n_face = F.normalize(n_face, dim=-1, eps=1e-8)

                e1 = o['e1']
                e2 = o['e2']

                # project e1 to face plane
                e1 = e1 - (e1 * n_face).sum(-1, keepdim=True) * n_face
                e1 = F.normalize(e1, dim=-1, eps=1e-8)

                # rebuild e2 in the face plane
                e2_new = torch.cross(n_face, e1, dim=-1)
                e2_new = F.normalize(e2_new, dim=-1, eps=1e-8)

                # choose sign closest to original e2
                flip = ((e2_new * e2).sum(-1, keepdim=True) < 0)
                e2 = torch.where(flip, -e2_new, e2_new)

                # recompute q from projected frame
                R = torch.stack([e1, e2, n_face], dim=1)
                q_raw = R3_to_sh4_zonal(R)
                q = q_raw / (q_raw.norm(dim=-1, keepdim=True) + 1e-8)

                # initial reference frame at face center (normal/principal)
                n_ref = self.oracle.normal(pts).detach()
                n_ref = F.normalize(n_ref, dim=-1, eps=1e-8)
                if ref_mode == 'principal':
                    k1, k2, _ = compute_hessian_principal_dirs(self.oracle, pts)
                    e1_ref, e2_ref = k1.detach(), k2.detach()
                elif ref_mode == 'axis_swap':
                    r1, r2 = build_reference_frame_axis_swap(n_ref)
                    e1_ref, e2_ref = r1.detach(), r2.detach()
                else:
                    r1, r2 = build_reference_frame(n_ref)
                    e1_ref, e2_ref = r1.detach(), r2.detach()

                # per-face feature alignment (same pieces as ``loss_feat``):
                #   feat_align = ||grad_{t_feat} F(q, t_feat) - 4 t_feat||^2  (no weight)
                #   w_curv     = curvature weight from ``compute_feature_direction_mincurv``
                #   feat_align_weighted = w_curv * feat_align  (per-face integrand of L_feat)
                t_feat, w_curv = compute_feature_direction_mincurv(
                    self.oracle, pts,
                    feat_align_weight=feat_align_weight,
                    curv_weight_mode=curv_weight_mode,
                    curv_weight_floor=curv_weight_floor,
                    curv_bottom_quantile=curv_bottom_quantile,
                )
                t_feat = F.normalize(t_feat.detach(), dim=-1, eps=1e-8)
                w_curv = w_curv.detach()
                gF = grad_oct_polynomial_sh4(t_feat, q)
                fa = ((gF - 4.0 * t_feat) ** 2).sum(-1)
                fa_w = fa * w_curv

                # keep original theta; export projected e1/e2/q
                acc['theta'].append(o['theta'].detach().cpu().numpy())
                acc['e1'].append(e1.detach().cpu().numpy())
                acc['e2'].append(e2.detach().cpu().numpy())
                acc['q'].append(q.detach().cpu().numpy())
                acc['e1_ref'].append(e1_ref.cpu().numpy())
                acc['e2_ref'].append(e2_ref.cpu().numpy())
                acc['n_ref'].append(n_ref.cpu().numpy())
                acc['t_feat'].append(t_feat.cpu().numpy())
                acc['feat_align'].append(fa.detach().cpu().numpy())
                acc['w_curv'].append(w_curv.cpu().numpy())
                acc['feat_align_weighted'].append(fa_w.detach().cpu().numpy())

        normals = self.oracle.normal(
            torch.from_numpy(fc).to(dev)
        ).detach().cpu().numpy()

        data = {
            'pts': fc,
            'face_centers': fc,
            'face_normals': normals,
            'face_normals_mesh': mesh.face_normals_np.astype(np.float32),
            'face_ids': np.arange(len(Faces), dtype=np.int64),
            'mesh_vertices': V,
            'mesh_faces': Faces,
            'ref_mode': np.asarray(ref_mode),
        }
        for k in keys + keys_ref:
            data[k] = np.concatenate(acc[k], axis=0)

        # Per-face SH smoothness: transport-aligned vs plain ||q_i - q_j||^2 (edge-adj + vertex-ring)
        e1_t = torch.from_numpy(data['e1']).to(dev)
        e2_t = torch.from_numpy(data['e2']).to(dev)
        n_t = torch.from_numpy(data['face_normals']).to(dev)
        nf = len(Faces)

        def _per_face_sh_smooth(adj_np, transport: bool):
            if adj_np.size == 0:
                return np.zeros(nf, dtype=np.float32)
            ed = torch.from_numpy(adj_np).long().to(dev)
            i0, i1 = ed[:, 0], ed[:, 1]
            ni = F.normalize(n_t[i0], dim=-1, eps=1e-8)
            nj = F.normalize(n_t[i1], dim=-1, eps=1e-8)

            q_i = R3_to_sh4_zonal(torch.stack([e1_t[i0], e2_t[i0], ni], dim=1))
            q_i = q_i / (q_i.norm(dim=-1, keepdim=True) + 1e-8)
            if transport:
                e1_j_t = shortest_rotation_apply(nj, ni, e1_t[i1])
                e2_j_t = shortest_rotation_apply(nj, ni, e2_t[i1])
                q_jm = R3_to_sh4_zonal(torch.stack([e1_j_t, e2_j_t, ni], dim=1))
            else:
                q_jm = R3_to_sh4_zonal(torch.stack([e1_t[i1], e2_t[i1], nj], dim=1))
            q_jm = q_jm / (q_jm.norm(dim=-1, keepdim=True) + 1e-8)
            d_ij = ((q_i - q_jm) ** 2).sum(-1)

            # e1_i_t = shortest_rotation_apply(ni, nj, e1_t[i0])
            # e2_i_t = shortest_rotation_apply(ni, nj, e2_t[i0])
            # q_ij = R3_to_sh4_zonal(torch.stack([e1_i_t, e2_i_t, nj], dim=1))
            # q_ij = q_ij / (q_ij.norm(dim=-1, keepdim=True) + 1e-8)
            # q_j = R3_to_sh4_zonal(torch.stack([e1_t[i1], e2_t[i1], nj], dim=1))
            # q_j = q_j / (q_j.norm(dim=-1, keepdim=True) + 1e-8)
            # d_ji = ((q_j - q_ij) ** 2).sum(-1)
            # d = 0.5 * (d_ij + d_ji)
            d = d_ij
            acc_d = torch.zeros(nf, device=dev, dtype=d.dtype)
            deg = torch.zeros(nf, device=dev, dtype=d.dtype)
            acc_d.index_add_(0, i0, d)
            acc_d.index_add_(0, i1, d)
            ones = torch.ones(d.shape[0], device=dev, dtype=d.dtype)
            deg.index_add_(0, i0, ones)
            deg.index_add_(0, i1, ones)
            return (acc_d / deg.clamp(min=1.0)).cpu().numpy().astype(np.float32)

        with torch.no_grad():
            ee = undirected_face_adjacency_edges(Faces)
            vv = undirected_face_vertex_ring_edges(Faces)
            data['smooth_transport_edge_adj'] = _per_face_sh_smooth(ee, True)
            data['smooth_transport_vertex_ring'] = _per_face_sh_smooth(vv, True)
            data['smooth_knn_edge_adj'] = _per_face_sh_smooth(ee, False)
            data['smooth_knn_vertex_ring'] = _per_face_sh_smooth(vv, False)

        if include_feat_curves:
            fpts, ftan, _, fpairs = extract_feature_curves(
                mesh.vertices_np,
                mesh.faces_np,
                mesh.face_normals_np,
                angle_deg=float(feat_curve_angle_deg),
            )
            fpts = np.asarray(fpts, dtype=np.float32)
            ftan = np.asarray(ftan, dtype=np.float32)
            fpairs = np.asarray(fpairs, dtype=np.int64)
            data['feat_curve_pts'] = fpts
            data['feat_curve_tangent'] = ftan
            data['feat_curve_face_pairs'] = fpairs
            data['feat_curve_angle_deg'] = np.array(
                float(feat_curve_angle_deg), dtype=np.float32)
            z0 = np.zeros((0, 3), dtype=np.float32)
            z0e = np.zeros((0,), dtype=np.float32)
            if fpts.shape[0] == 0:
                data['feat_curve_k1'] = z0
                data['feat_curve_k2'] = z0
                data['feat_curve_n_sdf'] = z0
                data['feat_curve_L_feat_edge'] = z0e
                print("  feat curves: 0 points (skip principal dirs / L_feat_edge)")
            else:
                k1_parts, k2_parts, n_parts = [], [], []
                with torch.enable_grad():
                    for s in range(0, fpts.shape[0], bs):
                        sl = slice(s, s + bs)
                        pts_t = torch.from_numpy(fpts[sl]).to(dev).float().requires_grad_(True)
                        k1, k2, _ = compute_hessian_principal_dirs(self.oracle, pts_t)
                        n_s = self.oracle.normal(pts_t).detach()
                        k1_parts.append(k1.detach().cpu().numpy().astype(np.float32))
                        k2_parts.append(k2.detach().cpu().numpy().astype(np.float32))
                        n_parts.append(n_s.cpu().numpy().astype(np.float32))
                data['feat_curve_k1'] = np.concatenate(k1_parts, axis=0)
                data['feat_curve_k2'] = np.concatenate(k2_parts, axis=0)
                data['feat_curve_n_sdf'] = np.concatenate(n_parts, axis=0)

                # L_feat_edge per feature edge (same transport + oct alignment as training)
                e1r, e2r, nr = [], [], []
                with torch.enable_grad():
                    for s in range(0, len(fc), bs):
                        e = min(s + bs, len(fc))
                        pts = torch.from_numpy(fc[s:e]).to(dev)
                        o = self.forward_surface(pts, create_graph=False, ref_mode=ref_mode)
                        e1r.append(o['e1'].detach())
                        e2r.append(o['e2'].detach())
                        nr.append(o['n'].detach())
                e1_raw = torch.cat(e1r, dim=0)
                e2_raw = torch.cat(e2r, dim=0)
                n_raw = torch.cat(nr, dim=0)
                fmid = torch.from_numpy(fpts).to(dev).float()
                with torch.no_grad():
                    n_mid_e = F.normalize(
                        self.oracle.normal(fmid).detach(), dim=-1, eps=1e-8)
                    if feat_edge_target == 'tangent':
                        t_mid_e = torch.from_numpy(ftan).to(dev).float()
                    else:
                        t_mid_e, _ = compute_feature_direction_mincurv(
                            self.oracle, fmid,
                            feat_align_weight=feat_align_weight,
                            curv_weight_mode=None)
                        t_mid_e = t_mid_e.detach()
                    t_mid_e = t_mid_e - (t_mid_e * n_mid_e).sum(-1, keepdim=True) * n_mid_e
                    t_mid_e = F.normalize(t_mid_e, dim=-1, eps=1e-8)
                f0 = torch.from_numpy(fpairs[:, 0]).long().to(dev)
                f1 = torch.from_numpy(fpairs[:, 1]).long().to(dev)
                L_edge_chunks = []
                Ee = int(f0.shape[0])
                with torch.enable_grad():
                    for s in range(0, Ee, bs):
                        sl = slice(s, min(s + bs, Ee))
                        L_edge_chunks.append(
                            self.loss_feat_edge_transport_per_edge(
                                e1_raw[f0[sl]], e2_raw[f0[sl]], n_raw[f0[sl]],
                                e1_raw[f1[sl]], e2_raw[f1[sl]], n_raw[f1[sl]],
                                n_mid_e[sl], t_mid_e[sl],
                            ).detach().cpu().numpy().astype(np.float32))
                data['feat_curve_L_feat_edge'] = np.concatenate(L_edge_chunks, axis=0)
                print(
                    f"  feat curves: {fpts.shape[0]} pts; L_feat_edge per edge saved "
                    f"(mean={float(data['feat_curve_L_feat_edge'].mean()):.6e})")

        if save_path:
            os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
            np.savez(save_path, **data)
            print(f"Exported -> {save_path}")
        return data

    def visualize(self, sample_surface_fn, N=5000,
                  save_path=None, show=True, ref_mode='normal'):
        import matplotlib.pyplot as plt
        data = self.evaluate(sample_surface_fn, N)
        pts = data['pts']

        # initial reference frame (normal/principal) + per-point feat_align at sampled pts
        with torch.no_grad():
            pts_t = torch.from_numpy(pts).to(self.device)
            n_ref = self.oracle.normal(pts_t).detach()
            n_ref = F.normalize(n_ref, dim=-1, eps=1e-8)
            if ref_mode == 'principal':
                k1, k2, _ = compute_hessian_principal_dirs(self.oracle, pts_t)
                e1_ref, e2_ref = k1, k2
            elif ref_mode == 'axis_swap':
                r1, r2 = build_reference_frame_axis_swap(n_ref)
                e1_ref, e2_ref = r1, r2
            else:
                r1, r2 = build_reference_frame(n_ref)
                e1_ref, e2_ref = r1, r2
            t_feat, _ = compute_feature_direction_mincurv(self.oracle, pts_t)
            t_feat = F.normalize(t_feat, dim=-1, eps=1e-8)
            q_t = torch.from_numpy(data['q']).to(self.device)
            q_t = q_t / (q_t.norm(dim=-1, keepdim=True) + 1e-8)
            gF = grad_oct_polynomial_sh4(t_feat, q_t)
            fa = ((gF - 4.0 * t_feat) ** 2).sum(-1)
        data['e1_ref'] = e1_ref.cpu().numpy()
        data['e2_ref'] = e2_ref.cpu().numpy()
        data['n_ref'] = n_ref.cpu().numpy()
        data['t_feat'] = t_feat.cpu().numpy()
        data['feat_align'] = fa.cpu().numpy()
        data['ref_mode'] = np.asarray(ref_mode)

        fig = plt.figure(figsize=(20, 16))

        # 1) theta field
        ax = fig.add_subplot(2, 2, 1, projection='3d')
        sc = ax.scatter(pts[:,0], pts[:,1], pts[:,2],
                        c=data['theta'].flatten(), cmap='hsv', s=2, alpha=0.8)
        fig.colorbar(sc, ax=ax, shrink=0.6, label='theta')
        ax.set_title('Twist angle theta')

        # 2) |theta|
        ax = fig.add_subplot(2, 2, 2, projection='3d')
        sc = ax.scatter(pts[:,0], pts[:,1], pts[:,2],
                        c=np.abs(data['theta'].flatten()),
                        cmap='viridis', s=2, alpha=0.8)
        fig.colorbar(sc, ax=ax, shrink=0.6, label='|theta|')
        ax.set_title('Angle magnitude')

        # 3) Frame quiver
        ax = fig.add_subplot(2, 2, 3, projection='3d')
        step = max(1, len(pts)//400); s = 0.03
        ax.scatter(pts[:,0], pts[:,1], pts[:,2],
                   c='lightgray', s=0.5, alpha=0.3)
        ax.quiver(pts[::step,0], pts[::step,1], pts[::step,2],
                  data['e1'][::step,0]*s, data['e1'][::step,1]*s,
                  data['e1'][::step,2]*s,
                  color='steelblue', alpha=0.7, linewidth=0.8)
        ax.quiver(pts[::step,0], pts[::step,1], pts[::step,2],
                  data['e2'][::step,0]*s, data['e2'][::step,1]*s,
                  data['e2'][::step,2]*s,
                  color='red', alpha=0.7, linewidth=0.8)
        ax.set_title('Frame (blue=e1, red=e2)')

        # 4) SH q[4]
        ax = fig.add_subplot(2, 2, 4, projection='3d')
        sc = ax.scatter(pts[:,0], pts[:,1], pts[:,2],
                        c=data['q'][:,4], cmap='viridis', s=2, alpha=0.8)
        fig.colorbar(sc, ax=ax, shrink=0.6, label='q_4')
        ax.set_title('SH q_4 (Y_4^0)')

        fig.suptitle('theta-Field Frame Field (Fixed Oracle)',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        if show: plt.show()
        else: plt.close(fig)
        npz = (save_path.rsplit('.',1)[0] if save_path else 'frame_field')+'.npz'
        np.savez(npz, **data); print(f"Exported -> {npz}")


if __name__ == "__main__":
    _debug_check_q_vs_frame()
