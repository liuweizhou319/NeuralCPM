"""
Surfacescreendiffusion.py - neural screened-Dirichlet diffusion field on a surface
==================================================================================

Core solver of this repository.  A small sine MLP  u(x, n(x), code)  is trained
so that it minimises a convex quadratic screened-Dirichlet energy on a
triangle mesh M.  In words: the field tries to be smooth and bounded
(Dirichlet + mass terms), and on each prescribed source S_k it should match
the prescribed value a_k.

Place in the pipeline
---------------------
This file is a library (plus a small viewer).  train_screen.py builds the
normalised mesh (Mesh_geometry.py) and the SDF oracle (SparseGridFDM_SDF.py),
creates a ``Surfacescreendiffusion`` object and calls ``train()``,
``export()`` (writes diffusion.npz) and ``save()`` (writes model.pt).
fem_screen.py solves the same variational problem with P1 finite elements and
provides the reference solution used by compare_nn_fem_mse.py and
evaluate_dirichlet_generalization.py.  The evaluation and export scripts
rebuild the network from model.pt, either through ``MLP_Scalar``
(evaluate_dirichlet_generalization.py) or through
``Surfacescreendiffusion.load()`` (export_split_samples.py,
export_worst_rel_l2.py).

The only command-line use of this file is the polyscope viewer for an
exported field, results/single/<run>/diffusion.npz (run from the repository
root).  The example below continues the train_screen.py usage example
(--out-dir results/single/fandisk).  run_single.sh writes by default to
results/single/fandisk_res200_halo6 instead.

    python Surfacescreendiffusion.py --vis results/single/fandisk/diffusion.npz

Energy
------
For m sources S_1..S_m with target values a_k and weights lambda_k, the terms
L_src + lambda_dir * L_dir + lambda_mass * L_mass of the loss (see below) are
a Monte-Carlo estimate of

    E(u) =   (lambda_dir  / |M|) * int_M |grad u|^2 dA            (smoothness)
           + (lambda_mass / |M|) * int_M u^2 dA                   (screen / mass)
           + (1 / m) * sum_k lambda_k * mean_{S_k} (u - a_k)^2    (Dirichlet data)

The Dirichlet and mass terms are area means over M and carry no 1/2 factor.
The data term is averaged twice: over the samples of each source
(mean_{S_k}) and then over the sources (1/m), so that longer / denser sources
do not accidentally dominate.  fem_screen.py minimises the same energy up to
a constant factor, which does not change the minimiser: it multiplies the
energy by |M|, which is why the source term of the FEM system carries the
factor |M| / m.

When all a_k = 1 the result is the classical screened-Poisson influence
field (positive only).  When some a_k are negative, the result is a
*signed* screened-harmonic interpolation field - positive sources push u
upward, negative sources pull it downward, and  l ~ sqrt(lambda_dir/lambda_mass)
sets the influence radius.  In all cases the problem stays convex quadratic.

Loss as implemented (one batch per epoch)
-----------------------------------------
    L = L_src + lambda_dir * L_dir + lambda_mass * L_mass
        + lambda_cpm * L_cpm + lambda_range * L_range

    L_dir   : mean of |g|^2 over ``num_surface`` area-uniform surface samples,
              where  g = du/dx + J_n^T du/dn  is the chain-rule gradient of
              u(x, n(x)) through the SDF normal (J_n = dn/dx).  g is NOT
              projected onto the tangent plane.
    L_mass  : mean of u^2 over the same samples.
    L_src   : (1/m) * sum_k lambda_k * mean over the samples of S_k of
              (u - a_k)^2, with ``num_per_source`` samples per source.  Here
              m counts the sources that produced samples in this epoch (all of
              them unless a disk iso-contour is empty).  Source samples are
              projected with the SDF closest-point map before evaluation.
    L_cpm   : closest-point extension regulariser (``use_cpm``): mean over
              [narrow-band points | surface samples] of
              (du/dx . n)^2 + (u(x) - u(cp(x)))^2.  The ``num_narrowband``
              band points are surface samples plus isotropic Gaussian noise of
              standard deviation ``band_width`` and use the SDF normal at
              their closest point; the surface samples use the interpolated
              mesh normal.
    L_range : optional soft clamp of u to the range of the source targets,
              mean relu(v_min - u)^2 + mean relu(u - v_max)^2 (off by default).

An SDF oracle is required for training: without one the evaluation of the
source term raises NotImplementedError.

Network input
-------------
``_make_input`` packs the row  [x | n | code]  and ``MLP_Scalar`` encodes it
as  [PE(x) | n | code]:

    PE(x) : sin(f x), cos(f x) for f in (1, 2, 4, 8, 16); 30 numbers, the raw
            x is not kept.
    n     : SDF normal at x (3 numbers).  Present only when an SDF oracle is
            used; replaced by zeros with ``no_normal_lift=True`` (ablation).
            Inside L_cpm the row carries the normal described under L_cpm.
    code  : conditioning code (``code_size`` numbers, see "Generalization");
            the constant 0 when nothing is generalized.

The normal and the code are appended raw (no positional encoding).  Four sine
layers of width ``hidden`` are followed by a linear scalar output.

Sources
-------
``train()`` / ``set_sources()`` take a list of dicts; the source-spec JSON
files read by train_screen.py and fem_screen.py contain such a list.  Fields
shared by all types:

    type             : 'vertex' | 'disk_curve' | 'curve' (required)
    value            : target value a_k (default 1.0)
    weight           : lambda_k (default 1.0)
    value_generalize : bool (default False), see "Generalization"
    value_range      : [min, max] with min < max (default [-2, 2]); used only
                       when value_generalize is true

A source S_k is one of:

    * 'vertex'      : a small disk of mesh vertices around a chosen
                      vertex (a true point Dirichlet is too weak in 2D
                      H^1, so we fatten it to a few-vertex patch on the
                      mesh, sampled with replacement each epoch).
                      Fields: ``ids`` (non-empty list of vertex ids) and
                      ``patch_radius`` (default 0).  With patch_radius > 0
                      the ids are replaced by all vertices within that graph
                      distance of any listed vertex.
    * 'disk_curve'  : the iso-contour at a fixed geodesic radius around
                      a chosen seed vertex (Dijkstra-based), sampled
                      uniformly by length each epoch.
                      Fields: ``seed`` (vertex id, required), ``radius``
                      (default 0.1), ``generalize`` (default False) and
                      ``r_range`` ([min, max], required if generalize).
    * 'curve'       : a user-provided polyline (K, 3), K >= 2; sampled along
                      its length each epoch.
                      Fields: ``points`` and ``closed`` (default False).
                      train_screen.py also accepts ``points_npy`` in the JSON
                      and loads it into ``points``.

Each source carries (value a_k, weight lambda_k).  Source-wise loss averaging
ensures longer / denser sources don't accidentally dominate.

Geodesic radii are graph distances with doubled interior edges
--------------------------------------------------------------
The disk ``radius`` (and ``r_range``) and the vertex ``patch_radius`` are
thresholds on a Dijkstra distance along mesh edges, in the units of the mesh
passed in (train_screen.py normalises it).  The edge graph lists both
orientations of every face edge and duplicate entries are summed, so every
interior edge (shared by two faces) counts TWICE; a boundary edge counts
once.  On a closed mesh the graph distance is therefore 2x the edge-path
length: ``radius = 0.1`` is the contour at edge-path distance 0.05.
fem_screen.py uses the same convention, which keeps the sources of the
network and of the FEM reference identical.  All published numbers depend on
it; do not "fix" it.

Generalization (OFF by default)
-------------------------------
The network supports two mutually exclusive conditioning modes:

* ``value``: every source with ``value_generalize=True`` receives its own
  component of a dynamic code vector.  Components are assigned in ascending
  source index (the j-th marked source gets component j, stored as
  ``value_code_index``).  ``code_size`` must equal the number of marked
  sources, and all of them must share one ``value_range``.  Each epoch one
  code vector is drawn, every component independently and uniformly from
  ``value_range``; it conditions the whole batch, and component j is the
  target value of its source in that epoch.
* ``radius`` (legacy): every ``disk_curve`` with ``generalize=True`` uses the
  same scalar code, drawn uniformly from the shared ``r_range`` each epoch,
  as its radius; the iso-contours are re-extracted every epoch and
  ``code_size`` is 1.

The modes cannot be mixed; radius mode retains its legacy shared scalar.
Without generalization ``code_size`` must be 1 and the code input is 0.

Files written
-------------
``save(path)`` writes a torch checkpoint (model.pt in train_screen.py), a
dict with the keys

    model_u, optimizer    state dicts of the network and of Adam
    input_size, geo_size, code_size, hidden, lr
    sources               processed source dicts (arrays stored as lists;
                          vertex ``ids`` are the patch-expanded ids)
    generalization_mode   'none' | 'radius' | 'value'
    code_range            [min, max] of the conditioning code
    no_normal_lift        bool
    has_generalization    True in radius and in value mode (legacy flag)
    r_range               [min, max] in radius mode, [0, 0] otherwise (legacy)

``export(mesh, save_path)`` evaluates the field at the mesh vertices and
writes an uncompressed npz (diffusion.npz in train_screen.py) with

    mesh_vertices, mesh_faces    the mesh the field was evaluated on
    u_vertex                     u at the vertices
    grad_mag                     norm of the tangentially projected gradient
    code_value, code_values, code_size, value_code_source_indices
                                 conditioning code used for this export
    value_min, value_max         range of the source targets
    src_pts, src_pts_value, src_seg_p1, src_seg_p2, src_seg_value
                                 source markers / segments for the viewer
    has_generalization, r_range, generalization_mode, code_range,
    no_normal_lift               run metadata

``train()`` additionally writes ``<save_dir>/diffusion_epoch_<epoch>.pt`` when
``save_every > 0`` and, if ``train_codes_path`` is given, a TSV with the
conditioning code of every epoch.
"""

import os
import numpy as np
from typing import Optional, List, Sequence, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import grad

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra as _scipy_dijkstra


# ============================================================
# 1. Network - PE on x only; [normal, code] tail appended raw
# ============================================================

class MLP_Scalar(nn.Module):
    """Scalar potential: (x, n(x), code) -> scalar.

    PE on x only, sine hidden activations.  The input row is
    [x (input_size) | geo (geo_size)], where the geo tail is
    [normal (3) | code] with an SDF oracle and [code] without.  x is replaced
    by [sin(f x), cos(f x)] for f in ``freqs`` (the raw x is not kept); the
    geo tail is appended unchanged.
    """

    def __init__(self, input_size=3, geo_size=3, hidden=128):
        super().__init__()
        self.dim = input_size
        self.freqs = [1, 2, 4, 8, 16]

        d_in = input_size * len(self.freqs) * 2 + geo_size

        self.layer1 = nn.Linear(d_in, hidden)
        self.layer2 = nn.Linear(hidden, hidden)
        self.layer3 = nn.Linear(hidden, hidden)
        self.layer4 = nn.Linear(hidden, hidden)
        self.layer5 = nn.Linear(hidden, 1)

    def forward(self, x):
        """Evaluate u for rows [x | geo] of shape (N, input_size + geo_size).

        Returns a tensor of shape (N, 1).
        """
        # ``geo`` is either [code] (mesh-only) or [normal, code] (SDF).
        # The code tail is dynamic: value-generalization assigns one component
        # to every generalized source, while legacy radius mode keeps one.
        geo = x[:, self.dim:]
        x0 = x[:, :self.dim]

        enc = []
        for f in self.freqs:
            enc.append(torch.sin(f * x0))
            enc.append(torch.cos(f * x0))

        # Normal ablation is applied in ``_make_input`` so the network can
        # append the complete, dynamically-sized geometry/code tail here.
        enc.append(geo)

        h = torch.cat(enc, dim=1)
        h = torch.sin(self.layer1(h))
        h = torch.sin(self.layer2(h))
        h = torch.sin(self.layer3(h))
        h = torch.sin(self.layer4(h))
        return self.layer5(h)


# ============================================================
# 2. Geodesic-disk source helpers
# ============================================================

def _mesh_edge_graph(mesh):
    """Sparse edge graph of the mesh, weighted by edge length (float32).

    Both orientations of every face edge are listed and csr_matrix sums
    duplicate entries, so an interior edge (shared by two faces) gets TWICE
    its length as weight and a boundary edge once.  Dijkstra distances on a
    closed mesh are therefore 2x the edge-path length.  fem_screen.py builds
    its graph the same way and all source radii (disk ``radius``,
    ``patch_radius``) are thresholds on this distance; the published results
    depend on this convention, so it must not be changed.
    """
    V = mesh.vertices_np
    Fm = mesh.faces_np
    edges = np.concatenate(
        [Fm[:, [0, 1]], Fm[:, [1, 2]], Fm[:, [2, 0]],
         Fm[:, [1, 0]], Fm[:, [2, 1]], Fm[:, [0, 2]]], axis=0)
    w = np.linalg.norm(V[edges[:, 0]] - V[edges[:, 1]], axis=1).astype(np.float32)
    n_v = len(V)
    return csr_matrix((w, (edges[:, 0], edges[:, 1])), shape=(n_v, n_v))


def _face_iso_segments(mesh, dist_vertex, r):
    """Segments of the iso-contour {dist_vertex = r} of a per-vertex scalar.

    Every face with exactly two edges crossing the level r contributes one
    segment (linear interpolation along the two crossing edges).  Returns
    (p1, p2, nu), float32 arrays of shape (S, 3): the segment end points and
    the unit in-face gradient direction of the distance.  The callers in this
    file only use p1 and p2.
    """
    V = mesh.vertices_np
    Fm = mesh.faces_np
    pts = V[Fm]
    dv  = dist_vertex[Fm]

    a_idx = np.array([0, 1, 2])
    b_idx = np.array([1, 2, 0])
    da = dv[:, a_idx]
    db = dv[:, b_idx]
    crosses = ((da < r) & (db > r)) | ((da > r) & (db < r))

    denom = db - da
    denom = np.where(np.abs(denom) < 1e-20, 1e-20, denom)
    t = (r - da) / denom
    pa = pts[:, a_idx, :]
    pb = pts[:, b_idx, :]
    edge_pts = pa + t[..., None] * (pb - pa)

    num_cross = crosses.sum(axis=1)
    valid = num_cross == 2
    if not valid.any():
        z = np.zeros((0, 3), dtype=np.float32)
        return z, z.copy(), z.copy()

    vf = np.where(valid)[0]
    S = len(vf)
    vc = crosses[vf]
    vp = edge_pts[vf]
    order = np.argsort(-vc.astype(np.int8), axis=1)
    row = np.arange(S)
    p1 = vp[row, order[:, 0]]
    p2 = vp[row, order[:, 1]]

    pts_v = pts[vf]
    dv_v  = dv[vf]
    e01 = pts_v[:, 1] - pts_v[:, 0]
    e02 = pts_v[:, 2] - pts_v[:, 0]
    nface = np.cross(e01, e02)
    twoA = np.linalg.norm(nface, axis=1)
    safe = np.maximum(twoA, 1e-20)[:, None]
    nface_u = nface / safe
    v_term = (dv_v[:, 0:1] * (pts_v[:, 2] - pts_v[:, 1])
              + dv_v[:, 1:2] * (pts_v[:, 0] - pts_v[:, 2])
              + dv_v[:, 2:3] * (pts_v[:, 1] - pts_v[:, 0]))
    grad_d = np.cross(nface_u, v_term) / safe
    gmag = np.linalg.norm(grad_d, axis=1, keepdims=True)
    nu = grad_d / np.maximum(gmag, 1e-20)

    return (p1.astype(np.float32),
            p2.astype(np.float32),
            nu.astype(np.float32))


def _sample_on_segments(p1, p2, num_samples, rng=None):
    """Length-uniform sampling along the polyline of segments. Returns (N,3)."""
    if rng is None:
        rng = np.random
    S = len(p1)
    if S == 0 or num_samples <= 0:
        return np.zeros((0, 3), dtype=np.float32)
    lens = np.linalg.norm(p2 - p1, axis=1)
    tot = lens.sum()
    if tot < 1e-20:
        idx = rng.randint(0, S, num_samples)
        return p1[idx].astype(np.float32)
    cdf = np.cumsum(lens) / tot
    r = rng.rand(num_samples)
    idx = np.clip(np.searchsorted(cdf, r, side='right'), 0, S - 1)
    t = rng.rand(num_samples).astype(np.float32)[:, None]
    pts = p1[idx] + t * (p2[idx] - p1[idx])
    return pts.astype(np.float32)


def _sample_polyline(curve_np, num_samples, closed=False, rng=None):
    """Length-uniform sampling along an arbitrary polyline (K, 3)."""
    if rng is None:
        rng = np.random
    K = len(curve_np)
    if K < 2 or num_samples <= 0:
        if K >= 1:
            idx = rng.randint(0, K, size=num_samples)
            return curve_np[idx].astype(np.float32)
        return np.zeros((0, 3), dtype=np.float32)
    if closed:
        p1 = curve_np
        p2 = np.roll(curve_np, -1, axis=0)
    else:
        p1 = curve_np[:-1]
        p2 = curve_np[1:]
    return _sample_on_segments(p1.astype(np.float32),
                               p2.astype(np.float32),
                               num_samples, rng=rng)


class _GeodesicDiskSource:
    """Cached Dijkstra + iso-contour extractor for a set of seed vertices.

    All distances are graph distances on ``_mesh_edge_graph`` (interior edges
    count twice, see there).
    """

    def __init__(self, mesh):
        self.mesh = mesh
        self._graph = None
        self._seed_dists = {}
        self._segments = {}

    def _graph_lazy(self):
        if self._graph is None:
            self._graph = _mesh_edge_graph(self.mesh)
        return self._graph

    def dist_from(self, seed_id):
        """Per-vertex Dijkstra distance from seed_id (float32, cached)."""
        if seed_id not in self._seed_dists:
            g = self._graph_lazy()
            d = _scipy_dijkstra(g, directed=False, indices=int(seed_id))
            self._seed_dists[seed_id] = d.astype(np.float32)
            fin = np.isfinite(d)
            if fin.any():
                print(f"    Dijkstra seed={seed_id}: "
                      f"d in [{d[fin].min():.4f}, {d[fin].max():.4f}]")
            else:
                print(f"    Dijkstra seed={seed_id}: no finite distances")
        return self._seed_dists[seed_id]

    def extract(self, seed_id, radius, use_cache=True):
        """Iso-contour segments (p1, p2, nu) at distance ``radius`` of seed_id."""
        key = (int(seed_id), round(float(radius), 8))
        if use_cache and key in self._segments:
            return self._segments[key]
        d = self.dist_from(seed_id)
        segs = _face_iso_segments(self.mesh, d, float(radius))
        if use_cache:
            self._segments[key] = segs
        return segs

    def k_ring(self, seed_id, radius):
        """Return all vertex IDs whose Dijkstra distance from seed_id <= radius."""
        d = self.dist_from(seed_id)
        return np.where(d <= float(radius))[0]


# ============================================================
# 3. Main class - screened diffusion field with multi-source
# ============================================================

class Surfacescreendiffusion:
    """Neural screened-Dirichlet diffusion field on a surface with
    heterogeneous Dirichlet sources (vertex / disk_curve / curve), each
    carrying its own target value a_k and weight lambda_k.

    Parameters
    ----------
    input_size : dimension of the point coordinates (3).
    hidden     : width of the sine MLP.
    lr         : Adam learning rate.
    device     : torch device of the network and of all samples.
    sdf        : SDF oracle providing ``normal(x)``, ``normal_jacobian(x)``
                 and ``closest_point(x)`` (SparseGridFDM_SDF).  Required for
                 training; with ``None`` the network has no normal input.
    code_size  : number of conditioning-code components: the number of
                 value-generalized sources in value mode, otherwise 1.

    Typical use (see train_screen.py): construct, ``train(mesh, sources, ...)``,
    then ``export(mesh, path)`` and ``save(path)``.
    """

    def __init__(self, input_size=3, hidden=128, lr=1e-3,
                 device='cpu', sdf=None, code_size=1):
        self.device = torch.device(device)
        self.input_size = input_size
        self.hidden = hidden
        self.lr = lr
        self.sdf = sdf

        self.code_size = int(code_size)
        if self.code_size < 1:
            raise ValueError(f"code_size must be positive, got {self.code_size}")

        geo_size = (3 if sdf is not None else 0) + self.code_size
        self.geo_size = geo_size

        self.model_u = MLP_Scalar(input_size, geo_size, hidden).to(self.device)
        self.optimizer = optim.Adam(self.model_u.parameters(), lr=lr)

        n_params = sum(p.numel() for p in self.model_u.parameters())
        mode = "SDF chain-rule" if sdf is not None else "mesh-only"
        print(f"Surfacescreendiffusion ({mode}, geo_size={geo_size}): "
              f"u-net {n_params} params  [multi-source diffusion]")

        # Sources state
        self._sources: List[Dict[str, Any]] = []
        self._disk: Optional[_GeodesicDiskSource] = None
        self._generalization_mode: str = 'none'
        self._code_range = (0.0, 0.0)
        # source index -> code component.  In value mode the mapping is
        # one-to-one; legacy radius mode may map multiple disks to component 0.
        self._code_source_mapping: Dict[int, int] = {}
        self._no_normal_lift: bool = False

        # Legacy aliases kept for callers/checkpoints that predate value
        # generalization.  In radius mode _r_range equals _code_range.
        self._has_generalization: bool = False
        self._r_range = (0.0, 0.0)

    # ---------------- input packing ----------------
    def _code_tensor(self, code_value, dtype, device):
        """Conditioning code as a 1-D tensor with ``code_size`` entries."""
        if torch.is_tensor(code_value):
            code = code_value.detach().to(device=device, dtype=dtype).reshape(-1)
        else:
            code_np = np.asarray(code_value, dtype=np.float64).reshape(-1)
            code = torch.as_tensor(code_np, device=device, dtype=dtype)
        if code.numel() != self.code_size:
            raise ValueError(
                f"expected {self.code_size} code value(s), got {code.numel()}")
        return code

    def _code_array(self, code_value) -> np.ndarray:
        """Conditioning code as a float64 array of ``code_size`` finite entries."""
        if torch.is_tensor(code_value):
            code = code_value.detach().cpu().numpy()
        else:
            code = np.asarray(code_value, dtype=np.float64)
        code = np.asarray(code, dtype=np.float64).reshape(-1)
        if len(code) != self.code_size:
            raise ValueError(
                f"expected {self.code_size} code value(s), got {len(code)}")
        if not np.isfinite(code).all():
            raise ValueError("code values must be finite")
        return code

    def _make_input(self, pts_leaf, normals, code_value):
        """Pack the network input [x | n | code] (or [x | code] without SDF).

        The same code is broadcast to all N points.  With ``no_normal_lift``
        the normals are replaced by zeros.
        """
        N = pts_leaf.shape[0]
        code_one = self._code_tensor(
            code_value, dtype=pts_leaf.dtype, device=pts_leaf.device)
        code = code_one.unsqueeze(0).expand(N, -1)
        if self.sdf is not None:
            if getattr(self, '_no_normal_lift', False):
                normals = torch.zeros_like(normals)   # ablation: drop n(x)
            return torch.cat([pts_leaf, normals, code], dim=1)
        else:
            return torch.cat([pts_leaf, code], dim=1)

    # ---------------- value-only forward (used for L_src) ----------------
    def _forward_value(self, pts, code_value):
        pts_det = pts.detach()
        if self.sdf is not None:
            n_det = self.sdf.normal(pts_det).detach()
        else:
            raise NotImplementedError(
                "Source evaluation without SDF is not supported.")
        x_full = self._make_input(pts_det, n_det, code_value)
        return self.model_u(x_full).squeeze(-1)

    # ----------- value + gradient (used for L_dir and L_mass) -----------
    def _forward_value_grad(self, pts, code_value, normals_detached=None,
                            subtract_normal=True):
        """Return (u, g, n) at ``pts``.

        SDF branch (oracle with a ``vertex_sdf`` attribute; the branch used
        with SparseGridFDM_SDF):  g = du/dx + J_n^T du/dn, the chain-rule
        gradient of u(x, n(x)) through the SDF normal.  g is returned
        unprojected and ``subtract_normal`` is ignored.

        Fallback branch (any other oracle, or mesh-only with
        ``normals_detached``):  g is the autograd gradient with respect to
        x, with its normal component removed when ``subtract_normal`` is True.
        """
        if self.sdf is not None and hasattr(self.sdf, 'vertex_sdf'):
            pts_det = pts.detach()
            n_det = self.sdf.normal(pts_det).detach()
            Jn    = self.sdf.normal_jacobian(pts_det).detach()

            pts_leaf = pts_det.requires_grad_(True)
            n_leaf   = n_det.clone().requires_grad_(True)

            x_full = self._make_input(pts_leaf, n_leaf, code_value)
            f_val = self.model_u(x_full)
            dphi_dx = grad(f_val.sum(), pts_leaf, create_graph=True,
                           retain_graph=True)[0]
            dphi_dn = grad(f_val.sum(), n_leaf,   create_graph=True,
                           retain_graph=True, allow_unused=True)[0]
            if dphi_dn is None:
                dphi_dn = torch.zeros_like(n_leaf)
            total_grad = dphi_dx + torch.bmm(
                Jn.transpose(1, 2), dphi_dn.unsqueeze(-1)).squeeze(-1)
            # The chain-rule gradient is used as is: it is NOT projected onto
            # the tangent plane and ``subtract_normal`` has no effect here.
            g = total_grad
            return f_val.squeeze(-1), g, n_det

        pts_leaf = pts.detach().requires_grad_(True)
        if self.sdf is not None:
            n = self.sdf.normal(pts_leaf)
        else:
            n = normals_detached.detach()
        x_full = self._make_input(pts_leaf, n, code_value)
        f_val = self.model_u(x_full)
        total_grad = grad(f_val.sum(), pts_leaf, create_graph=True,
                          retain_graph=True)[0]
        if subtract_normal:
            g = total_grad - (total_grad * n).sum(-1, keepdim=True) * n
        else:
            g = total_grad
        n_out = n.detach() if self.sdf is not None else n
        return f_val.squeeze(-1), g, n_out

    # ----------- CPM loss -----------
    def _cpm_loss(self, pts, normals, code_value):
        """Per-point closest-point-extension penalty, shape (N,).

        L_dn = (du/dx . n)^2 with the given ``normals`` held fixed, plus (with
        an SDF) L_fc = (u(x) - u(cp(x)))^2, where cp(x) is the SDF closest
        point, evaluated with the SDF normal at cp(x).
        """
        pts_leaf = pts.detach().requires_grad_(True)
        n_det = normals.detach()
        x_full = self._make_input(pts_leaf, n_det, code_value)
        f_val = self.model_u(x_full).squeeze(-1)
        partial_grad = grad(f_val.sum(), pts_leaf, create_graph=True)[0]
        L_dn = (partial_grad * n_det).sum(-1) ** 2

        if self.sdf is None:
            return L_dn

        cp = self.sdf.closest_point(pts_leaf.detach()).detach()
        n_cp = self.sdf.normal(cp).detach()
        x_cp = self._make_input(cp, n_cp, code_value)
        f_cp = self.model_u(x_cp).squeeze(-1)
        L_fc = (f_val - f_cp) ** 2
        return L_dn + L_fc

    # ------------------------------------------------------------
    # Sources: list-of-dict spec
    # ------------------------------------------------------------
    def set_sources(self, mesh, sources: Sequence[Dict[str, Any]]):
        """Configure multi-source Dirichlet data.

        ``mesh`` must provide ``vertices_np`` and ``faces_np``.  Each entry
        in `sources` is a dict with keys:

            type    : 'vertex' | 'disk_curve' | 'curve'    (required)
            value   : float, target value a_k              (default 1.0)
            weight  : float, lambda_k                      (default 1.0)
            value_generalize : bool, default False.  If True, this source's
                         target is its independently sampled code component.
            value_range : (value_min, value_max), optional and defaulting to
                         (-2, 2) when value_generalize=True.  All marked
                         sources must use the same range.

          For 'vertex':
            ids       : list[int], mesh vertex IDs forming this source
            patch_radius : float, optional small Dijkstra radius (mesh-units)
                          to fatten the point into a small patch: the ids
                          are replaced by the sorted set of all vertices
                          within that distance of any listed vertex.  If 0
                          or omitted, only the listed vertices are used.

          For 'disk_curve':
            seed       : int, mesh vertex ID at the disk center
            radius     : float, Dijkstra radius of the iso-contour (default
                         0.1).  Not used for sampling when generalize=True,
                         where the per-epoch code is the radius.
            generalize : bool, default False.  If True, this disk's radius
                         becomes a per-epoch random code in
                         [r_range[0], r_range[1]] fed to the network.
            r_range    : (r_min, r_max), required if generalize=True.

          For 'curve':
            points : (K, 3) array of polyline vertices, K >= 2
            closed : bool, default False

        Dijkstra radii are graph distances in which every interior edge counts
        twice (see the module docstring).  Value-generalized sources receive
        ``value_code_index`` = their rank among the marked sources in source
        order; an index supplied in the input is discarded and re-derived.
        """
        self._sources = []
        self._disk = None
        self._generalization_mode = 'none'
        self._code_range = (0.0, 0.0)
        self._code_source_mapping = {}
        self._has_generalization = False
        self._r_range = (0.0, 0.0)
        any_disk = any(s.get('type') == 'disk_curve' for s in sources)
        if any_disk:
            self._disk = _GeodesicDiskSource(mesh)

        # Every value-generalized source gets its own component, but all value
        # components use the same range.  Radius mode retains one shared code.
        radius_ranges: List[Tuple[float, float]] = []
        value_ranges: List[Tuple[float, float]] = []
        radius_source_indices: List[int] = []
        value_source_indices: List[int] = []

        print(f"  Sources ({len(sources)}):")
        for k, raw in enumerate(sources):
            s = dict(raw)  # copy
            stype = s.get('type')
            if stype not in ('vertex', 'disk_curve', 'curve'):
                raise ValueError(f"source[{k}]: unknown type {stype!r}")
            s['value']  = float(s.get('value', 1.0))
            s['weight'] = float(s.get('weight', 1.0))
            s['value_generalize'] = bool(s.get('value_generalize', False))
            # Always derive this mapping from source order; never trust a stale
            # value copied from a checkpoint or hand-written source spec.
            s.pop('value_code_index', None)
            if s['value_generalize']:
                vr_raw = s.get('value_range', (-2.0, 2.0))
                if not isinstance(vr_raw, (list, tuple)) or len(vr_raw) != 2:
                    raise ValueError(
                        f"source[{k}]: value_range must contain exactly two values")
                vr = (float(vr_raw[0]), float(vr_raw[1]))
                if not np.isfinite(vr).all() or vr[0] >= vr[1]:
                    raise ValueError(
                        f"source[{k}]: invalid value_range {vr}; expected min < max")
                s['value_range'] = vr
                s['value_code_index'] = len(value_source_indices)
                value_source_indices.append(k)
                value_ranges.append(vr)

            if stype == 'vertex':
                ids = list(map(int, s.get('ids', [])))
                if len(ids) == 0:
                    raise ValueError(f"source[{k}] vertex: empty ids")
                max_v = len(mesh.vertices_np) - 1
                bad = [v for v in ids if not (0 <= v <= max_v)]
                if bad:
                    raise ValueError(f"source[{k}] vertex ids out of range: {bad}")

                patch_radius = float(s.get('patch_radius', 0.0))
                if patch_radius > 0:
                    if self._disk is None:
                        self._disk = _GeodesicDiskSource(mesh)
                    expanded = set()
                    for v in ids:
                        ring = self._disk.k_ring(v, patch_radius)
                        expanded.update(ring.tolist())
                    ids = sorted(expanded)
                s['ids'] = np.asarray(ids, dtype=np.int64)
                s['vertex_xyz'] = mesh.vertices_np[s['ids']].astype(np.float32)
                print(f"    [{k}] vertex   value={s['value']:+.3f} "
                      f"weight={s['weight']:.2f}  "
                      f"|ids|={len(s['ids'])}  "
                      f"patch_radius={patch_radius:.4f}")

            elif stype == 'disk_curve':
                if 'seed' not in s:
                    raise ValueError(f"source[{k}] disk_curve: missing 'seed'")
                s['seed']   = int(s['seed'])
                s['radius'] = float(s.get('radius', 0.1))
                s['generalize'] = bool(s.get('generalize', False))
                if s['generalize']:
                    if 'r_range' not in s:
                        raise ValueError(
                            f"source[{k}] disk_curve: generalize=True needs r_range")
                    rr = (float(s['r_range'][0]), float(s['r_range'][1]))
                    if not np.isfinite(rr).all() or rr[0] > rr[1]:
                        raise ValueError(
                            f"source[{k}] disk_curve: invalid r_range {rr}; "
                            "expected min <= max")
                    s['r_range'] = rr
                    radius_source_indices.append(k)
                    radius_ranges.append(rr)
                # warm Dijkstra cache
                self._disk.dist_from(s['seed'])
                # also pre-extract default radius for non-generalizing case
                if not s['generalize']:
                    p1, p2, _ = self._disk.extract(s['seed'], s['radius'])
                    L = np.linalg.norm(p2 - p1, axis=1).sum() if len(p1) > 0 else 0.0
                    print(f"    [{k}] disk     value={s['value']:+.3f} "
                          f"weight={s['weight']:.2f}  "
                          f"seed={s['seed']} r={s['radius']:.4f}  "
                          f"{len(p1)} segs, perimeter~{L:.4f}")
                else:
                    print(f"    [{k}] disk     value={s['value']:+.3f} "
                          f"weight={s['weight']:.2f}  "
                          f"seed={s['seed']} GENERALIZE r~U{s['r_range']}")

            elif stype == 'curve':
                pts = np.asarray(s.get('points'), dtype=np.float32)
                if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 2:
                    raise ValueError(
                        f"source[{k}] curve: points must be (K,3) with K>=2")
                s['points'] = pts
                s['closed'] = bool(s.get('closed', False))
                # length for printing
                if s['closed']:
                    L = np.linalg.norm(np.roll(pts, -1, 0) - pts, axis=1).sum()
                else:
                    L = np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()
                print(f"    [{k}] curve    value={s['value']:+.3f} "
                      f"weight={s['weight']:.2f}  "
                      f"K={len(pts)}  length~{L:.4f}  closed={s['closed']}")

            if s['value_generalize']:
                print(f"         value GENERALIZE g~U{s['value_range']}")
            self._sources.append(s)

        # validate generalization range coherence
        if radius_ranges and value_ranges:
            raise ValueError(
                "Radius and Dirichlet-value generalization cannot be enabled "
                "together: they use distinct conditioning semantics")

        active_ranges = radius_ranges or value_ranges
        if active_ranges:
            ref = active_ranges[0]
            for rr in active_ranges[1:]:
                if abs(rr[0] - ref[0]) > 1e-12 or abs(rr[1] - ref[1]) > 1e-12:
                    kind = 'r_range' if radius_ranges else 'value_range'
                    raise ValueError(
                        f"All sources generalized by {kind} must share the "
                        f"same range; got {active_ranges}")
            self._generalization_mode = 'radius' if radius_ranges else 'value'
            self._code_range = ref
            self._has_generalization = True
            if self._generalization_mode == 'value':
                self._code_source_mapping = {
                    source_index: code_index
                    for code_index, source_index in enumerate(value_source_indices)
                }
                expected_code_size = len(value_source_indices)
            else:
                self._code_source_mapping = {
                    source_index: 0 for source_index in radius_source_indices
                }
                expected_code_size = 1
                self._r_range = ref
            if self.code_size != expected_code_size:
                raise ValueError(
                    f"model code_size={self.code_size}, but mode "
                    f"{self._generalization_mode!r} requires {expected_code_size}")
            print(f"  Generalization ON: mode={self._generalization_mode}, "
                  f"code_size={self.code_size}, shared code_range={ref}, "
                  f"source_to_code={self._code_source_mapping}")
        elif self.code_size != 1:
            raise ValueError(
                f"non-generalized mode requires legacy code_size=1, "
                f"got {self.code_size}")

    def _source_target(self, src, code_val):
        """Resolve a source's target for the current conditioning code."""
        if src.get('value_generalize', False):
            code_index = int(src['value_code_index'])
            return float(self._code_array(code_val)[code_index])
        return float(src['value'])

    def _sample_one_source(self, src, num_per_source, code_val):
        """Returns (pts: (N,3) numpy, value scalar, weight scalar)."""
        rng = np.random
        stype = src['type']
        target_value = self._source_target(src, code_val)
        if stype == 'vertex':
            ids = src['ids']
            xyz = src['vertex_xyz']
            if len(ids) == 0:
                pts = np.zeros((0, 3), dtype=np.float32)
            else:
                idx = rng.randint(0, len(ids), num_per_source)
                pts = xyz[idx]
            return pts, target_value, src['weight']

        if stype == 'disk_curve':
            seed = src['seed']
            if src['generalize']:
                r_use = float(self._code_array(code_val)[0])
                use_cache = False
            else:
                r_use = src['radius']
                use_cache = True
            p1, p2, _ = self._disk.extract(seed, r_use, use_cache=use_cache)
            if len(p1) == 0:
                pts = np.zeros((0, 3), dtype=np.float32)
            else:
                pts = _sample_on_segments(p1, p2, num_per_source, rng=rng)
            return pts, target_value, src['weight']

        if stype == 'curve':
            pts = _sample_polyline(src['points'], num_per_source,
                                   closed=src['closed'], rng=rng)
            return pts, target_value, src['weight']

        raise RuntimeError(f"unknown source type {stype!r}")

    def _sample_sources(self, num_per_source, code_val):
        """Returns (pts, values, weights, src_lens):
            pts (N,3), values (N,), weights (N,) torch tensors over all
            N_total = sum(src_lens) samples, and src_lens, the list of sample
            counts per source (0 for a source that produced no samples).
        Each source contributes up to num_per_source samples.  With an SDF the
        points are projected by its closest-point map.  Source-wise
        averaging happens later (at the loss site): the per-source mean is
        taken inside `train()`, which reads lambda_k from the source dict;
        the per-sample weights returned here are not used there.
        """
        all_pts, all_v, all_w, src_lens = [], [], [], []
        for src in self._sources:
            pts_np, v, w = self._sample_one_source(src, num_per_source, code_val)
            if len(pts_np) == 0:
                src_lens.append(0)
                continue
            all_pts.append(pts_np)
            all_v.append(np.full(len(pts_np), v, dtype=np.float32))
            all_w.append(np.full(len(pts_np), w, dtype=np.float32))
            src_lens.append(len(pts_np))

        if not all_pts:
            raise RuntimeError("All sources produced 0 samples this epoch.")

        pts_np = np.concatenate(all_pts, axis=0)
        v_np   = np.concatenate(all_v, axis=0)
        w_np   = np.concatenate(all_w, axis=0)

        pts_t = torch.from_numpy(pts_np).to(self.device)
        if self.sdf is not None:
            pts_t = self.sdf.closest_point(pts_t).detach()
        v_t = torch.from_numpy(v_np).to(self.device)
        w_t = torch.from_numpy(w_np).to(self.device)
        return pts_t, v_t, w_t, src_lens

    def _value_range(self) -> Tuple[float, float]:
        """Min/max of all source target values, used by L_range."""
        if not self._sources:
            return 0.0, 1.0
        vals = []
        for src in self._sources:
            if src.get('value_generalize', False):
                vals.extend(src.get('value_range', self._code_range))
            else:
                vals.append(src['value'])
        return float(min(vals)), float(max(vals))

    def _epoch_code_value(self):
        """Sample conditioning values for one epoch.

        Value mode samples every source component independently.  Legacy
        radius mode continues to use a single scalar shared by its disks.
        """
        if self._generalization_mode == 'value':
            code_min, code_max = self._code_range
            return np.random.uniform(
                code_min, code_max, size=self.code_size).astype(np.float64)
        if self._generalization_mode == 'radius':
            code_min, code_max = self._code_range
            return float(np.random.uniform(code_min, code_max))
        return 0.0

    def _format_code(self, code_value) -> str:
        """Short printable form of a conditioning code (3 decimals)."""
        code = self._code_array(code_value)
        if len(code) == 1:
            return f"{code[0]:.3f}"
        return np.array2string(code, precision=3, separator=',')

    # ------------------------------------------------------------
    # Train
    # ------------------------------------------------------------
    def train(
        self,
        mesh,
        sources: Sequence[Dict[str, Any]],
        # epoch config
        num_epochs: int = 3000,
        num_surface: int = 10000,
        num_narrowband: int = 10000,
        num_per_source: int = 256,
        band_width: float = 0.01,
        # loss weights
        lambda_dir:   float = 1.0,
        lambda_mass:  float = 1.0,
        lambda_cpm:   float = 10.0,
        use_cpm:      bool  = True,                 # master switch of the CPM term
        no_normal_lift: bool = False,
        lambda_range: float = 0.0,    # soft clamp of u to [v_min, v_max]
        clip_grad: float = 1.0,
        log_every: int = 100,
        save_every: int = 0,
        save_dir: Optional[str] = None,
        train_codes_path: Optional[str] = None,
    ):
        """Train the field on ``mesh`` for the given list of source dicts.

        ``mesh`` must provide ``vertices_np``, ``faces_np`` and
        ``sample_surface(N, device) -> (points, normals)`` (MeshGeometry).
        ``sources`` is parsed by ``set_sources`` (see there for the fields).

        Each epoch draws the conditioning code, ``num_per_source`` samples on
        every source, ``num_surface`` surface samples and ``num_narrowband``
        narrow-band samples (surface samples plus Gaussian noise of standard
        deviation ``band_width``), and takes one Adam step on

            L = L_src + lambda_dir * L_dir + lambda_mass * L_mass
                + lambda_cpm * L_cpm + lambda_range * L_range

        (terms as described in the module docstring).  The learning rate
        follows a cosine schedule down to 1e-5 over ``num_epochs`` and the
        gradient norm is clipped to ``clip_grad``.

        use_cpm        : False drops the term L_cpm.  The flag is also passed
                         on as ``subtract_normal``, which has no effect on
                         L_dir when the SDF chain-rule gradient is used.
        no_normal_lift : ablation, feed zeros instead of the normal n(x).
        lambda_range   : weight of the soft clamp of u to the range of the
                         source targets; 0 disables it.
        log_every      : print a log line every ``log_every`` epochs (and at
                         epoch 1); values < 1 are treated as 1.
        save_every, save_dir : when both are set, write
                         <save_dir>/diffusion_epoch_<epoch>.pt every
                         ``save_every`` epochs.
        train_codes_path : if set, write the code of every epoch to this TSV.
                         Header: "epoch" followed by one "source_<i>" column
                         per code component in value mode (i is the code
                         component, i.e. the i-th value-generalized source),
                         otherwise "epoch" and "code".  Values use %.17g.

        An SDF oracle is required: with ``sdf=None`` the evaluation of the
        source term raises NotImplementedError.  Returns ``self``.
        """
        print(f"use_cpm: {use_cpm}, no_normal_lift: {no_normal_lift}")
        self._no_normal_lift = bool(no_normal_lift)
        self.set_sources(mesh, sources)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        codes_file = None
        if train_codes_path:
            parent = os.path.dirname(os.path.abspath(train_codes_path))
            if parent:
                os.makedirs(parent, exist_ok=True)
            codes_file = open(train_codes_path, "w", encoding="utf-8")
            if self._generalization_mode == "value":
                hdr = "epoch" + "".join(
                    f"\tsource_{i}" for i in range(self.code_size)
                )
            else:
                hdr = "epoch\tcode"
            codes_file.write(hdr + "\n")
            print(f"  Logging per-epoch codes -> {train_codes_path}")

        v_min, v_max = self._value_range()

        dev = self.device
        self.model_u.train()
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=num_epochs, eta_min=1e-5)

        print(f"\nLosses (multi-source screened-Dirichlet diffusion):")
        print(f"  lambda_dir={lambda_dir}  lambda_mass={lambda_mass}  "
              f"lambda_cpm={lambda_cpm}  lambda_range={lambda_range}")
        print(f"  Per-source weights/values:")
        for k, s in enumerate(self._sources):
            if s.get('value_generalize', False):
                target_desc = f"a=g in {tuple(s['value_range'])}"
            else:
                target_desc = f"a={s['value']:+.3f}"
            print(f"    [{k}] {s['type']:>10s}  {target_desc}  "
                  f"lambda={s['weight']:.3f}")
        print(f"  Source value range:  [{v_min}, {v_max}]")
        print(f"  Influence length scale  l ~ sqrt(lambda_dir/lambda_mass) = "
              f"{(lambda_dir / max(lambda_mass, 1e-12)) ** 0.5:.4f}")

        for epoch in range(1, num_epochs + 1):
            # ---- code value for this epoch ----
            code_val = self._epoch_code_value()
            if codes_file is not None:
                code = np.atleast_1d(self._code_array(code_val))
                codes_file.write(
                    f"{epoch}\t" + "\t".join(f"{float(c):.17g}" for c in code) + "\n"
                )
                if epoch % max(log_every, 1) == 0:
                    codes_file.flush()

            # ---- multi-source samples ----
            src_pts, src_vals, src_w, src_lens = self._sample_sources(
                num_per_source, code_val)

            # ---- surface samples ----
            surf_pts, surf_n = mesh.sample_surface(N=num_surface, device=dev)
            surf_n = surf_n.float().detach()

            # ---- narrow-band samples ----
            surf_nb, _ = mesh.sample_surface(num_narrowband, device=dev)
            nb_pts = surf_nb + torch.randn_like(surf_nb) * band_width
            if self.sdf is not None:
                nb_cp = self.sdf.closest_point(nb_pts.detach()).detach()
                nb_n  = self.sdf.normal(nb_cp).detach()
            else:
                nb_n = surf_n

            # ---- u and grad u on the surface samples ----
            # With an SDF oracle g_s is the chain-rule gradient
            # du/dx + J_n^T du/dn.  It is not projected onto the tangent plane
            # and subtract_normal (= use_cpm) has no effect in that branch.
            u_s, g_s, _ = self._forward_value_grad(
                surf_pts, code_val, surf_n,
                subtract_normal=use_cpm)

            L_dir  = (g_s ** 2).sum(-1).mean()
            L_mass = (u_s ** 2).mean()

            # ---- L_src: weighted source-wise means, averaged over sources ----
            u_src = self._forward_value(src_pts, code_val)        # (N_total,)
            per_pt_sq = (u_src - src_vals) ** 2                   # (N_total,)
            # split by source for source-wise averaging (avoids long sources
            # dominating short ones); each source contributes its own
            # lambda_k once.
            offsets = np.cumsum([0] + list(src_lens))
            L_src_terms = []
            for k, n_k in enumerate(src_lens):
                if n_k == 0:
                    continue
                a, b = int(offsets[k]), int(offsets[k] + n_k)
                lam_k = float(self._sources[k]['weight'])
                L_src_terms.append(lam_k * per_pt_sq[a:b].mean())
            if L_src_terms:
                # mean over sources:
                #   L_src = (1/m) * sum_k lambda_k * mean_{S_k}(u - a_k)^2,
                # where m is the number of sources that produced samples in
                # this epoch.  The sum IS divided by m; fem_screen.py mirrors
                # this normalisation with its source scale |M|/m.
                L_src = torch.stack(L_src_terms).sum() / max(len(L_src_terms), 1)
            else:
                L_src = torch.tensor(0.0, device=dev)

            # ---- L_cpm (skipped entirely when use_cpm=False) ----
            if use_cpm:
                L_cpm_nb   = self._cpm_loss(nb_pts, nb_n, code_val)
                L_cpm_surf = self._cpm_loss(surf_pts, surf_n, code_val)
                L_cpm = torch.cat([L_cpm_nb, L_cpm_surf], dim=0).mean()
            else:
                L_cpm = torch.tensor(0.0, device=dev)

            # ---- L_range: soft clamp to [v_min, v_max] (auto from sources) ----
            if lambda_range > 0:
                L_range = (F.relu(v_min - u_s).pow(2).mean()
                           + F.relu(u_s - v_max).pow(2).mean())
            else:
                L_range = torch.tensor(0.0, device=dev)

            loss = (L_src
                    + lambda_dir   * L_dir
                    + lambda_mass  * L_mass
                    + lambda_cpm   * L_cpm
                    + lambda_range * L_range)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model_u.parameters(), clip_grad)

            self.optimizer.step()
            scheduler.step()

            # ---- log ----
            if epoch % max(log_every, 1) == 0 or epoch == 1:
                with torch.no_grad():
                    gn_mean = g_s.norm(dim=-1).mean().item()
                    u_min_v = u_s.min().item()
                    u_max_v = u_s.max().item()
                    u_mean  = u_s.mean().item()
                    # mean predicted value per source (helpful diagnostic)
                    src_means = []
                    offsets = np.cumsum([0] + list(src_lens))
                    for k, n_k in enumerate(src_lens):
                        if n_k == 0:
                            src_means.append("--")
                        else:
                            a, b = int(offsets[k]), int(offsets[k] + n_k)
                            src_means.append(f"{u_src[a:b].mean().item():+.3f}")
                tag = (f"g={self._format_code(code_val)} "
                       if self._has_generalization else "")
                epoch_targets = [self._source_target(s, code_val)
                                 for s in self._sources]
                src_str = "/".join([
                    f"{a}{m}" for a, m in zip(
                        [f"{a:+.2f}" for a in epoch_targets],
                        [f"->{m}" for m in src_means])
                ])
                print(
                    f"  [{epoch:>5}/{num_epochs}]  {tag}"
                    f"loss={loss.item():.4f}  "
                    f"src={L_src.item():.5f} [{src_str}]  "
                    f"dir={L_dir.item():.5f}  "
                    f"mass={L_mass.item():.5f}  "
                    f"cpm={L_cpm.item():.6f}  "
                    f"|grad u|~{gn_mean:.3f}  "
                    f"u in [{u_min_v:.3f},{u_max_v:.3f}](<u>={u_mean:.3f})  "
                    f"lr={scheduler.get_last_lr()[0]:.6f}"
                )

            if save_dir and save_every > 0 and epoch % save_every == 0:
                self.save(os.path.join(save_dir, f"diffusion_epoch_{epoch}.pt"))

        if codes_file is not None:
            codes_file.close()
            print(f"Wrote train codes: {train_codes_path}")

        print("\nTraining complete.")
        return self

    # ------------------------------------------------------------
    # Export: per-vertex u and |grad_M u|
    # ------------------------------------------------------------
    def export(self, mesh, save_path, batch_size=8192,
               code_value: Optional[Any] = None):
        """Evaluate the field at the mesh vertices and write an npz file.

        code_value : conditioning code used for the evaluation.  Default: the
            midpoint of the code range (one midpoint per component in value
            mode), or 0 when nothing is generalized.

        u is evaluated at the mesh vertices with the SDF normal (the mesh
        vertex normals when ``sdf`` is None).  ``grad_mag`` is the norm of
        the tangentially projected gradient, so it is not the (unprojected)
        integrand of L_dir.  The npz keys are listed in the module docstring;
        the same dict is returned.

        The source markers are built from the state prepared by
        ``set_sources`` (called by ``train``), so this method is meant to be
        called after training; that state is not restored by ``load``.
        """
        if code_value is None:
            if self._generalization_mode == 'value':
                midpoint = 0.5 * (self._code_range[0] + self._code_range[1])
                code_value = np.full(self.code_size, midpoint, dtype=np.float64)
            elif self._generalization_mode == 'radius':
                code_value = 0.5 * (self._code_range[0] + self._code_range[1])
            else:
                code_value = 0.0
        code_values = self._code_array(code_value)
        self.model_u.eval()
        dev = self.device
        V = mesh.vertices_np.astype(np.float32)
        Nv = len(V)
        u_all    = np.empty(Nv, dtype=np.float32)
        gmag_all = np.empty(Nv, dtype=np.float32)

        for s in range(0, Nv, batch_size):
            e = min(s + batch_size, Nv)
            pts = torch.from_numpy(V[s:e]).to(dev)
            if self.sdf is not None:
                n = self.sdf.normal(pts).detach()
            else:
                n = torch.from_numpy(
                    mesh.vertex_normals_np[s:e].astype(np.float32)).to(dev)
            with torch.no_grad():
                x_full = self._make_input(pts, n, code_values)
                u_b = self.model_u(x_full).squeeze(-1)
            u_all[s:e] = u_b.cpu().numpy()
            with torch.enable_grad():
                pts_leaf = pts.detach().requires_grad_(True)
                if self.sdf is not None:
                    n_leaf = n.clone().requires_grad_(True)
                    x_full = self._make_input(pts_leaf, n_leaf, code_values)
                else:
                    x_full = self._make_input(pts_leaf, n, code_values)
                f = self.model_u(x_full)
                dphi_dx = torch.autograd.grad(f.sum(), pts_leaf,
                                              retain_graph=True)[0]
                if self.sdf is not None:
                    dphi_dn = torch.autograd.grad(
                        f.sum(), n_leaf, allow_unused=True)[0]
                    if dphi_dn is None:
                        dphi_dn = torch.zeros_like(n_leaf)
                    Jn = self.sdf.normal_jacobian(pts).detach()
                    total_grad = dphi_dx + torch.bmm(
                        Jn.transpose(1, 2), dphi_dn.unsqueeze(-1)
                    ).squeeze(-1)
                else:
                    total_grad = dphi_dx
                g_tan = total_grad - (total_grad * n).sum(-1, keepdim=True) * n
                gmag_all[s:e] = g_tan.norm(dim=-1).detach().cpu().numpy()

        # ---- collect viz primitives per source ----
        v_min, v_max = self._value_range()
        viz_pts_list  = []     # marker points (vertex sources, sampled curve points)
        viz_pts_vals  = []
        viz_seg_p1    = []     # disk_curve / curve segments
        viz_seg_p2    = []
        viz_seg_vals  = []     # one value per segment

        for src in self._sources:
            # Value-generalized source markers must match the field code used
            # for this export, rather than displaying the fallback value.
            v = self._source_target(src, code_values)
            if src['type'] == 'vertex':
                pts = src['vertex_xyz']
                viz_pts_list.append(pts)
                viz_pts_vals.append(np.full(len(pts), v, dtype=np.float32))

            elif src['type'] == 'disk_curve':
                seed = src['seed']
                if src['generalize']:
                    r_use = float(code_values[0])
                else:
                    r_use = src['radius']
                p1, p2, _ = self._disk.extract(
                    seed, r_use, use_cache=(not src['generalize']))
                if len(p1) > 0:
                    viz_seg_p1.append(p1)
                    viz_seg_p2.append(p2)
                    viz_seg_vals.append(np.full(len(p1), v, dtype=np.float32))

            elif src['type'] == 'curve':
                pts = src['points']
                if src['closed']:
                    p1 = pts; p2 = np.roll(pts, -1, axis=0)
                else:
                    p1 = pts[:-1]; p2 = pts[1:]
                viz_seg_p1.append(p1.astype(np.float32))
                viz_seg_p2.append(p2.astype(np.float32))
                viz_seg_vals.append(np.full(len(p1), v, dtype=np.float32))

        viz_pts  = (np.concatenate(viz_pts_list, axis=0)
                    if viz_pts_list  else np.zeros((0, 3), dtype=np.float32))
        viz_pts_v = (np.concatenate(viz_pts_vals, axis=0)
                    if viz_pts_vals else np.zeros((0,), dtype=np.float32))
        seg_p1   = (np.concatenate(viz_seg_p1,  axis=0)
                    if viz_seg_p1   else np.zeros((0, 3), dtype=np.float32))
        seg_p2   = (np.concatenate(viz_seg_p2,  axis=0)
                    if viz_seg_p2   else np.zeros((0, 3), dtype=np.float32))
        seg_vals = (np.concatenate(viz_seg_vals, axis=0)
                    if viz_seg_vals else np.zeros((0,), dtype=np.float32))

        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        legacy_code_value = (np.float32(code_values[0])
                             if self.code_size == 1
                             else code_values.astype(np.float32))
        value_code_sources = [
            source_index for source_index, _ in sorted(
                self._code_source_mapping.items(), key=lambda item: item[1])
        ] if self._generalization_mode == 'value' else []
        result = {
            'mesh_vertices':  V,
            'mesh_faces':     mesh.faces_np,
            'u_vertex':       u_all,
            'grad_mag':       gmag_all,
            'code_value':     legacy_code_value,
            'code_values':    code_values.astype(np.float32),
            'code_size':      np.int64(self.code_size),
            'value_code_source_indices': np.asarray(
                value_code_sources, dtype=np.int64),
            # source info for visualization
            'value_min':      np.float32(v_min),
            'value_max':      np.float32(v_max),
            'src_pts':        viz_pts,
            'src_pts_value':  viz_pts_v,
            'src_seg_p1':     seg_p1,
            'src_seg_p2':     seg_p2,
            'src_seg_value':  seg_vals,
            'has_generalization': bool(self._has_generalization),
            'r_range':        np.asarray(self._r_range, dtype=np.float32),
            'generalization_mode': np.asarray(self._generalization_mode),
            'code_range':     np.asarray(self._code_range, dtype=np.float32),
            'no_normal_lift': bool(self._no_normal_lift),
        }
        np.savez(save_path, **result)
        print(f"Exported -> {save_path}")
        print(f"  u in [{u_all.min():.4f}, {u_all.max():.4f}]  "
              f"<u>={u_all.mean():.4f}  "
              f"mean |grad_M u|={gmag_all.mean():.4f}")
        return result

    # ------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------
    def save(self, path):
        """Write a checkpoint (network, Adam state, run metadata) to ``path``.

        Keys: model_u, optimizer, input_size, geo_size, code_size, hidden, lr,
        sources, generalization_mode, code_range, no_normal_lift,
        has_generalization, r_range (see the module docstring).
        """
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        # source dicts contain numpy arrays (vertex ids, curve points): they
        # are converted to plain lists and the cached vertex coordinates
        # ('vertex_xyz') are dropped.  The stored vertex ids are the
        # patch-expanded ids.
        sources_dump = []
        for s in self._sources:
            d = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                 for k, v in s.items() if k != 'vertex_xyz'}
            sources_dump.append(d)
        torch.save({
            'model_u':        self.model_u.state_dict(),
            'optimizer':      self.optimizer.state_dict(),
            'input_size':     self.input_size,
            'geo_size':       self.geo_size,
            'code_size':      self.code_size,
            'hidden':         self.hidden,
            'lr':             self.lr,
            'sources':        sources_dump,
            'generalization_mode': self._generalization_mode,
            'code_range':     list(self._code_range),
            'no_normal_lift': self._no_normal_lift,
            # Legacy radius-generalization metadata retained so older tooling
            # can still inspect newly written radius checkpoints.
            'has_generalization': self._has_generalization,
            'r_range':        list(self._r_range),
        }, path)
        print(f"  saved -> {path}")

    @classmethod
    def load(cls, path, device='cpu', sdf=None):
        """Rebuild an object from a checkpoint written by ``save``.

        ``sdf`` must be given if and only if the checkpoint was trained with
        an SDF oracle (checked through geo_size).  The network weights, the
        Adam state and the generalization metadata are restored, which is
        what is needed to evaluate the field through ``_make_input`` and
        ``model_u``.  The source geometry is not rebuilt (see the comment in
        the body).

        The file is read with ``torch.load(weights_only=False)``, i.e. with
        full pickle: only load checkpoints you trust.
        """
        ckpt = torch.load(path, map_location=device, weights_only=False)
        mode = ckpt.get('generalization_mode')
        if mode is None:
            # Old checkpoints only knew disk-radius generalization.
            mode = ('radius'
                    if ckpt.get('has_generalization', False)
                    else 'none')
        if mode not in ('none', 'radius', 'value'):
            raise ValueError(
                f"Checkpoint has unknown generalization_mode {mode!r}")

        # All pre-vector checkpoints had exactly one scalar code.  New value
        # checkpoints declare their dynamic width explicitly.
        has_vector_metadata = 'code_size' in ckpt
        code_size = int(ckpt.get('code_size', 1))
        if code_size < 1:
            raise ValueError(f"Checkpoint has invalid code_size {code_size}")
        if mode in ('none', 'radius') and code_size != 1:
            raise ValueError(
                f"Checkpoint mode {mode!r} requires code_size=1, got {code_size}")

        expected_geo_size = (3 if sdf is not None else 0) + code_size
        saved_geo_size = int(ckpt.get('geo_size', expected_geo_size))
        if saved_geo_size != expected_geo_size:
            need = "with an SDF" if saved_geo_size == code_size + 3 else "without an SDF"
            raise ValueError(
                f"Checkpoint geo_size={saved_geo_size}, but the requested load "
                f"configuration implies geo_size={expected_geo_size}; load {need}")

        obj = cls(
            input_size=ckpt.get('input_size', 3),
            hidden=ckpt.get('hidden', 128),
            lr=ckpt.get('lr', 1e-3),
            device=device,
            sdf=sdf,
            code_size=code_size,
        )
        obj.model_u.load_state_dict(ckpt['model_u'])
        obj.optimizer.load_state_dict(ckpt['optimizer'])

        if 'code_range' in ckpt:
            code_range = tuple(map(float, ckpt['code_range']))
        elif mode == 'radius':
            code_range = tuple(map(float, ckpt.get('r_range', (0.0, 0.0))))
        else:
            code_range = (0.0, 0.0)
        if (len(code_range) != 2 or not np.isfinite(code_range).all()
                or code_range[0] > code_range[1]):
            raise ValueError(
                f"Checkpoint has invalid code_range {code_range!r}")

        obj._generalization_mode = mode
        obj._code_range = code_range
        obj._has_generalization = mode != 'none'
        obj._r_range = (code_range if mode == 'radius' else (0.0, 0.0))
        obj._no_normal_lift = bool(ckpt.get('no_normal_lift', False))
        # Keep the serialized specs available, but they are deliberately raw
        # (plain lists, no vertex coordinates, no geodesic-disk cache), so
        # export/sampling do not work on them.  To restore that state call
        # set_sources(mesh, sources) with the same list that was passed to
        # train().  For a train_screen.py run that list is more than the
        # source JSON: its entries with points_npy already loaded into
        # points, then the inline --vertex-ids / --disk-seed / --curve-npy
        # sources, plus the value_generalize / value_range markers added for
        # --value-generalize-source (train_screen.build_sources).
        # set_sources re-derives the generalization mode, code range and
        # source-to-code mapping from that list: without the markers it
        # raises ValueError for code_size > 1 and silently falls back to the
        # non-generalized mode for code_size == 1.  Do NOT pass obj._sources:
        # the saved vertex ids are already patch-expanded and still carry
        # patch_radius, so they would be expanded a second time.
        obj._sources = [dict(s) for s in ckpt.get('sources', [])]
        mapping: Dict[int, int] = {}
        if mode == 'value':
            marked = [
                (source_index, source)
                for source_index, source in enumerate(obj._sources)
                if bool(source.get('value_generalize', False))
            ]
            for source_index, source in marked:
                if 'value_code_index' in source:
                    code_index = int(source['value_code_index'])
                elif code_size == 1 and not has_vector_metadata:
                    # Legacy value checkpoints used one shared scalar.
                    code_index = 0
                    source['value_code_index'] = 0
                else:
                    raise ValueError(
                        f"Checkpoint source[{source_index}] is missing "
                        "value_code_index")
                if code_index < 0 or code_index >= code_size:
                    raise ValueError(
                        f"Checkpoint source[{source_index}] has invalid "
                        f"value_code_index={code_index} for code_size={code_size}")
                if code_size > 1 and code_index in mapping.values():
                    raise ValueError(
                        f"Checkpoint repeats value_code_index={code_index}")
                mapping[source_index] = code_index
            if (has_vector_metadata
                    and sorted(mapping.values()) != list(range(code_size))):
                raise ValueError(
                    "Checkpoint value_code_index values must be contiguous "
                    f"0..{code_size - 1}")
        elif mode == 'radius':
            mapping = {
                source_index: 0
                for source_index, source in enumerate(obj._sources)
                if source.get('type') == 'disk_curve'
                and bool(source.get('generalize', False))
            }
        obj._code_source_mapping = mapping
        print(f"Loaded from {path}")
        return obj


# ============================================================
# 4. Polyscope visualizer
# ============================================================

def visualize_polyscope(npz_path):
    """Interactive polyscope viewer for an npz written by ``export``.

    Shows u on the mesh (symmetric coolwarm map when the source targets span
    both signs, viridis otherwise), the gradient magnitude when present, and
    the source points / curves colored by their target value.  Requires the
    ``polyscope`` package, which is imported only here.
    """
    import polyscope as ps
    data = np.load(npz_path, allow_pickle=True)
    V = data['mesh_vertices']
    Faces = data['mesh_faces']
    u    = data['u_vertex']
    gmag = data['grad_mag'] if 'grad_mag' in data.files else None

    v_min = float(data['value_min']) if 'value_min' in data.files else 0.0
    v_max = float(data['value_max']) if 'value_max' in data.files else 1.0

    ps.init()
    ps.set_up_dir("z_up")
    ps.set_ground_plane_mode("shadow_only")

    sm = ps.register_surface_mesh("mesh", V, Faces, smooth_shade=True)

    # If sources span both signs, use a symmetric coolwarm map; else viridis.
    if v_min < 0 < v_max:
        vabs = float(max(abs(v_min), abs(v_max)))
        sm.add_scalar_quantity("u (signed diffusion)", u, cmap='coolwarm',
                               vminmax=(-vabs, vabs), enabled=True)
    else:
        sm.add_scalar_quantity("u (diffusion)", u, cmap='viridis',
                               vminmax=(v_min, v_max), enabled=True)

    sm.add_scalar_quantity("u (autoscale)", u, cmap='viridis', enabled=False)
    if gmag is not None:
        sm.add_scalar_quantity("|grad_M u|", gmag, cmap='blues', enabled=False)

    # vertex-source markers (colored by their target value)
    if 'src_pts' in data.files and len(data['src_pts']) > 0:
        pts = data['src_pts']
        vals = data['src_pts_value']
        pc = ps.register_point_cloud("source_points", pts, radius=0.006)
        if v_min < 0 < v_max:
            vabs = float(max(abs(v_min), abs(v_max)))
            pc.add_scalar_quantity("value", vals, cmap='coolwarm',
                                   vminmax=(-vabs, vabs), enabled=True)
        else:
            pc.add_scalar_quantity("value", vals, cmap='viridis',
                                   vminmax=(v_min, v_max), enabled=True)

    # curve / disk_curve segments (colored by per-segment target value)
    if 'src_seg_p1' in data.files and len(data['src_seg_p1']) > 0:
        p1 = data['src_seg_p1']; p2 = data['src_seg_p2']
        vals = data['src_seg_value']
        nodes = np.concatenate([p1, p2], axis=0)
        S = len(p1)
        edges = np.stack([np.arange(S), np.arange(S, 2 * S)], axis=1)
        cn = ps.register_curve_network("source_curves", nodes, edges,
                                       radius=0.003)
        # color one value per edge (= per p1 entry)
        if v_min < 0 < v_max:
            vabs = float(max(abs(v_min), abs(v_max)))
            cn.add_scalar_quantity("value", vals, defined_on='edges',
                                   cmap='coolwarm', vminmax=(-vabs, vabs),
                                   enabled=True)
        else:
            cn.add_scalar_quantity("value", vals, defined_on='edges',
                                   cmap='viridis', vminmax=(v_min, v_max),
                                   enabled=True)

    ps.show()


if __name__ == '__main__':
    import sys
    if '--vis' in sys.argv:
        visualize_polyscope(sys.argv[sys.argv.index('--vis') + 1])
    else:
        print("Usage:")
        print("  Train:     python train_screen.py <mesh.obj> [options]")
        print("  Visualize: python Surfacescreendiffusion.py "
              "--vis <out_dir>/diffusion.npz")
