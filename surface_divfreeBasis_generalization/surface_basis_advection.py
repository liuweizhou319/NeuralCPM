"""
surface_basis_advection.py - Density-field advection on a triangle-mesh surface
================================================================================

Loads a trained `SurfaceDFVecLaplacianBasis` checkpoint and runs a semi-Lagrangian
advection of a Gaussian density blob driven by an initial localized tangent flow.

Discretization
--------------
- Fixed quadrature points: mesh vertices p_i  (i = 1 ... N).
- Density rho_i lives on vertices.
- Velocity is represented by coefficients alpha (k,) over the learned basis:
        v(p_i) = sum_k phi_k(p_i) alpha_k

Per step, with backtrace at fixed p_i:
        v^t        = Phi_V @ alpha^t
        q_i        = closest_point( p_i - dt * v^t(p_i) )            # cp_M
        rho^{t+1}  = barycentric interp of rho^t at q_i               # I_M[rho^t](q)
        v_bar      = Phi(q_i) @ alpha^t                               # v_bar in T_{q_i}M
        v_til      = Rodrigues_transport( v_bar : n(q) -> n(p) )      # back to T_{p_i}M
        alpha^{t+1}= solve( Phi_V^T A Phi_V + ridge I, Phi_V^T A v_til )
        v^{t+1}    = Phi_V @ alpha^{t+1}

Mesh, vertex normals/areas, and Phi_V are precomputed once.

In this folder the file is used as a library: surface_basis_advection_time.py
imports its geometry helpers (vertex areas and normals, closest points,
barycentric coordinates, parallel transport) for the advection on a sequence
of shapes.  The command line below runs the advection on a single mesh with a
single-shape model.

Usage
-----
  python surface_basis_advection.py <mesh.obj> <model.pt> \\
      --center 0.0,0.5,0.0 --speed 1.5 --n-steps 1500 --dt 0.005 \\
      --out adv.npz --vis
"""

import argparse
import os
import numpy as np
import torch
from Mesh_geometry import MeshGeometry
from SparseGridFDM_SDF import (
    SparseGridFDM_SDF,
    _build_fcpw_scene,
    _fcpw_find_closest_points,
    _fcpw_make_interaction_list,
    _fcpw_make_gpu_interaction_list,
    _fcpw_interaction_positions,
)
from SurfaceDFVecLaplacianBasis import SurfaceDFVecLaplacianBasis


# ============================================================
# Mesh helpers
# ============================================================

def vertex_areas_barycentric(V, F):
    """1/3 of the sum of incident triangle areas. Returns (Nv,) numpy."""
    v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    fa = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)  # (Nf,)
    A = np.zeros(len(V), dtype=np.float64)
    for k in range(3):
        np.add.at(A, F[:, k], fa / 3.0)
    return A.astype(np.float32)


def vertex_normals_areaweighted(V, F):
    """Area-weighted vertex normals (no normalization done yet on faces). Returns (Nv,3)."""
    v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)  # area-weighted face normals
    N = np.zeros_like(V, dtype=np.float64)
    for k in range(3):
        np.add.at(N, F[:, k], fn)
    norms = np.linalg.norm(N, axis=1, keepdims=True)
    return (N / np.clip(norms, 1e-12, None)).astype(np.float32)

def points_to_barycentric_np(tri, p, eps=1e-12):
    """
    tri: (N, 3, 3)
    p  : (N, 3)

    Return barycentric coordinates of p w.r.t. tri.
    """
    a = tri[:, 0]
    b = tri[:, 1]
    c = tri[:, 2]

    v0 = b - a
    v1 = c - a
    v2 = p - a

    d00 = np.sum(v0 * v0, axis=1)
    d01 = np.sum(v0 * v1, axis=1)
    d11 = np.sum(v1 * v1, axis=1)
    d20 = np.sum(v2 * v0, axis=1)
    d21 = np.sum(v2 * v1, axis=1)

    denom = d00 * d11 - d01 * d01
    denom = np.where(np.abs(denom) < eps, eps, denom)

    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w

    bary = np.stack([u, v, w], axis=1).astype(np.float32)

    # FCPW closest point should be on the triangle, but this removes tiny numerical drift.
    bary = np.clip(bary, 0.0, 1.0)
    bary = bary / np.clip(bary.sum(axis=1, keepdims=True), eps, None)
    return bary.astype(np.float32)


def _is_fcpw_gpu_scene(scene):
    cname = scene.__class__.__name__
    return ("GPUScene" in cname) or cname.lower().startswith("gpu_scene")


def _make_fcpw_interactions_for_scene(scene):
    if _is_fcpw_gpu_scene(scene):
        return _fcpw_make_gpu_interaction_list()
    return _fcpw_make_interaction_list()


def _fcpw_interaction_primitive_indices(interactions):
    """
    Robustly extract closest primitive indices from FCPW interactions.

    For a single mesh object, primitiveIndex is exactly the triangle id.
    """
    # Some builds may expose a bulk getter.
    for name in (
        "get_primitive_indices",
        "get_primitive_indexes",
        "get_primitive_ids",
        "get_primitive_index",
    ):
        if hasattr(interactions, name):
            fidx = getattr(interactions, name)()
            return np.asarray(fidx, dtype=np.int64)

    # Fallback: iterate over individual Interaction objects.
    try:
        out = []
        for it in interactions:
            for attr in ("primitiveIndex", "primitive_index", "primitive_id", "primIndex"):
                if hasattr(it, attr):
                    out.append(int(getattr(it, attr)))
                    break
            else:
                raise AttributeError
        return np.asarray(out, dtype=np.int64)
    except Exception as e:
        raise RuntimeError(
            "Cannot extract FCPW primitive indices. "
            "Run `print(dir(interactions))` and `print(dir(interactions[0]))` "
            "to inspect your fcpw binding. Need primitiveIndex for barycentric interpolation."
        ) from e


def fcpw_closest_points_and_faces(scene, query_np):
    """
    FCPW closest-point query.

    Returns
    -------
    q    : (N, 3) float32 closest points
    fidx : (N,) int64 closest triangle ids
    """
    query_np = np.ascontiguousarray(query_np.astype(np.float32))
    sqr_max = np.full((len(query_np),), np.inf, dtype=np.float32)

    interactions = _make_fcpw_interactions_for_scene(scene)

    _fcpw_find_closest_points(
        scene,
        query_np,
        sqr_max,
        interactions,
        want_normals=True,
    )

    q = _fcpw_interaction_positions(interactions).astype(np.float32)
    fidx = _fcpw_interaction_primitive_indices(interactions).astype(np.int64)

    if len(q) != len(query_np) or len(fidx) != len(query_np):
        raise RuntimeError(
            f"FCPW query returned inconsistent sizes: "
            f"query={len(query_np)}, q={len(q)}, fidx={len(fidx)}"
        )

    if np.any(fidx < 0):
        bad = int(np.sum(fidx < 0))
        raise RuntimeError(f"FCPW closest point failed for {bad} query points.")

    return q, fidx


def parallel_transport_min_rotation(v, n_from, n_to, eps=1e-6):
    """
    Transport tangent vector v from frame at n_from to frame at n_to via the
    minimum rotation that sends n_from -> n_to (Rodrigues' formula).
    A final tangent projection at n_to is applied for numerical safety.

    Shapes: v, n_from, n_to all (..., 3). Returns same shape.
    """
    axis = torch.cross(n_from, n_to, dim=-1)
    sin_t = torch.norm(axis, dim=-1, keepdim=True)              # (..., 1)
    cos_t = (n_from * n_to).sum(-1, keepdim=True)               # (..., 1)
    safe = sin_t > eps
    axis_n = torch.where(safe, axis / (sin_t + eps), torch.zeros_like(axis))
    cross_av = torch.cross(axis_n, v, dim=-1)
    dot_av = (axis_n * v).sum(-1, keepdim=True)
    v_rot = v * cos_t + cross_av * sin_t + axis_n * dot_av * (1.0 - cos_t)
    # Project onto tangent plane at n_to (kills any residual normal component).
    v_rot = v_rot - (v_rot * n_to).sum(-1, keepdim=True) * n_to
    return v_rot


# ============================================================
# Simulator
# ============================================================

class SurfaceBasisAdvection:

    def __init__(self, mesh_path, model_path, *,
                 normalize=True, code_value=0.5, ridge=1e-6,
                 use_sdf=True, sdf_res=500, sdf_padding=0.15, sdf_halo=2,
                 num_basis=None,                              # NEW
                 device=None):
        self.device = (torch.device(device) if device else
                       torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.code_value = float(code_value)
        self.ridge = float(ridge)

        # ---- Mesh ----
        self.mesh = MeshGeometry(mesh_path, normalize=normalize, device=self.device)
        self.V_np = self.mesh.vertices_np.astype(np.float32)
        self.F_np = self.mesh.faces_np.astype(np.int64)
        Nv, Nf = len(self.V_np), len(self.F_np)

        # ---- FCPW closest-point scene for semi-Lagrangian backtrace ----
        print("[init] Building FCPW closest-point scene ...")
        try:
            self._fcpw_scene = _build_fcpw_scene(
                self.V_np,
                self.F_np.astype(np.int32),
                use_gpu=torch.cuda.is_available(),
            )
            mode = "GPU" if _is_fcpw_gpu_scene(self._fcpw_scene) else "CPU"
            print(f"[init] FCPW scene ready ({mode})")
        except Exception as e:
            print(f"[init] FCPW GPU path failed: {e}")
            print("[init] Falling back to CPU FCPW scene")
            self._fcpw_scene = _build_fcpw_scene(
                self.V_np,
                self.F_np.astype(np.int32),
                use_gpu=False,
            )
            print("[init] FCPW scene ready (CPU)")

        # ---- SDF (must match training; same params!) ----
        self.sdf = None
        if use_sdf:
            print(f"[init] Building FDM SDF oracle (res={sdf_res}, pad={sdf_padding}) ...")
            self.sdf = SparseGridFDM_SDF.from_mesh(
                self.mesh, resolution=sdf_res, padding=sdf_padding,
                fd_halo=sdf_halo, device=self.device,
            )

        # ---- Basis ----
        self.basis = SurfaceDFVecLaplacianBasis.load(
            model_path, device=self.device, sdf=self.sdf
        )
        self.basis.set_eval()
        if num_basis is not None:                             # NEW
            assert 1 <= num_basis <= self.basis.k, \
                f"num_basis must be in [1, {self.basis.k}]"
            self.basis.k = int(num_basis)                     # truncate eval loop
        self.k = self.basis.k

        # ---- Vertex normals + areas ----
        V_t = torch.from_numpy(self.V_np).to(self.device)
        if self.sdf is not None:
            with torch.enable_grad():
                self.N_v = self.sdf.normal(V_t).detach()
        else:
            self.N_v = torch.from_numpy(
                vertex_normals_areaweighted(self.V_np, self.F_np)
            ).to(self.device)
        self.A_v = torch.from_numpy(
            vertex_areas_barycentric(self.V_np, self.F_np)
        ).to(self.device)
        self.V_t = V_t
        self.F_t = torch.from_numpy(self.F_np).to(self.device)

        print(f"[init] Mesh: {Nv} vertices, {Nf} faces, total area = {self.A_v.sum():.4f}")

        # ---- Precompute basis at vertices + Gram matrix ----
        print(f"[init] Precomputing basis at vertices (k={self.k}) ...")
        with torch.enable_grad():
            self.Phi_V = self.basis.eval_all_velocities(
                V_t, self.N_v, self.code_value, detach=True
            )  # (Nv, k, 3)
        # G_kl = sum_i A_i * <phi_k(p_i), phi_l(p_i)>
        self.G = torch.einsum('i,ikc,ilc->kl', self.A_v, self.Phi_V, self.Phi_V)
        self.G_reg = self.G + self.ridge * torch.eye(self.k, device=self.device)
        diag = torch.diag(self.G).cpu().numpy()
        print(f"[init] Gram diag (vertex-weighted L2 norms^2): "
              f"min={diag.min():.3f}, max={diag.max():.3f}")

        # ---- State ----
        self.rho   = torch.zeros(Nv, device=self.device)
        self.alpha = torch.zeros(self.k, device=self.device)
        # Quick check: vertices projected back to mesh should stay almost unchanged.
        _q_test, _f_test, _b_test = self._cp_and_bary(self.V_np[: min(1024, Nv)])
        _cp_err = np.linalg.norm(_q_test - self.V_np[: len(_q_test)], axis=1).max()
        print(f"[init] FCPW cp sanity: max |cp(V)-V| = {_cp_err:.3e}")

    # --------------------------------------------------------
    # Initial conditions
    # --------------------------------------------------------
    def make_density_blob(self, center, radius):
        """rho_i = exp( -|p_i - center|^2 / (2 r^2) ).  Euclidean dist (good enough for V1)."""
        c = torch.as_tensor(center, dtype=torch.float32, device=self.device).view(3)
        d2 = ((self.V_t - c) ** 2).sum(-1)
        return torch.exp(-d2 / (2.0 * float(radius) ** 2))

    def make_target_velocity(self, center, radius, direction, speed):
        """
        Localized tangent target velocity:
            v_tar(p_i) = speed * exp(-|p-c|^2/2r^2) * P_{T_{p_i}M}(direction)

        `direction` is expected to be a unit tangent at the seed vertex (caller's
        responsibility). Away from the seed P_T may shrink, but the envelope kills
        the contribution there anyway, so we deliberately do NOT normalize tau:
        that avoids exploding tangent components when d is nearly normal to N_v.
        """
        c = torch.as_tensor(center, dtype=torch.float32, device=self.device).view(3)
        d = torch.as_tensor(direction, dtype=torch.float32, device=self.device).view(1, 3)
        tau = d - (d * self.N_v).sum(-1, keepdim=True) * self.N_v          # (Nv, 3)
        env = torch.exp(-((self.V_t - c) ** 2).sum(-1, keepdim=True) /
                        (2.0 * float(radius) ** 2))
        return float(speed) * env * tau

    # --------------------------------------------------------
    # Projection / reconstruction at fixed vertices
    # --------------------------------------------------------
    def project_velocity(self, v_target):
        """alpha = (Phi^T A Phi + ridge I)^{-1} Phi^T A v_target  (area-weighted L2)."""
        b = torch.einsum('i,ikc,ic->k', self.A_v, self.Phi_V, v_target)
        return torch.linalg.solve(self.G_reg, b)

    def reconstruct_at_vertices(self, alpha):
        return torch.einsum('ikc,k->ic', self.Phi_V, alpha)

    # --------------------------------------------------------
    # Closest-point projection to mesh via FCPW
    # --------------------------------------------------------
    def _cp_and_bary(self, query_np):
        """
        Returns
        -------
        q    : (Nq, 3) closest points on mesh
        fidx : (Nq,) closest triangle ids
        bary : (Nq, 3) barycentric coordinates in closest triangle
        """
        q, fidx = fcpw_closest_points_and_faces(self._fcpw_scene, query_np)

        tri = self.V_np[self.F_np[fidx]]          # (Nq, 3, 3)
        bary = points_to_barycentric_np(tri, q)   # (Nq, 3)

        return q.astype(np.float32), fidx.astype(np.int64), bary.astype(np.float32)

    # --------------------------------------------------------
    # One semi-Lagrangian step
    # --------------------------------------------------------
    def step(self, dt):
        # 1. Current velocity at vertices
        v_p = self.reconstruct_at_vertices(self.alpha)                   # (Nv, 3)

        # 2. Backtrace + closest-point projection
        query_np = (self.V_t - dt * v_p).detach().cpu().numpy()
        q_np, fidx_np, bary_np = self._cp_and_bary(query_np)
        q_t    = torch.from_numpy(q_np).to(self.device)
        bary_t = torch.from_numpy(bary_np).to(self.device)               # (Nv, 3)
        tri_v  = self.F_t[torch.from_numpy(fidx_np).to(self.device)]     # (Nv, 3) vert ids

        # 3. Density advection: rho^{t+1}(p_i) = sum_v bary[v] * rho[tri_v[i,v]]
        rho_new = (bary_t * self.rho[tri_v]).sum(-1)

        # 4. Velocity at q in T_{q_i}M:  v_bar = sum_k phi_k(q_i) alpha_k
        if self.sdf is not None:
            with torch.enable_grad():
                n_q = self.sdf.normal(q_t).detach()
        else:
            n_q = (bary_t.unsqueeze(-1) * self.N_v[tri_v]).sum(1)
            n_q = n_q / torch.clamp(torch.norm(n_q, dim=-1, keepdim=True), min=1e-12)

        # velocity at the backtraced points: barycentric interpolation of the
        # vertex velocities
        v_p_all = torch.einsum('ikc,k->ic', self.Phi_V, self.alpha)   # (Nv, 3)
        v_bar   = (bary_t.unsqueeze(-1) * v_p_all[tri_v]).sum(1)      # (Nv, 3)

        # 5. Parallel transport v_bar from T_{q_i}M to T_{p_i}M
        v_til = parallel_transport_min_rotation(v_bar, n_q, self.N_v)

        # 6. Re-project onto basis at vertices
        alpha_new = self.project_velocity(v_til)

        # Commit
        self.rho   = rho_new
        self.alpha = alpha_new

    # --------------------------------------------------------
    # Run loop
    # --------------------------------------------------------
    def run(self, n_steps, dt, save_every=10, source_fn=None, log_every=50):
        hist = {'t': [], 'rho': [], 'alpha': [], 'energy': [], 'mass': []}
        t = 0.0

        def snap():
            hist['t'].append(t)
            hist['rho'].append(self.rho.detach().cpu().numpy().copy())
            hist['alpha'].append(self.alpha.detach().cpu().numpy().copy())
            # Surface-L2 energy 1/2 ||v||^2 = 1/2 alpha^T G alpha
            hist['energy'].append(0.5 * float(self.alpha @ self.G @ self.alpha))
            hist['mass'].append(float((self.rho * self.A_v).sum()))

        snap()
        for s in range(1, n_steps + 1):
            self.step(dt)
            t += dt

            if source_fn is not None:
                self.rho = torch.maximum(self.rho, source_fn(t))

            if s % save_every == 0:
                snap()

            if s % log_every == 0 or s == 1:
                print(f"  step {s:5d}/{n_steps}  t={t:7.3f}  "
                      f"E={hist['energy'][-1]:.6f}  "
                      f"mass={hist['mass'][-1]:.6f}  "
                      f"rho_max={float(self.rho.max()):.4f}")
        return hist


# ============================================================
# Polyscope viewer (optional)
# ============================================================

def visualize_polyscope(npz_or_dict, fps=30):
    import polyscope as ps
    import polyscope.imgui as psim
    import time

    if isinstance(npz_or_dict, str):
        d = np.load(npz_or_dict, allow_pickle=False)
        V, F = d['mesh_vertices'], d['mesh_faces']
        rho_seq = d['rho']; t_seq = d['t']
    else:
        V, F = npz_or_dict['mesh_vertices'], npz_or_dict['mesh_faces']
        rho_seq = np.array(npz_or_dict['rho']); t_seq = np.array(npz_or_dict['t'])

    n_frames = len(rho_seq)
    rmax = float(max(r.max() for r in rho_seq))

    ps.init()
    ps.set_up_dir("z_up")
    ps.set_ground_plane_mode("shadow_only")
    sm = ps.register_surface_mesh("mesh", V, F, smooth_shade=True)
    sm.add_scalar_quantity("density", rho_seq[0],
                           cmap='viridis', enabled=True, vminmax=(0.0, rmax))

    state = {'frame': 0, 'play': True, 'last': time.time()}

    def cb():
        psim.PushItemWidth(180)
        _, state['frame'] = psim.SliderInt("frame", state['frame'], 0, n_frames - 1)
        psim.PopItemWidth()
        _, state['play'] = psim.Checkbox("play", state['play'])
        if state['play']:
            now = time.time()
            if now - state['last'] > 1.0 / fps:
                state['frame'] = (state['frame'] + 1) % n_frames
                state['last'] = now
            ps.request_redraw()
        sm.add_scalar_quantity("density", rho_seq[state['frame']],
                               cmap='viridis', enabled=True, vminmax=(0.0, rmax))
        psim.Text(f"t = {t_seq[state['frame']]:.4f}   frame {state['frame']}/{n_frames - 1}")
    ps.set_user_callback(cb)
    ps.show()


# ============================================================
# CLI
# ============================================================

def _vec3(s):
    parts = [float(x) for x in s.split(',')]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Expected 'x,y,z'")
    return parts


def main():
    p = argparse.ArgumentParser(description="Surface basis advection (semi-Lagrangian)")
    p.add_argument("mesh", help="Path to .obj")
    p.add_argument("model", help="Path to model.pt")
    p.add_argument("--no-sdf", action="store_true",
                   help="Do NOT rebuild SDF (only valid if model was trained with --skip-sdf)")
    p.add_argument("--num-basis", type=int, default=None,
                   help="Use only first num_basis trained modes (default: all)")
    p.add_argument("--no-normalize", action="store_true")
    p.add_argument("--bsp-res", type=int, default=500, help="SDF resolution (must match training)")
    p.add_argument("--padding", type=float, default=0.15)
    p.add_argument("--code-value", type=float, default=0.5)
    p.add_argument("--ridge", type=float, default=1e-6)

    # Initial blob / velocity
    g = p.add_mutually_exclusive_group()
    g.add_argument("--center", type=_vec3, default=None,
                   help="Density blob center 'x,y,z' (snapped to nearest vertex)")
    g.add_argument("--center-vertex", type=int, default=None,
                   help="Use vertex index as blob center")
    p.add_argument("--radius", type=float, default=0.05)
    p.add_argument("--direction", type=_vec3, default=None,
                   help="Initial 3D flow direction; tangent-projected. Random if omitted.")
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--continuous-source", action="store_true",
                   help="Continuously inject blob density at the seed location")

    # Simulation
    p.add_argument("--dt", type=float, default=0.005)
    p.add_argument("--n-steps", type=int, default=1000)
    p.add_argument("--save-every", type=int, default=5)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--out", type=str, default="advection_results.npz")
    p.add_argument("--vis", action="store_true", help="Open polyscope viewer after run")
    p.add_argument("--vis-only", type=str, default=None,
                   help="Skip simulation; just open polyscope on this NPZ")
    p.add_argument("--seed", type=int, default=0)

    args = p.parse_args()

    if args.vis_only:
        visualize_polyscope(args.vis_only)
        return

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    # sim = SurfaceBasisAdvection(
    #     mesh_path=args.mesh, model_path=args.model,
    #     normalize=not args.no_normalize, code_value=args.code_value,
    #     ridge=args.ridge,
    #     use_sdf=not args.no_sdf, sdf_res=args.bsp_res, sdf_padding=args.padding,
    # )
    sim = SurfaceBasisAdvection(
        mesh_path=args.mesh, model_path=args.model,
        normalize=not args.no_normalize, code_value=args.code_value,
        ridge=args.ridge,
        use_sdf=not args.no_sdf, sdf_res=args.bsp_res, sdf_padding=args.padding,
        num_basis=args.num_basis,                             # NEW
    )

    # ---- Resolve seed vertex ----
    if args.center_vertex is not None:
        v_idx = int(args.center_vertex)
        if not (0 <= v_idx < len(sim.V_np)):
            raise SystemExit(f"--center-vertex {v_idx} out of range [0, {len(sim.V_np)})")
        center = sim.V_np[v_idx]
    elif args.center is not None:
        c = np.asarray(args.center, dtype=np.float32)
        v_idx = int(np.argmin(np.linalg.norm(sim.V_np - c, axis=1)))
        center = sim.V_np[v_idx]
    else:
        v_idx = int(rng.integers(0, len(sim.V_np)))
        center = sim.V_np[v_idx]
    n_c = sim.N_v[v_idx].detach().cpu().numpy()
    print(f"\n[seed] vertex {v_idx}: center={center}, normal={n_c}")

    # ---- Resolve tangent direction ----
    if args.direction is None:
        d = rng.standard_normal(3).astype(np.float32)
    else:
        d = np.asarray(args.direction, dtype=np.float32)
    d = d - (d @ n_c) * n_c
    if np.linalg.norm(d) < 1e-6:                       # extremely unlucky alignment
        d = rng.standard_normal(3).astype(np.float32)
        d = d - (d @ n_c) * n_c
    d = d / np.linalg.norm(d)
    print(f"[seed] tangent direction: {d}")

    # ---- Initial state ----
    sim.rho   = sim.make_density_blob(center, args.radius)
    v_target  = sim.make_target_velocity(center, args.radius, d, args.speed)
    sim.alpha = sim.project_velocity(v_target)

    # Diagnostic: how well does the basis represent the target?
    with torch.no_grad():
        v_recon = sim.reconstruct_at_vertices(sim.alpha)
        num = float((sim.A_v * ((v_recon - v_target) ** 2).sum(-1)).sum())
        den = float((sim.A_v * (v_target ** 2).sum(-1)).sum())
        rel = (num / max(den, 1e-30)) ** 0.5
    print(f"[seed] initial alpha = {sim.alpha.detach().cpu().numpy()}")
    print(f"[seed] projection rel L2 error: {rel:.3%}")
    E0 = 0.5 * float(sim.alpha @ sim.G @ sim.alpha)
    print(f"[seed] initial L2 energy E = {E0:.6f}")

    source_fn = None
    if args.continuous_source:
        rho_seed = sim.make_density_blob(center, args.radius).detach().clone()
        source_fn = lambda _t: rho_seed

    print(f"\n[run] {args.n_steps} steps, dt={args.dt}, save_every={args.save_every}")
    hist = sim.run(args.n_steps, args.dt,
                   save_every=args.save_every,
                   log_every=args.log_every,
                   source_fn=source_fn)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.savez(
        args.out,
        mesh_vertices=sim.V_np, mesh_faces=sim.F_np,
        t=np.asarray(hist['t'], dtype=np.float32),
        rho=np.asarray(hist['rho'], dtype=np.float32),
        alpha=np.asarray(hist['alpha'], dtype=np.float32),
        energy=np.asarray(hist['energy'], dtype=np.float32),
        mass=np.asarray(hist['mass'], dtype=np.float32),
        center=center.astype(np.float32),
        direction=d.astype(np.float32),
        radius=np.float32(args.radius),
        speed=np.float32(args.speed),
        dt=np.float32(args.dt),
        code_value=np.float32(args.code_value),
        seed_vertex=np.int64(v_idx),
    )
    print(f"\n[done] {len(hist['t'])} frames saved to {args.out}")

    if args.vis:
        visualize_polyscope(args.out)


if __name__ == "__main__":
    main()
