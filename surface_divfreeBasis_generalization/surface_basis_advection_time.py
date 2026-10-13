"""
surface_basis_advection_time.py - Density advection on a moving mesh sequence.

Loads a time-varying SurfaceDFVecLaplacianBasis trained with train_basis_time.py.
Requires vertex/face correspondence across frames (same as training assumption).

Surface analogue of run_simulation.py's 2D SemiLagrangianSimulator.simulate().
The key abstraction is identical: mesh deformation is fundamentally a change of
basis. A 2D continuous shape parameter g(t) becomes a discrete frame index m,
and on each fixed frame we run several semi-Lagrangian steps before switching.

Per frame m (basis Phi^m, scene M^m frozen for `steps_per_frame` iterations):
    u_m   = sum_k alpha_k phi_k(x_i^m, n_i^m, code_m)        [velocity at M^m]
    q_i   = cp_{M^m}(x_i^m - dt * u_m_i)                     [tangent backtrace]
    rho_new[i] = bary-interp rho at q_i  on M^m              [vertex correspondence]
    v_bar = bary-interp velocity at q_i  on M^m
    alpha_new = (G_m + ridge I)^-1 Phi_m^T A_m v_bar         [project on M^m basis]

Frame transition m -> m+1:
    rho   <- rho      (vertex correspondence; no extra interp)
    alpha <- alpha    (coefficient identity; basis Phi^m -> Phi^{m+1} handles
                       the geometry change automatically)

This mirrors the 2D code's handling of g changes:
    self.set_geometry(g_new); self.v[:] = old_v;  self.rho = old_rho

Usage
-----
  python surface_basis_advection_time.py results/hand_2_top60/model.pt \
      --obj-dir data/hand_2_top60 --normalize shared \
      --center-vertex 1000 --radius 0.05 --speed 1.0 \
      --dt 0.0005 --steps-per-frame 100 \
      --out results/hand_2_top60/advection_time.npz
  See run_advection_time.sh for a full list of options.
"""

import argparse
import glob
import os
import numpy as np
import torch

from Mesh_geometry import load_obj
from SparseGridFDM_SDF import _build_fcpw_scene
from SurfaceDFVecLaplacianBasis import SurfaceDFVecLaplacianBasis
from surface_basis_advection import (
    vertex_areas_barycentric,
    vertex_normals_areaweighted,
    fcpw_closest_points_and_faces,
    points_to_barycentric_np,
    parallel_transport_min_rotation,
)


def _shared_normalize(V_list):
    """Replicates train_basis_time.py's --normalize shared (single global transform)."""
    all_v = np.concatenate(V_list, axis=0)
    center = (all_v.max(axis=0) + all_v.min(axis=0)) / 2
    scale = float(np.sqrt(((all_v - center) ** 2).sum(axis=1)).max())
    print(f"[normalize] shared: center={center}, scale={scale:.4f}")
    return [((V - center) / scale).astype(np.float32) for V in V_list]


# ============================================================
# Simulator
# ============================================================

class TimeSurfaceBasisAdvection:

    def __init__(self, mesh_paths, model_path, *, normalize="none",
                 ridge=1e-6, num_basis=None, device=None):
        self.device = (torch.device(device) if device else
                       torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.ridge = float(ridge)

        # ---- Load and validate the sequence ----
        T = len(mesh_paths)
        if T < 2:
            raise SystemExit("Need >=2 mesh frames")

        V_list, F_ref = [], None
        for p in mesh_paths:
            V, F = load_obj(p)
            V_list.append(V.astype(np.float32))
            if F_ref is None:
                F_ref = F.astype(np.int64); Nv = len(V)
            else:
                if len(V) != Nv:
                    raise SystemExit(f"{p}: {len(V)} verts, expected {Nv}")
                if F.shape != F_ref.shape or not (F.astype(np.int64) == F_ref).all():
                    raise SystemExit(f"{p}: face topology differs from frame 0")

        if normalize == "shared":
            V_list = _shared_normalize(V_list)
        elif normalize != "none":
            raise SystemExit(f"unknown normalize: {normalize!r}")

        self.T, self.Nv = T, Nv
        self.V_np  = V_list                                                   # list of (Nv,3)
        self.F_np  = F_ref                                                    # (Nf,3)
        self.F_t   = torch.from_numpy(F_ref).to(self.device)
        # codes match train_basis_time.py: linspace [0,1] over frames
        self.codes = list(np.linspace(0.0, 1.0, T))

        # ---- Load basis (no SDF, matching training) ----
        self.basis = SurfaceDFVecLaplacianBasis.load(
            model_path, device=self.device, sdf=None,
        )
        self.basis.set_eval()
        if num_basis is not None:
            assert 1 <= num_basis <= self.basis.k, \
                f"num_basis must be in [1, {self.basis.k}]"
            self.basis.k = int(num_basis)
        self.k = self.basis.k

        # ---- Per-frame precomputation: V, N, A, Phi, G_reg, FCPW scene ----
        I = torch.eye(self.k, device=self.device)
        self.V_t, self.N_t, self.A_t = [], [], []
        self.Phi, self.G_reg, self.scene = [], [], []
        use_gpu = torch.cuda.is_available()

        print(f"[init] Precomputing per-frame data: T={T}, Nv={Nv}, k={self.k}")
        for m in range(T):
            V_np = V_list[m]
            V_t  = torch.from_numpy(V_np).to(self.device)
            N_t  = torch.from_numpy(
                vertex_normals_areaweighted(V_np, F_ref)
            ).to(self.device)
            A_t  = torch.from_numpy(
                vertex_areas_barycentric(V_np, F_ref)
            ).to(self.device)

            with torch.enable_grad():
                Phi_m = self.basis.eval_all_velocities(
                    V_t, N_t, self.codes[m], detach=True
                )                                                              # (Nv,k,3)
            G = torch.einsum('i,ikc,ilc->kl', A_t, Phi_m, Phi_m)
            G_reg = G + self.ridge * I

            try:
                scene = _build_fcpw_scene(V_np, F_ref.astype(np.int32), use_gpu=use_gpu)
            except Exception:
                scene = _build_fcpw_scene(V_np, F_ref.astype(np.int32), use_gpu=False)

            self.V_t.append(V_t)
            self.N_t.append(N_t)
            self.A_t.append(A_t)
            self.Phi.append(Phi_m)
            self.G_reg.append(G_reg)
            self.scene.append(scene)

        # ---- State ----
        self.rho   = torch.zeros(Nv, device=self.device)
        self.alpha = torch.zeros(self.k, device=self.device)
        self.m     = 0
        self._rho_src     = None    # list of (Nv,) per frame, or None
        self._alpha_src   = None    # list of (k,)  per frame, or None

    # ------------------------------------------------------------------
    # Initial conditions / projection (all on a chosen frame)
    # ------------------------------------------------------------------
    def make_density_blob(self, center, radius, frame=0):
        c = torch.as_tensor(center, dtype=torch.float32, device=self.device).view(3)
        d2 = ((self.V_t[frame] - c) ** 2).sum(-1)
        return torch.exp(-d2 / (2.0 * float(radius) ** 2))

    def make_target_velocity(self, center, radius, direction, speed, frame=0):
        c = torch.as_tensor(center, dtype=torch.float32, device=self.device).view(3)
        d = torch.as_tensor(direction, dtype=torch.float32, device=self.device).view(1, 3)
        N = self.N_t[frame]
        tau = d - (d * N).sum(-1, keepdim=True) * N
        env = torch.exp(-((self.V_t[frame] - c) ** 2).sum(-1, keepdim=True) /
                        (2.0 * float(radius) ** 2))
        return float(speed) * env * tau

    def project(self, v_target, frame):
        b = torch.einsum('i,ikc,ic->k', self.A_t[frame], self.Phi[frame], v_target)
        return torch.linalg.solve(self.G_reg[frame], b)

    def initialize(self, center, radius, direction, speed, start_frame=0):
        self.m   = start_frame
        self.rho = self.make_density_blob(center, radius, start_frame)
        v_tar = self.make_target_velocity(center, radius, direction, speed, start_frame)
        self.alpha = self.project(v_tar, start_frame)

    def enable_sources(self, v_idx, radius, direction_world, speed,
                       rho_source=True, vel_source=True):
        """Per-frame material-tracking sources at vertex `v_idx`.

        rho_source: each step clamp rho >= initial Gaussian blob (in current frame coords).
        vel_source: each step add a fixed-magnitude tangent velocity at the source point,
                    in the direction obtained by tangent-projecting `direction_world` onto
                    the current frame's tangent plane at v_idx.
        """
        d_w = torch.as_tensor(direction_world, dtype=torch.float32,
                              device=self.device).view(1, 3)
        rho_list, alpha_list = [], []
        for m in range(self.T):
            c = self.V_t[m][v_idx]
            env = torch.exp(-((self.V_t[m] - c) ** 2).sum(-1) /
                            (2.0 * float(radius) ** 2))
            if rho_source:
                rho_list.append(env)                                       # (Nv,)
            if vel_source:
                N = self.N_t[m]
                tau = d_w - (d_w * N).sum(-1, keepdim=True) * N            # (Nv,3)
                v_src = float(speed) * env.unsqueeze(-1) * tau
                alpha_list.append(self.project(v_src, m))                  # (k,)
        self._rho_src   = rho_list   if rho_source else None
        self._alpha_src = alpha_list if vel_source else None

    # ------------------------------------------------------------------
    # One step m -> m+1
    # ------------------------------------------------------------------
    def step(self, dt, advance_mesh=True):
        m = self.m
        if advance_mesh and m + 1 >= self.T:
            return False
        target = m + 1 if advance_mesh else m

        # 1. Velocity at OLD frame vertices
        u_m = torch.einsum('ikc,k->ic', self.Phi[m], self.alpha)              # (Nv,3)

        # 2. Backtrace + closest point on OLD surface
        query_np = (self.V_t[m] - dt * u_m).detach().cpu().numpy()
        q_np, fidx_np = fcpw_closest_points_and_faces(self.scene[m], query_np)
        tri_np  = self.V_np[m][self.F_np[fidx_np]]
        bary_np = points_to_barycentric_np(tri_np, q_np)

        bary_t = torch.from_numpy(bary_np).to(self.device)                    # (Nv,3)
        tri_v  = self.F_t[torch.from_numpy(fidx_np).to(self.device)]          # (Nv,3) vert ids
        # q_t    = torch.from_numpy(q_np.astype(np.float32)).to(self.device)

        # 3. Density: bary interp on OLD frame; lands on same vertex index in m+1
        rho_new = (bary_t * self.rho[tri_v]).sum(-1)

        # 4. Velocity at q^m: re-eval the network with OLD frame code
        # n_q = (bary_t.unsqueeze(-1) * self.N_t[m][tri_v]).sum(1)
        # n_q = n_q / torch.clamp(n_q.norm(dim=-1, keepdim=True), min=1e-12)
        # with torch.enable_grad():
        #     Phi_q = self.basis.eval_all_velocities(
        #         q_t, n_q, self.codes[m], detach=True
        #     )                                                                  # (Nv,k,3)
        # v_bar = torch.einsum('ikc,k->ic', Phi_q, self.alpha)

        # 4. Velocity at q^m: bary-interp of vertex velocities on OLD frame (cheap)
        n_q = (bary_t.unsqueeze(-1) * self.N_t[m][tri_v]).sum(1)
        n_q = n_q / torch.clamp(n_q.norm(dim=-1, keepdim=True), min=1e-12)
        v_bar = (bary_t.unsqueeze(-1) * u_m[tri_v]).sum(1)                    # (Nv,3)

        # 5. Transport from T_{q^m}M^m to T_{x^{target}}M^{target}
        v_til = parallel_transport_min_rotation(v_bar, n_q, self.N_t[target])

        # 6. Project onto target frame basis
        alpha_new = self.project(v_til, target)

        self.rho, self.alpha, self.m = rho_new, alpha_new, target
        # Continuous sources (applied on the post-step frame `target`)
        if self._rho_src is not None:
            self.rho = torch.maximum(self.rho, self._rho_src[target])
        if self._alpha_src is not None:
            # `_alpha_src[m]` is the projection of a *force-shaped* velocity
            # onto Phi[m]; multiplying by dt makes it a Riemann sum over the
            # frame, exactly matching SemiLagrangianSimulator.add_force in 2D
            # (`self.v[k] += self.dt * <force, basis_k>`). Without `* dt`,
            # running e.g. 1000 steps per frame would accumulate the source
            # 1000x and blow up.
            self.alpha = self.alpha + dt * self._alpha_src[target]
        return True

    # ------------------------------------------------------------------
    # Run loop - one fixed frame at a time (mirrors 2D simulate())
    # ------------------------------------------------------------------
    def run(self, dt=0.002, steps_per_frame=1, save_every=1, log_every=10):
        """Per-frame semi-Lagrangian on a fixed geometry.

        Surface analogue of run_simulation.py's SemiLagrangianSimulator.simulate():

            for each frame m in [start_frame, T):
                # geometry / basis are now M^m, Phi^m, scene^m (set via self.m)
                for s in range(steps_per_frame):
                    one semi-Lagrangian step on the FIXED frame m
                # End of frame m: move to m+1 by simply incrementing self.m;
                # rho and alpha inherit unchanged (vertex correspondence +
                # coefficient identity), exactly like the 2D code keeps
                # `self.v` and `self.rho` across `set_geometry(g_new)`.

        Notes
        -----
        * `step(dt, advance_mesh=False)` is the surface one-step SL on the
          fixed current frame: backtrace, density bary-interp and velocity
          projection are all done on M^{self.m}; no transport across frames.
        * Frame transitions are NOT explicit transport steps. The mesh
          deformation x_i^m -> x_i^{m+1} is given via the shared vertex
          index, and the time-varying basis Phi[m] -> Phi[m+1] handles the
          basis change.
        """
        hist = {'frame': [], 'step_in_frame': [], 'rho': [], 'alpha': [],
                'energy': [], 'mass': []}

        def snap(step_in_frame):
            G = self.G_reg[self.m] - self.ridge * torch.eye(
                self.k, device=self.device
            )
            hist['frame'].append(self.m)
            hist['step_in_frame'].append(step_in_frame)
            hist['rho'].append(self.rho.detach().cpu().numpy().copy())
            hist['alpha'].append(self.alpha.detach().cpu().numpy().copy())
            hist['energy'].append(0.5 * float(self.alpha @ G @ self.alpha))
            hist['mass'].append(float((self.rho * self.A_t[self.m]).sum()))

        snap(step_in_frame=0)
        n_frames = self.T - self.m
        total = n_frames * steps_per_frame
        s = 0
        for m in range(self.m, self.T):
            self.m = m   # current geometry/basis = frame m
            for i_step in range(steps_per_frame):
                # advance_mesh=False -> target = m, all operations stay on M^m
                if not self.step(dt, advance_mesh=False):
                    return hist
                s += 1
                if s % save_every == 0 or s == total:
                    snap(step_in_frame=i_step + 1)
                if s % log_every == 0 or s == 1 or s == total:
                    print(f"  step {s:5d}/{total}  frame={self.m:03d}  "
                          f"local={i_step+1}/{steps_per_frame}  "
                          f"E={hist['energy'][-1]:.5f}  "
                          f"mass={hist['mass'][-1]:.5f}  "
                          f"rho_max={float(self.rho.max()):.4f}")
            # End of frame m: outer loop advances m -> m+1.
            # rho and alpha are inherited unchanged.
        return hist


# ============================================================
# CLI
# ============================================================

def _vec3(s):
    parts = [float(x) for x in s.split(',')]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Expected 'x,y,z'")
    return parts


def main():
    ap = argparse.ArgumentParser(description="Time-varying surface density advection")
    ap.add_argument("model", help="Path to time-varying model.pt")
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--obj-dir", help="Folder with frame .obj files")
    grp.add_argument("--obj-list", nargs='+', help="Explicit list of .obj paths (in order)")
    ap.add_argument("--obj-pattern", default="*.obj")
    ap.add_argument("--normalize", choices=["none", "shared"], default="none",
                    help="Must match training (--normalize from train_basis_time.py)")

    ap.add_argument("--num-basis", type=int, default=None,
                    help="Use only first num_basis modes (default: all trained)")
    ap.add_argument("--ridge", type=float, default=1e-6)

    g = ap.add_mutually_exclusive_group()
    g.add_argument("--center", type=_vec3, default=None,
                   help="Blob center 'x,y,z' (snapped to nearest vertex on start frame)")
    g.add_argument("--center-vertex", type=int, default=None,
                   help="Vertex index for blob center")
    ap.add_argument("--radius", type=float, default=0.05)
    ap.add_argument("--direction", type=_vec3, default=None,
                    help="Initial 3D direction (tangent-projected). Random if omitted.")
    ap.add_argument("--speed", type=float, default=1.0)

    ap.add_argument("--dt", type=float, default=0.002,
                    help="Per-step time increment (analogous to the 2D "
                         "SemiLagrangianSimulator's dt). Each frame runs "
                         "steps_per_frame steps of size dt on its fixed "
                         "geometry.")
    ap.add_argument("--steps-per-frame", type=int, default=1,
                    help="Number of semi-Lagrangian steps to run on each "
                         "FIXED mesh frame before moving to the next. "
                         "Frame transitions inherit rho and alpha unchanged "
                         "(vertex correspondence + coefficient identity), "
                         "mirroring how the 2D simulator inherits state "
                         "across set_geometry(g_new).")
    ap.add_argument("--start-frame", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=1)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--out", default="advection_time.npz")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--continuous-sources", action="store_true",
                help="Re-inject density blob and velocity at seed vertex every step")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    if args.obj_dir:
        mesh_paths = sorted(glob.glob(os.path.join(args.obj_dir, args.obj_pattern)))
        if not mesh_paths:
            raise SystemExit(f"No .obj matching '{args.obj_pattern}' in {args.obj_dir}")
    else:
        mesh_paths = list(args.obj_list)
    print(f"[load] {len(mesh_paths)} frames: {os.path.basename(mesh_paths[0])} ... "
          f"{os.path.basename(mesh_paths[-1])}")

    sim = TimeSurfaceBasisAdvection(
        mesh_paths=mesh_paths, model_path=args.model,
        normalize=args.normalize, ridge=args.ridge, num_basis=args.num_basis,
    )

    # ---- Resolve seed on start_frame ----
    sf = int(args.start_frame)
    if not (0 <= sf < sim.T - 1):
        raise SystemExit(f"--start-frame must be in [0, {sim.T-2}]")
    V0 = sim.V_np[sf]
    if args.center_vertex is not None:
        v_idx = int(args.center_vertex)
        if not (0 <= v_idx < sim.Nv):
            raise SystemExit(f"--center-vertex out of range [0, {sim.Nv})")
        center = V0[v_idx]
    elif args.center is not None:
        c = np.asarray(args.center, dtype=np.float32)
        v_idx = int(np.argmin(np.linalg.norm(V0 - c, axis=1)))
        center = V0[v_idx]
    else:
        v_idx = int(rng.integers(0, sim.Nv))
        center = V0[v_idx]
    n_c = sim.N_t[sf][v_idx].detach().cpu().numpy()
    print(f"[seed] frame={sf}, vertex={v_idx}, center={center}, normal={n_c}")

    if args.direction is None:
        d = rng.standard_normal(3).astype(np.float32)
    else:
        d = np.asarray(args.direction, dtype=np.float32)
    d = d - (d @ n_c) * n_c
    if np.linalg.norm(d) < 1e-6:
        d = rng.standard_normal(3).astype(np.float32)
        d = d - (d @ n_c) * n_c
    d = d / np.linalg.norm(d)
    print(f"[seed] tangent direction: {d}")

    sim.initialize(center=center, radius=args.radius,
                   direction=d, speed=args.speed, start_frame=sf)

    if args.continuous_sources:
        sim.enable_sources(v_idx=v_idx, radius=args.radius,
                           direction_world=d, speed=args.speed)
        print(f"[seed] continuous sources enabled at vertex {v_idx}")

    with torch.no_grad():
        v_recon = torch.einsum('ikc,k->ic', sim.Phi[sf], sim.alpha)
        v_tar   = sim.make_target_velocity(center, args.radius, d, args.speed, sf)
        num = float((sim.A_t[sf] * ((v_recon - v_tar) ** 2).sum(-1)).sum())
        den = float((sim.A_t[sf] * (v_tar ** 2).sum(-1)).sum())
        rel = (num / max(den, 1e-30)) ** 0.5
    print(f"[seed] initial alpha = {sim.alpha.detach().cpu().numpy()}")
    print(f"[seed] projection rel L2 error: {rel:.3%}")

    print(f"\n[run] frames {sf} -> {sim.T - 1}, dt={args.dt}, "
          f"steps_per_frame={args.steps_per_frame}")
    hist = sim.run(dt=args.dt, steps_per_frame=args.steps_per_frame,
               save_every=args.save_every, log_every=args.log_every)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.savez(
        args.out,
        # full mesh sequence (so visualizer can show density on the right frame)
        mesh_vertices_seq=np.stack(sim.V_np, axis=0).astype(np.float32),       # (T,Nv,3)
        mesh_faces=sim.F_np,
        codes=np.asarray(sim.codes, dtype=np.float32),
        # per-snapshot
        frame=np.asarray(hist['frame'], dtype=np.int64),
        step_in_frame=np.asarray(hist['step_in_frame'], dtype=np.int64),
        rho=np.asarray(hist['rho'], dtype=np.float32),
        alpha=np.asarray(hist['alpha'], dtype=np.float32),
        energy=np.asarray(hist['energy'], dtype=np.float32),
        mass=np.asarray(hist['mass'], dtype=np.float32),
        # metadata
        center=np.asarray(center, dtype=np.float32),
        direction=d.astype(np.float32),
        radius=np.float32(args.radius),
        speed=np.float32(args.speed),
        dt=np.float32(args.dt),
        start_frame=np.int64(sf),
        seed_vertex=np.int64(v_idx),
    )
    print(f"\n[done] {len(hist['frame'])} snapshots -> {args.out}")


if __name__ == "__main__":
    main()
