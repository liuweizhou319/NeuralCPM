"""
SurfaceDFVecLaplacianBasis - Divergence-Free Vector Laplacian Eigenbasis
=========================================================================

Each basis function is a divergence-free tangent vector field on a surface,
given by a neural stream function:

    psi   = Psi(x, n, code)
    u     = n x grad_S psi          tangent and divergence-free by construction
    omega = -Lap_S psi              surface vorticity

The k modes minimise the Rayleigh quotient

    R[psi] = int (Lap_S psi)^2 / int |grad_S psi|^2

one after the other: every mode is deflated against the previous ones
(sequential deflation), which keeps the modes orthogonal.

In this folder the class is used by train_basis_time.py for a family of
surfaces: `code` is the normalised shape parameter (time) in [0, 1], the
networks are shared by all shapes of the family, and no SDF is used
(sdf=None).  The normals are the mesh normals, and train_basis_time.py feeds
zeros to the normal input of the networks.

Viewer for one exported mode:

    python SurfaceDFVecLaplacianBasis.py --vis results/hand_2_top60/frame_0000/basis_000.npz
"""

import os
import math
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Callable, List, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad


# ============================================================
# 1. Network: scalar stream function
# ============================================================


class MLP_Scalar(nn.Module):
    """Scalar field with positional encoding on both x and n.

    Input is always:
        [x(3), n(3), code(1)]
    """

    def __init__(self, input_size=3, geo_size=4, hidden=128):
        super().__init__()
        self.dim = input_size

        # x needs stronger PE; n only needs low-frequency PE.
        self.x_freqs = [1, 2, 4, 8, 16, 32, 64]
        self.n_freqs = [1, 2, 4]

        x_dim = 3 * len(self.x_freqs) * 2
        n_dim = 3 * len(self.n_freqs) * 2
        code_dim = 1

        in_dim = x_dim + n_dim + code_dim

        self.layer1 = nn.Linear(in_dim, hidden)
        self.layer2 = nn.Linear(hidden, hidden)
        self.layer3 = nn.Linear(hidden, hidden)
        self.layer4 = nn.Linear(hidden, hidden)
        self.layer5 = nn.Linear(hidden, 1)

    @staticmethod
    def _pe(z, freqs):
        enc = []
        for f in freqs:
            enc.append(torch.sin(f * z))
            enc.append(torch.cos(f * z))
        return torch.cat(enc, dim=-1)

    def forward(self, x):
        x0 = x[:, :3]      # spatial position
        n  = x[:, 3:6]     # surface normal
        g  = x[:, 6:7]     # code / radius

        x_enc = self._pe(x0, self.x_freqs)
        n_enc = self._pe(n,  self.n_freqs)

        h = torch.cat([x_enc, n_enc, g], dim=1)

        h = torch.sin(self.layer1(h))
        h = torch.sin(self.layer2(h))
        h = torch.sin(self.layer3(h))
        h = torch.sin(self.layer4(h))
        return self.layer5(h)



# ============================================================
# 2. Main class
# ============================================================

class SurfaceDFVecLaplacianBasis:
    """
    k divergence-free vector Laplacian eigenbases on a closed surface.

    Each basis is a neural stream function psi_i; the div-free tangent
    velocity is: u_i = n x grad_S psi_i

    With SDF:
        psi_S(p) = Psi(p, n(p), code)
        grad_S psi computed via autograd chain rule through frozen SDF.
    Without SDF:
        psi(x, code), mesh normals used for projection.
    """

    def __init__(
        self,
        num_basis: int = 10,
        input_size: int = 3,
        hidden: int = 128,
        lr: float = 1e-3,
        device: Union[str, torch.device] = 'cpu',
        sdf=None,
    ):
        self.k = num_basis
        self.device = torch.device(device)
        self.input_size = input_size
        self.hidden = hidden
        self.lr = lr
        self.sdf = sdf

        self.geo_size = 4   # only a label now: the input width is fixed inside MLP_Scalar

        # Build k models and optimizers
        self.models: List[MLP_Scalar] = []
        self.optimizers: List[optim.Adam] = []
        for _ in range(self.k):
            m = MLP_Scalar(input_size, self.geo_size, hidden).to(self.device)
            self.models.append(m)
            self.optimizers.append(optim.Adam(m.parameters(), lr=lr))

        mode = "SDF chain-rule" if sdf else "mesh-only"
        n_params = sum(p.numel() for p in self.models[0].parameters())
        print(f"SurfaceDFVecLaplacianBasis ({mode}, geo_size={self.geo_size}): "
              f"{self.k} bases, {n_params} params/basis, device={self.device}")

    # --------------------------------------------------------
    # Input packing
    # --------------------------------------------------------

    def _make_input(self, pts_leaf, normals, code_value):
        """Build network input [x(3), n(3), code(1)].
        The active MLP_Scalar always expects 7 dims; when sdf=None, normals
        come from the mesh instead of the SDF chain rule."""
        N = pts_leaf.shape[0]
        code = torch.full((N, 1), float(code_value),
                          device=self.device, dtype=pts_leaf.dtype)
        return torch.cat([pts_leaf, normals, code], dim=1)

    # --------------------------------------------------------
    # Surface operators: gradient + Laplacian (chain-rule)
    # --------------------------------------------------------

    def _surface_ops(self, model, pts, code_value, normals_detached=None):
        """
        Compute surface gradient and Laplacian for stream function psi.

        With SDF:
            pts_leaf -> sdf.normal(pts_leaf) -> x_full -> model -> psi
            grad(psi, pts_leaf) gives total derivative via chain rule:
                dPsi/dx = d_x Psi + J_n^T d_n Psi
        Without SDF:
            pts_leaf -> x_full -> model -> psi, grad gives d_x Psi only,
            uses normals_detached for tangent projection.

        Returns: psi_val, total_grad, g_surf, surf_lap, n_used
        """
        pts_leaf = pts.detach().requires_grad_(True)

        if self.sdf is not None:
            n = self.sdf.normal(pts_leaf)  # graph: pts_leaf -> n
        else:
            n = normals_detached.detach()

        x_full = self._make_input(pts_leaf, n, code_value)
        psi_val = model(x_full)

        # Total gradient w.r.t. pts_leaf
        total_grad = grad(psi_val.sum(), pts_leaf, create_graph=True,
                          retain_graph=True)[0]

        # Surface gradient: project out normal component
        g = total_grad - (total_grad * n).sum(-1, keepdim=True) * n

        # Surface Laplacian: div_S(g) = div(g) - n . (gradg) . n
        N_pts = pts_leaf.shape[0]
        amb_div = torch.zeros(N_pts, device=self.device)
        correction = torch.zeros(N_pts, device=self.device)
        for i in range(3):
            dgi = grad(g[:, i].sum(), pts_leaf, create_graph=True,
                       retain_graph=True)[0]
            amb_div = amb_div + dgi[:, i]
            correction = correction + n[:, i] * (n * dgi).sum(-1)

        surf_lap = amb_div - correction
        n_out = n.detach() if self.sdf is not None else n
        return psi_val, total_grad, g, surf_lap, n_out

    # --------------------------------------------------------
    # Forward: psi -> u = n x grad_S psi
    # --------------------------------------------------------

    def forward_surface(self, model, pts, code_value, normals_mesh=None):
        """
        Full forward: psi -> grad_S psi -> u = n x grad_S psi -> omega = -Lap_S psi

        Returns
        -------
        u       : (N, 3)  div-free tangent velocity
        omega   : (N, 1)  surface vorticity
        psi     : (N, 1)  stream function
        g_surf  : (N, 3)  surface gradient of psi
        """
        psi, _, g_surf, surf_lap, n = self._surface_ops(
            model, pts, code_value, normals_mesh
        )
        u = torch.cross(n, g_surf, dim=-1)
        omega = -surf_lap.unsqueeze(1)  # omega = -Lap_S psi
        return u, omega, psi, g_surf

    # --------------------------------------------------------
    # CPM loss (detached normal -> partial d_x only)
    # --------------------------------------------------------

    def _cpm_loss(self, model, nb_pts, nb_normals, code_value):
        """
        CPM loss: (d_x Psi . n)^2 on narrow-band points.
        Normal is DETACHED so grad w.r.t. nb_leaf gives d_x Psi only.
        """
        nb_leaf = nb_pts.detach().requires_grad_(True)
        n_det = nb_normals.detach()

        x_full = self._make_input(nb_leaf, n_det, code_value)
        psi_val = model(x_full)

        partial_grad = grad(psi_val.sum(), nb_leaf, create_graph=True)[0]
        return (partial_grad * n_det).sum(-1) ** 2

    # --------------------------------------------------------
    # Velocity evaluation helpers
    # --------------------------------------------------------

    def eval_velocity(self, pts, normals, idx, code_value=0.5):
        """Evaluate velocity of basis idx at surface points."""
        return self.forward_surface(self.models[idx], pts, code_value, normals)

    def eval_all_velocities(self, pts, normals, code_value=0.5,
                            detach=True):
        """
        Evaluate all k bases at surface points.
        Returns vel_all : (N, k, 3)
        """
        vels = []
        for idx in range(self.k):
            u, _, _, _ = self.forward_surface(
                self.models[idx], pts, code_value, normals
            )
            vels.append(u.detach() if detach else u)
        return torch.stack(vels, dim=1)

    # --------------------------------------------------------
    # Deflation
    # --------------------------------------------------------

    @staticmethod
    def project_with_basis(vec, U_prev, ridge=1e-8):
        """
        Soft deflation: project out components in U_prev.
        vec : (M, 1),  U_prev : (M, m) or None
        """
        if U_prev is None:
            return vec, torch.zeros_like(vec), None
        UT = U_prev.T
        lhs = UT @ U_prev + ridge * torch.eye(
            UT.shape[0], device=vec.device, dtype=vec.dtype)
        rhs = UT @ vec
        q = torch.linalg.solve(lhs, rhs)
        reconstruct = U_prev @ q
        return vec - reconstruct, reconstruct, q

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    def train(
        self,
        mesh,
        num_epochs: int = 5000,
        num_surface: int = 4000,
        num_narrowband: int = 2000,
        code_value: float = 0.5,
        lambda_ortho: float = 1e3,
        lambda_norm: float = 10.0,
        lambda_mean: float = 5.0,
        lambda_cpm: float = 10.0,
        ridge: float = 1e-8,
        eps: float = 1e-12,
        band_width: float = 0.05,
        clip_grad: float = 1.0,
        log_every: int = 100,
        save_every: int = 2000,
        save_dir: Optional[str] = None,
    ):
        """
        Main training loop.

        Parameters
        ----------
        mesh : MeshGeometry
            Must have sample_surface, and either sample_adaptive_band
            (if build_adaptive_band was called) or sample_narrow_band.
        """
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)

        dev = self.device
        use_adaptive = hasattr(mesh, '_adaptive_anchors')

        for m in self.models:
            m.train()

        for epoch in range(1, num_epochs + 1):

            # ---- Sample surface points ----
            surf_pts, surf_normals = mesh.sample_surface(
                N=num_surface, device=dev
            )
            surf_normals = surf_normals.float().detach()

            # ---- Sample narrow-band points for CPM ----
            # ---- Narrow band: surface + Gaussian noise + SDF projection ----
            surf_nb, _ = mesh.sample_surface(num_narrowband, device=dev)
            nb_pts = surf_nb + torch.randn_like(surf_nb) * band_width

            if self.sdf is not None:
                nb_cp = self.sdf.closest_point(nb_pts.detach()).detach()
                nb_normals = self.sdf.normal(nb_cp).detach()
                nb_sdf = self.sdf.f(nb_pts.detach()).detach().squeeze(-1)
            else:
                _, nb_normals = mesh.sample_surface(num_narrowband, device=dev)
                nb_normals = nb_normals.float().detach()
                nb_sdf = torch.zeros(num_narrowband, device=dev)

            # ---- Deflation buffers ----
            U_prev = None   # velocity columns (3N, j)
            G_prev = None   # vorticity columns (N, j)

            loss_energy_sum = 0.0
            loss_ortho_sum = 0.0
            loss_mean_sum = 0.0
            loss_cpm_sum = 0.0
            loss_norm_sum = 0.0

            per_mode_energy = []
            per_mode_u_energy = []

            # ---- Train each basis in order ----
            for idx in range(self.k):

                # Forward on surface
                u, omega, psi, g_surf = self.forward_surface(
                    self.models[idx], surf_pts, code_value, surf_normals
                )

                # CPM on narrow band (detached normal)
                cpm_vals = self._cpm_loss(
                    self.models[idx], nb_pts, nb_normals, code_value
                )

                # Flatten for deflation
                u_vec = u.reshape(-1, 1)         # (3N, 1)
                omega_vec = omega.reshape(-1, 1)  # (N, 1)

                # Deflate
                u_proj, reconstruct, q = self.project_with_basis(
                    u_vec, U_prev, ridge
                )
                if U_prev is None:
                    omega_proj = omega_vec
                else:
                    omega_proj = omega_vec - (G_prev @ q)

                # Normalize
                u_proj_energy = (u_proj ** 2).mean()
                scale = torch.sqrt(u_proj_energy + eps)
                u_unit = u_proj / scale
                omega_unit = omega_proj / scale

                # === Loss 1: Rayleigh quotient ===
                loss_energy = (omega_unit ** 2).mean()

                # === Loss 2: Orthogonality ===
                u_energy = (u_vec ** 2).mean()
                loss_ortho = (reconstruct ** 2).mean() / (u_energy + eps)

                # === Loss 3: Normalization ===
                loss_norm = (u_proj_energy - 1.0).pow(2)

                # === Loss 4: Zero-mean stream function ===
                loss_mean = psi.mean().pow(2)

                # === Loss 5: CPM ===
                loss_cpm = cpm_vals.mean()

                # Total
                loss = (loss_energy
                        + lambda_ortho * loss_ortho
                        + lambda_norm * loss_norm
                        + lambda_mean * loss_mean
                        + lambda_cpm * loss_cpm)

                # Step
                self.optimizers[idx].zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.models[idx].parameters(), clip_grad
                )
                self.optimizers[idx].step()

                # Refresh deflation buffers (detached re-eval)
                u2, omega2, _, _ = self.forward_surface(
                    self.models[idx], surf_pts, code_value, surf_normals
                )

                u2_vec = u2.reshape(-1, 1).detach()
                omega2_vec = omega2.reshape(-1, 1).detach()

                u2_proj, _, q2 = self.project_with_basis(u2_vec, U_prev, ridge)
                if U_prev is None:
                    omega2_proj = omega2_vec
                else:
                    omega2_proj = omega2_vec - (G_prev @ q2)

                scale2 = torch.sqrt((u2_proj ** 2).mean() + eps)
                u2_unit = (u2_proj / scale2).detach()
                omega2_unit = (omega2_proj / scale2).detach()

                if U_prev is None:
                    U_prev, G_prev = u2_unit, omega2_unit
                else:
                    U_prev = torch.cat([U_prev, u2_unit], dim=1)
                    G_prev = torch.cat([G_prev, omega2_unit], dim=1)

                loss_energy_sum += float(loss_energy.detach())
                loss_ortho_sum += float(loss_ortho.detach())
                loss_norm_sum += float(loss_norm.detach())
                loss_mean_sum += float(loss_mean.detach())
                loss_cpm_sum += float(loss_cpm.detach())

                with torch.no_grad():
                    rq = (omega_vec ** 2).sum() / ((u_vec ** 2).sum() + eps)
                    per_mode_energy.append(float(rq))
                    per_mode_u_energy.append(float(u_proj_energy))

            # Log
            if epoch % log_every == 0 or epoch == 1:
                print(f"  [{epoch}/{num_epochs}]  "
                      f"energy={loss_energy_sum:.4f}  "
                      f"ortho={loss_ortho_sum:.6f}  "
                      f"norm={loss_norm_sum:.4f}  "
                      f"mean={loss_mean_sum:.4f}  "
                      f"cpm={loss_cpm_sum:.6f}")
                rq_str = ", ".join(f"{v:.2f}" for v in per_mode_energy)
                ue_str = ", ".join(f"{v:.3f}" for v in per_mode_u_energy)
                print(f"    RQ per mode: [{rq_str}]")
                print(f"    |u|^2 per mode: [{ue_str}]")

            if save_dir and save_every and epoch % save_every == 0:
                path = os.path.join(
                    save_dir,
                    f"basis_{self.k}_epoch_{epoch}.pt"
                )
                self.save(path)
                print(f"    -> checkpoint: {path}")

    # ================================================================
    # Export (per-face on mesh)
    # ================================================================

    def export(self, mesh, out_dir, code_value=0.5, batch_size=4096):
        """Export all k bases evaluated at face centers."""
        os.makedirs(out_dir, exist_ok=True)
        self.set_eval()
        dev = self.device

        V = mesh.vertices_np.astype(np.float32)
        Faces = mesh.faces_np.astype(np.int64)
        v0 = V[Faces[:, 0]]; v1 = V[Faces[:, 1]]; v2 = V[Faces[:, 2]]
        fc = ((v0 + v1 + v2) / 3.0).astype(np.float32)
        fn_mesh = mesh.face_normals_np.astype(np.float32)

        # Normals: from SDF or mesh
        if self.sdf is not None:
            fn_parts = []
            with torch.enable_grad():
                for s in range(0, len(fc), batch_size):
                    e = min(s + batch_size, len(fc))
                    pts = torch.from_numpy(fc[s:e]).to(dev)
                    fn_parts.append(
                        self.sdf.normal(pts).detach().cpu().numpy()
                    )
            fn = np.concatenate(fn_parts, axis=0).astype(np.float32)
        else:
            fn = fn_mesh

        for idx in range(self.k):
            all_psi, all_u, all_omega = [], [], []

            with torch.enable_grad():
                for s in range(0, len(fc), batch_size):
                    e = min(s + batch_size, len(fc))
                    pts = torch.from_numpy(fc[s:e]).to(dev)
                    nrm = torch.from_numpy(fn[s:e]).to(dev)
                    u, omega, psi, _ = self.forward_surface(
                        self.models[idx], pts, code_value, nrm
                    )
                    all_psi.append(psi.detach().cpu().numpy().flatten())
                    all_u.append(u.detach().cpu().numpy())
                    all_omega.append(omega.detach().cpu().numpy().flatten())

            path = os.path.join(out_dir, f"basis_{idx:03d}.npz")
            np.savez(path,
                     mesh_vertices=V,
                     mesh_faces=Faces,
                     face_centers=fc,
                     face_normals=fn,
                     face_normals_mesh=fn_mesh,
                     psi=np.concatenate(all_psi),
                     u=np.concatenate(all_u),
                     omega=np.concatenate(all_omega))
            print(f"  basis {idx} -> {path}")

    # ================================================================
    # Save / Load
    # ================================================================

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        torch.save({
            'num_basis': self.k,
            'input_size': self.input_size,
            'geo_size': self.geo_size,
            'hidden': self.hidden,
            'lr': self.lr,
            'model_states': [m.state_dict() for m in self.models],
            'optimizer_states': [o.state_dict() for o in self.optimizers],
        }, path)
        print(f"Saved {self.k} bases to {path}")

    @classmethod
    def load(cls, path: str, device='cpu', sdf=None):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        obj = cls(
            num_basis=ckpt['num_basis'],
            input_size=ckpt.get('input_size', 3),
            hidden=ckpt.get('hidden', 128),
            lr=ckpt.get('lr', 1e-3),
            device=device,
            sdf=sdf,
        )
        for i in range(obj.k):
            obj.models[i].load_state_dict(ckpt['model_states'][i])
            obj.optimizers[i].load_state_dict(ckpt['optimizer_states'][i])
            obj.models[i].eval()
        print(f"Loaded {obj.k} bases from {path}")
        return obj

    def set_eval(self):
        for m in self.models:
            m.eval()

    def set_train(self):
        for m in self.models:
            m.train()

    # ================================================================
    # Evaluation helpers
    # ================================================================

    def compute_gram_matrix(self, pts, normals, code_value=0.5):
        """Gram matrix G_ij = <u_i, u_j>. Returns (k, k)."""
        Phi = self.eval_all_velocities(pts, normals, code_value, detach=True)
        N = Phi.shape[0]
        Phi_flat = Phi.reshape(N * 3, self.k)
        return (Phi_flat.T @ Phi_flat) / (3.0 * N)

    def compute_coeffs(self, pts, normals, u_target, code_value=0.5):
        """Project target velocity onto the basis (least-squares)."""
        Phi = self.eval_all_velocities(pts, normals, code_value, detach=True)
        N = Phi.shape[0]
        Phi_flat = Phi.reshape(N * 3, self.k)
        u_flat = u_target.detach().reshape(N * 3)
        G = Phi_flat.T @ Phi_flat
        rhs = Phi_flat.T @ u_flat
        return torch.linalg.solve(G, rhs)

    def reconstruct(self, pts, normals, coeffs, code_value=0.5):
        """Reconstruct velocity from coefficients. Returns (N, 3)."""
        coeffs = coeffs.view(self.k).to(self.device)
        Phi = self.eval_all_velocities(pts, normals, code_value, detach=True)
        return torch.einsum('nkc,k->nc', Phi, coeffs)

    # ================================================================
    # Visualization
    # ================================================================

    def plot_basis(
        self,
        mesh,
        idx: int,
        code_value: float = 0.5,
        num_points: int = 5000,
        show: bool = True,
        save_path: Optional[str] = None,
    ):
        """
        Visualize a single basis with 4 panels:
        - Stream function psi
        - Speed |u|
        - Vorticity omega
        - Tangent vector field (quiver)
        """
        self.set_eval()
        dev = self.device

        pts, normals = mesh.sample_surface(N=num_points, device=dev)
        normals = normals.float().detach()

        with torch.enable_grad():
            u, omega, psi, _ = self.forward_surface(
                self.models[idx], pts, code_value, normals
            )

        pts_np = pts.detach().cpu().numpy()
        u_np = u.detach().cpu().numpy()
        omega_np = omega.detach().cpu().numpy().flatten()
        psi_np = psi.detach().cpu().numpy().flatten()
        speed = np.sqrt((u_np ** 2).sum(axis=1))

        fig = plt.figure(figsize=(20, 16))

        ax = fig.add_subplot(2, 2, 1, projection='3d')
        sc = ax.scatter(pts_np[:, 0], pts_np[:, 1], pts_np[:, 2],
                        c=psi_np, cmap='RdBu_r', s=2, alpha=0.8)
        fig.colorbar(sc, ax=ax, shrink=0.6, label='psi')
        ax.set_title(f'Stream function psi_{idx}')
        ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')

        ax = fig.add_subplot(2, 2, 2, projection='3d')
        sc = ax.scatter(pts_np[:, 0], pts_np[:, 1], pts_np[:, 2],
                        c=speed, cmap='inferno', s=2, alpha=0.8)
        fig.colorbar(sc, ax=ax, shrink=0.6, label='|u|')
        ax.set_title(f'Speed |u_{idx}|')
        ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')

        ax = fig.add_subplot(2, 2, 3, projection='3d')
        vlim = np.percentile(np.abs(omega_np), 95)
        sc = ax.scatter(pts_np[:, 0], pts_np[:, 1], pts_np[:, 2],
                        c=omega_np, cmap='RdBu_r', s=2, alpha=0.8,
                        vmin=-vlim, vmax=vlim)
        fig.colorbar(sc, ax=ax, shrink=0.6, label='omega')
        ax.set_title(f'Vorticity omega_{idx}')
        ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')

        N = len(pts_np)
        ax = fig.add_subplot(2, 2, 4, projection='3d')
        step = max(1, N // 500)
        ax.scatter(pts_np[:, 0], pts_np[:, 1], pts_np[:, 2],
                   c='lightgray', s=0.5, alpha=0.3)
        arrow_scale = 0.15 / (np.percentile(speed, 90) + 1e-8)
        ax.quiver(pts_np[::step, 0], pts_np[::step, 1], pts_np[::step, 2],
                  u_np[::step, 0] * arrow_scale,
                  u_np[::step, 1] * arrow_scale,
                  u_np[::step, 2] * arrow_scale,
                  color='steelblue', alpha=0.7, arrow_length_ratio=0.3,
                  linewidth=0.8)
        ax.set_title(f'Tangent field u_{idx}')
        ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')

        fig.suptitle(f'Basis psi_{idx}  (code={code_value:.2f})',
                     fontsize=14, fontweight='bold')

        # Export npz
        export_path = ((save_path.rsplit('.', 1)[0] if save_path
                        else f'basis_{idx}') + '.npz')
        np.savez(export_path,
                 pts=pts_np,
                 normals=normals.detach().cpu().numpy(),
                 u=u_np, omega=omega_np, psi=psi_np, speed=speed)
        print(f"Exported to {export_path}")

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close(fig)

    def plot_orthogonality(
        self,
        mesh,
        code_value: float = 0.5,
        num_points: int = 5000,
        show: bool = True,
        save_path: Optional[str] = None,
    ):
        """Plot Gram matrix to verify orthogonality."""
        self.set_eval()
        dev = self.device
        pts, normals = mesh.sample_surface(N=num_points, device=dev)
        normals = normals.float().detach()

        G = self.compute_gram_matrix(pts, normals, code_value).cpu().numpy()

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        ax = axes[0]
        im = ax.imshow(G, cmap='RdBu_r',
                       vmin=-np.abs(G).max(), vmax=np.abs(G).max())
        ax.set_title('Gram matrix G_ij = <u_i, u_j>')
        ax.set_xlabel('j'); ax.set_ylabel('i')
        fig.colorbar(im, ax=ax, shrink=0.8)

        ax = axes[1]
        G_off = G.copy(); np.fill_diagonal(G_off, 0)
        im = ax.imshow(np.abs(G_off), cmap='hot_r')
        ax.set_title('|Off-diagonal| (should be ~ 0)')
        ax.set_xlabel('j'); ax.set_ylabel('i')
        fig.colorbar(im, ax=ax, shrink=0.8)

        diag = np.diag(G)
        fig.suptitle(
            f'Orthogonality: diag [{diag.min():.3f}, {diag.max():.3f}], '
            f'max|off| = {np.abs(G_off).max():.4f}', fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()


# ============================================================
# Polyscope visualizer
# ============================================================

def visualize_polyscope(npz_path):
    import polyscope as ps

    data = np.load(npz_path, allow_pickle=True)
    V = data['mesh_vertices']
    F = data['mesh_faces']
    u = data['u']
    psi = data['psi']
    omega = data['omega']

    ps.init()
    ps.set_up_dir("z_up")
    ps.set_ground_plane_mode("shadow_only")

    sm = ps.register_surface_mesh("mesh", V, F, smooth_shade=True)
    sm.add_scalar_quantity("psi", psi, defined_on='faces',
                           cmap='coolwarm', enabled=True)
    sm.add_scalar_quantity("omega", omega, defined_on='faces',
                           cmap='RdBu_r', enabled=False)
    sm.add_scalar_quantity("|u|", np.linalg.norm(u, axis=1),
                           defined_on='faces', cmap='reds', enabled=False)
    sm.add_vector_quantity("u", u, defined_on='faces',
                           vectortype='ambient', enabled=True,
                           length=0.02, color=(0.2, 0.5, 0.9))
    ps.show()


if __name__ == '__main__':
    import sys
    if '--vis' in sys.argv:
        visualize_polyscope(sys.argv[sys.argv.index('--vis') + 1])
    else:
        print("Usage: python SurfaceDFVecLaplacianBasis.py --vis <out_dir>/frame_0000/basis_000.npz")
        print("Training: python train_basis_time.py --help")
