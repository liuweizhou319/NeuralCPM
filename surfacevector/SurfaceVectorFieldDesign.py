"""
SurfaceVectorFieldDesign - Hodge dual-scalar vector field design
=================================================================

    u = grad_S phi  +  n x grad_S psi

The two scalar potentials are neural fields

    phi = Phi(x, n(x), theta),    psi = Psi(x, n(x), theta)

where n(x) is the normal of the SDF (SparseGridFDM_SDF).  The autograd chain
rule through the SDF gives the total derivative

    dPhi/dx = d_x Phi + J_n^T d_n Phi

so the surface gradient and Laplacian are those of the surface field
phi_S(p) = Phi(p, n(p), theta).

Constraints ("anchors") are target tangent vectors at face centres.  The angle
theta rotates every anchor vector in its tangent plane:

    fixed angle            theta is given (train(fixed_theta=...)); the code
                           fed to the networks is the constant 0
    angle generalization   theta ~ U[0, 2*pi] at every epoch and is fed to
                           the networks, so one model gives the field for any
                           rotation of the anchors

Loss (see train()): data term at the anchors, squared surface Laplacian of phi
and psi (smoothness), closest-point term in a narrow band, zero-mean gauge and
an orthogonality term between the two parts of u.

Training needs the SDF.  The mesh-only code path (sdf=None) is kept for
reference only.

Viewer for an exported field:

    python SurfaceVectorFieldDesign.py --vis results/single/78481root_5w/vecfield_000.npz
"""

import os
import math
import time
import numpy as np
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad


# ============================================================
# 1. Network
# ============================================================

class MLP_Potential(nn.Module):
    """Scalar potential: input [x || n || g] -> scalar. PE on x only; n,g raw. Sine hiddens."""

    def __init__(self, input_size=3, geo_size=4, hidden=128):
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
        # x: [x(input_size), geo] with geo = [n(3), g(1)] or [g(1)] only (see geo_size).
        x0 = x[:, :self.dim]
        geo = x[:, self.dim:]

        enc = []
        for f in self.freqs:
            enc.append(torch.sin(f * x0))
            enc.append(torch.cos(f * x0))
        enc.append(geo)

        h = torch.cat(enc, dim=1)
        h = torch.sin(self.layer1(h))
        h = torch.sin(self.layer2(h))
        h = torch.sin(self.layer3(h))
        h = torch.sin(self.layer4(h))
        return self.layer5(h)


# ============================================================
# 2. Helpers
# ============================================================

def rotate_tangent_vectors(v, n, theta):
    c = math.cos(theta)
    s = math.sin(theta)
    if isinstance(v, np.ndarray):
        return c * v - s * np.cross(n, v)
    else:
        return c * v - s * torch.cross(n, v, dim=-1)


def project_to_tangent(v, n, preserve_norm=False, eps=1e-12):
    if isinstance(v, np.ndarray):
        dot = (v * n).sum(axis=-1, keepdims=True)
        tangent = v - dot * n
        if preserve_norm:
            v_norm = np.linalg.norm(v, axis=-1, keepdims=True)
            t_norm = np.linalg.norm(tangent, axis=-1, keepdims=True)
            scale = np.where(t_norm > eps, v_norm / np.maximum(t_norm, eps), 1.0)
            tangent = tangent * scale
        return tangent
    else:
        dot = (v * n).sum(dim=-1, keepdim=True)
        tangent = v - dot * n
        if preserve_norm:
            v_norm = v.norm(dim=-1, keepdim=True)
            t_norm = tangent.norm(dim=-1, keepdim=True)
            scale = torch.where(
                t_norm > eps,
                v_norm / t_norm.clamp(min=eps),
                torch.ones_like(t_norm),
            )
            tangent = tangent * scale
        return tangent


def generate_random_constraints(mesh, n_constraints=10, seed=42):
    rng = np.random.RandomState(seed)
    n_faces = len(mesh.faces_np)
    face_ids = rng.choice(n_faces, size=min(n_constraints, n_faces), replace=False)
    face_ids = np.sort(face_ids)
    normals = mesh.face_normals_np[face_ids]
    raw = rng.randn(len(face_ids), 3).astype(np.float32)
    dot = (raw * normals).sum(axis=1, keepdims=True)
    tangent = raw - dot * normals
    norms = np.sqrt((tangent ** 2).sum(axis=1, keepdims=True) + 1e-30)
    tangent = tangent / norms
    avg_edge = np.sqrt(mesh.total_area / len(mesh.faces_np))
    tangent *= avg_edge * 2.0
    return face_ids, tangent


def prepare_constraints(mesh, face_ids, vectors, device):
    V, Faces = mesh.vertices_np, mesh.faces_np
    v0 = V[Faces[face_ids, 0]]
    v1 = V[Faces[face_ids, 1]]
    v2 = V[Faces[face_ids, 2]]
    centers = ((v0 + v1 + v2) / 3.0).astype(np.float32)
    normals = mesh.face_normals_np[face_ids].astype(np.float32)
    vectors = np.asarray(vectors, dtype=np.float32)
    tangent = project_to_tangent(vectors, normals)
    return (torch.from_numpy(centers).to(device),
            torch.from_numpy(normals).to(device),
            torch.from_numpy(tangent).to(device))


# ============================================================
# 3. Main class
# ============================================================

class SurfaceVectorFieldDesign:
    """
    Hodge vector field design.

    With SDF:
        phi_S(p) = Phi(p, n(p), theta)    psi_S(p) = Psi(p, n(p), theta)
        grad_S phi_S computed via autograd chain rule through frozen SDF
    Without SDF:
        phi(x, theta), psi(x, theta)  with mesh normals for projection
    """

    def __init__(self, input_size=3, hidden=128, lr=1e-3,
                 device='cpu', sdf=None):
        self.device = torch.device(device)
        self.input_size = input_size
        self.hidden = hidden
        self.lr = lr
        self.sdf = sdf

        geo_size = 4 if sdf is not None else 1
        self.geo_size = geo_size

        self.model_phi = MLP_Potential(input_size, geo_size, hidden).to(self.device)
        self.model_psi = MLP_Potential(input_size, geo_size, hidden).to(self.device)

        self.optimizer = optim.Adam(
            list(self.model_phi.parameters()) + list(self.model_psi.parameters()),
            lr=lr,
        )

        n_phi = sum(p.numel() for p in self.model_phi.parameters())
        n_psi = sum(p.numel() for p in self.model_psi.parameters())
        mode = "SDF chain-rule" if sdf else "mesh-only"
        print(f"SurfaceVectorFieldDesign ({mode}, geo_size={geo_size}): "
              f"phi-net {n_phi} + psi-net {n_psi} = {n_phi+n_psi} params")
        # ---------------- Profiling state ----------------
        self._profile_enabled = False
        self._prof_accum = {}
        self._prof_count = 0

    def _prof_sync(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    def _prof_tic(self):
        if not self._profile_enabled:
            return None
        self._prof_sync()
        return time.perf_counter()

    def _prof_toc(self, key, t0):
        if (not self._profile_enabled) or (t0 is None):
            return
        self._prof_sync()
        dt = time.perf_counter() - t0
        self._prof_accum[key] = self._prof_accum.get(key, 0.0) + dt

    def _prof_reset_window(self):
        self._prof_accum = {}
        self._prof_count = 0

    def _prof_report_window(self, prefix="profile"):
        if self._prof_count <= 0:
            return f"{prefix}: no data"

        keys = [
            "sample_surface",
            "sample_narrow_band",
            "nb_sdf_refresh",

            "surface_ops_phi",
            "surface_ops_phi_query_sdf",
            "surface_ops_phi_dphi_dx_dn",
            "surface_ops_phi_dgi_loop",
            "surface_ops_phi_laplacian_assembly",

            "surface_ops_psi",
            "surface_ops_psi_query_sdf",
            "surface_ops_psi_dphi_dx_dn",
            "surface_ops_psi_dgi_loop",
            "surface_ops_psi_laplacian_assembly",

            "cpm_phi",
            "cpm_psi",
            "forward_u",
            "backward",
            "opt_step",
            "epoch_total",
        ]

        parts = [f"{prefix} avg over {self._prof_count} ep"]
        for k in keys:
            if k in self._prof_accum:
                ms = 1000.0 * self._prof_accum[k] / self._prof_count
                parts.append(f"{k}={ms:.2f}ms")
        return "  ".join(parts)
    # --------------------------------------------------------
    # Core: pack input and compute surface gradient
    # --------------------------------------------------------

    def _make_input(self, pts_leaf, normals, code_value):
        """Build network input [x(3), n(3), theta(1)] or [x(3), theta(1)]."""
        N = pts_leaf.shape[0]
        code = torch.full((N, 1), float(code_value),
                          device=self.device, dtype=pts_leaf.dtype)
        if self.sdf is not None:
            return torch.cat([pts_leaf, normals, code], dim=1)
        else:
            return torch.cat([pts_leaf, code], dim=1)

    def _surface_ops(self, model, pts, code_value, normals_detached=None, prof_tag=None):
        tag = prof_tag if prof_tag is not None else "unk"

        # ---- Decoupled FDM path ----
        if self.sdf is not None and hasattr(self.sdf, 'vertex_sdf'):
            # 1) query_sdf
            t = self._prof_tic()
            pts_det = pts.detach()
            n_det = self.sdf.normal(pts_det).detach()
            Jn = self.sdf.normal_jacobian(pts_det).detach()
            dJn_dx = self.sdf.normal_jacobian_deriv(pts_det).detach()
            self._prof_toc(f"surface_ops_{tag}_query_sdf", t)

            pts_leaf = pts_det.requires_grad_(True)
            n_leaf = n_det.clone().requires_grad_(True)

            # 2) dphi_dx_dn
            t = self._prof_tic()
            x_full = self._make_input(pts_leaf, n_leaf, code_value)
            f_val = model(x_full)

            dphi_dx = grad(f_val.sum(), pts_leaf, create_graph=True,
                        retain_graph=True)[0]
            dphi_dn = grad(f_val.sum(), n_leaf, create_graph=True,
                        retain_graph=True)[0]

            total_grad = dphi_dx + torch.bmm(
                Jn.transpose(1, 2), dphi_dn.unsqueeze(-1)
            ).squeeze(-1)

            g = total_grad - (total_grad * n_leaf).sum(-1, keepdim=True) * n_leaf

            # dJ_n/dx correction: M_{lj} = sum_k (d(J_n)_{kl}/dx_j) (dPhi/dn_k)
            M = torch.einsum('nklj,nk->nlj', dJn_dx, dphi_dn.detach())
            P_det = (torch.eye(3, device=self.device).unsqueeze(0)
                    - n_det.unsqueeze(2) * n_det.unsqueeze(1))
            PM = torch.bmm(P_det, M)
            self._prof_toc(f"surface_ops_{tag}_dphi_dx_dn", t)

            # 3) dgi_loop
            t = self._prof_tic()
            dgi_rows = []
            for i in range(3):
                dgi_dx = grad(g[:, i].sum(), pts_leaf, create_graph=True,
                            retain_graph=True)[0]
                dgi_dn = grad(g[:, i].sum(), n_leaf, create_graph=True,
                            retain_graph=True)[0]
                dgi_full = (
                    dgi_dx
                    + torch.bmm(Jn.transpose(1, 2), dgi_dn.unsqueeze(-1)).squeeze(-1)
                    + PM[:, i, :]
                )  # (N,3) = [dg_i/dx_0, dg_i/dx_1, dg_i/dx_2]
                dgi_rows.append(dgi_full)
            Dg = torch.stack(dgi_rows, dim=1)  # (N,3,3), Dg[:,i,j] = dg_i/dx_j
            self._prof_toc(f"surface_ops_{tag}_dgi_loop", t)

            # 4) laplacian_assembly
            t = self._prof_tic()
            amb_div = Dg.diagonal(dim1=1, dim2=2).sum(-1)           # trace(Dg)
            correction = torch.einsum('ni,nij,nj->n', n_det, Dg, n_det)
            surf_lap = amb_div - correction
            self._prof_toc(f"surface_ops_{tag}_laplacian_assembly", t)

            return f_val, total_grad, g, surf_lap, n_det

        # ---- Original autograd path ----
        pts_leaf = pts.detach().requires_grad_(True)

        # 1) query_sdf
        t = self._prof_tic()
        if self.sdf is not None:
            n = self.sdf.normal(pts_leaf)
        else:
            n = normals_detached.detach()
        self._prof_toc(f"surface_ops_{tag}_query_sdf", t)

        # 2) dphi_dx_dn  (no separate d/dn slot on this path; the timing key keeps the same name)
        t = self._prof_tic()
        x_full = self._make_input(pts_leaf, n, code_value)
        f_val = model(x_full)
        total_grad = grad(f_val.sum(), pts_leaf, create_graph=True,
                        retain_graph=True)[0]
        g = total_grad - (total_grad * n).sum(-1, keepdim=True) * n
        self._prof_toc(f"surface_ops_{tag}_dphi_dx_dn", t)

        # 3) dgi_loop
        t = self._prof_tic()
        dgi_rows = []
        for i in range(3):
            dgi = grad(g[:, i].sum(), pts_leaf, create_graph=True,
                    retain_graph=True)[0]
            dgi_rows.append(dgi)
        Dg = torch.stack(dgi_rows, dim=1)  # (N,3,3), Dg[:,i,j] = dg_i/dx_j
        self._prof_toc(f"surface_ops_{tag}_dgi_loop", t)

        # 4) laplacian_assembly
        t = self._prof_tic()
        amb_div = Dg.diagonal(dim1=1, dim2=2).sum(-1)
        correction = torch.einsum('ni,nij,nj->n', n, Dg, n)
        surf_lap = amb_div - correction
        self._prof_toc(f"surface_ops_{tag}_laplacian_assembly", t)

        n_out = n.detach() if self.sdf is not None else n
        return f_val, total_grad, g, surf_lap, n_out

    def _forward_u(self, pts, normals_mesh, code_value):
        # ---- Decoupled FDM path ----
        if self.sdf is not None and hasattr(self.sdf, 'vertex_sdf'):
            pts_det = pts.detach()
            n_det = self.sdf.normal(pts_det).detach()
            Jn = self.sdf.normal_jacobian(pts_det).detach()

            pts_leaf = pts_det.requires_grad_(True)
            n_leaf = n_det.clone().requires_grad_(True)
            x_full = self._make_input(pts_leaf, n_leaf, code_value)

            phi = self.model_phi(x_full)
            dphi_dx = grad(phi.sum(), pts_leaf, create_graph=True,
                           retain_graph=True)[0]
            dphi_dn = grad(phi.sum(), n_leaf, create_graph=True,
                           retain_graph=True)[0]
            grad_phi = dphi_dx + torch.bmm(
                Jn.transpose(1, 2), dphi_dn.unsqueeze(-1)).squeeze(-1)
            g_phi = grad_phi - (grad_phi * n_leaf).sum(-1, keepdim=True) * n_leaf

            psi = self.model_psi(x_full)
            dpsi_dx = grad(psi.sum(), pts_leaf, create_graph=True,
                           retain_graph=True)[0]
            dpsi_dn = grad(psi.sum(), n_leaf, create_graph=True,
                           retain_graph=True)[0]
            grad_psi = dpsi_dx + torch.bmm(
                Jn.transpose(1, 2), dpsi_dn.unsqueeze(-1)).squeeze(-1)
            g_psi = grad_psi - (grad_psi * n_leaf).sum(-1, keepdim=True) * n_leaf

            u_pot = g_phi
            u_stream = torch.cross(n_det, g_psi, dim=-1)
            u = u_pot + u_stream
            return phi, psi, grad_phi, grad_psi, u_pot, u_stream, u

        # ---- Original autograd path ----
        pts_leaf = pts.detach().requires_grad_(True)
        if self.sdf is not None:
            n = self.sdf.normal(pts_leaf)
        else:
            n = normals_mesh.detach()
        x_full = self._make_input(pts_leaf, n, code_value)
        n_det = n.detach() if self.sdf is not None else n
        phi = self.model_phi(x_full)
        grad_phi = grad(phi.sum(), pts_leaf, create_graph=True,
                        retain_graph=True)[0]
        g_phi = grad_phi - (grad_phi * n).sum(-1, keepdim=True) * n
        psi = self.model_psi(x_full)
        grad_psi = grad(psi.sum(), pts_leaf, create_graph=True,
                        retain_graph=True)[0]
        g_psi = grad_psi - (grad_psi * n).sum(-1, keepdim=True) * n
        u_pot = g_phi
        u_stream = torch.cross(n_det, g_psi, dim=-1)
        u = u_pot + u_stream
        return phi, psi, grad_phi, grad_psi, u_pot, u_stream, u

    def _cpm_loss(self, model, nb_pts, nb_normals, code_value):
        """
        CPM loss: (d_x Phi . n)^2 on narrow-band points.
        Normal is DETACHED - we want the partial derivative d_x only,
        because CPM says: along the normal fiber (fixed n), value is constant.
        """
        nb_leaf = nb_pts.detach().requires_grad_(True)
        n_det = nb_normals.detach()

        x_full = self._make_input(nb_leaf, n_det, code_value)
        f_val = model(x_full)

        # With n detached, grad w.r.t. nb_leaf gives d_x Phi only
        partial_grad = grad(f_val.sum(), nb_leaf, create_graph=True)[0]
        return (partial_grad * n_det).sum(-1) ** 2

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    def train(
        self,
        mesh,
        constraint_face_ids=None,
        constraint_vectors=None,
        n_random_constraints: int = 10,
        num_epochs: int = 5000,
        num_surface: int = 3000,
        num_narrowband: int = 2000,
        band_width: float = 0.05,
        lambda_data: float = 100.0,
        lambda_smooth_phi: float = 0.5,
        lambda_smooth_psi: float = 1.0,
        lambda_cpm: float = 100.0,
        lambda_mean: float = 5.0,
        lambda_orth: float = 1.0,
        clip_grad: float = 1.0,
        log_every: int = 100,
        save_every: int = 0,
        save_dir: Optional[str] = None,
        fixed_theta: Optional[float] = None,
        profile: bool = False,
    ):
        # ---- constraints ----
        if constraint_face_ids is None:
            print(f"No constraints -> generating {n_random_constraints} random ones")
            constraint_face_ids, constraint_vectors = generate_random_constraints(
                mesh, n_random_constraints
            )
        self._constraint_face_ids = np.asarray(constraint_face_ids)
        self._constraint_vectors = np.asarray(constraint_vectors, dtype=np.float32)
        self._fixed_theta = fixed_theta

        c_pts, c_normals, c_target_0 = prepare_constraints(
            mesh, self._constraint_face_ids, self._constraint_vectors, self.device
        )
        if self.sdf is not None:
            c_normals_geom = self.sdf.normal(c_pts.detach()).detach()
            c_target_base = project_to_tangent(
                c_target_0, c_normals_geom, preserve_norm=True
            )
        else:
            c_normals_geom = c_normals
            c_target_base = c_target_0
        print(f"Constraints: {len(c_pts)} faces, "
              f"avg |v|={np.linalg.norm(self._constraint_vectors, axis=1).mean():.4f}")

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        dev = self.device
        self.model_phi.train()
        self.model_psi.train()
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=num_epochs, eta_min=1e-5
        )

        if fixed_theta is not None:
            print(f"Fixed-angle mode: theta = {fixed_theta:.4f}")
        else:
            print("Angle-generalisation mode: theta ~ U[0, 2*pi] each epoch")

        self._profile_enabled = bool(profile)
        self._prof_reset_window()

        for epoch in range(1, num_epochs + 1):
            t_epoch = self._prof_tic()
            if fixed_theta is not None:
                theta = fixed_theta
                code_val = 0.0
            else:
                theta = torch.rand(1).item() * 2 * math.pi
                code_val = theta

            c_target_rot = rotate_tangent_vectors(c_target_base, c_normals_geom, theta)

            # ---- sample surface ----
            t = self._prof_tic()
            surf_pts, surf_n = mesh.sample_surface(N=num_surface, device=dev)
            surf_n = surf_n.float().detach()
            self._prof_toc("sample_surface", t)

            # ---- sample narrow band ----
            t = self._prof_tic()
            # ---- Narrow band: surface + Gaussian noise + SDF projection ----
            surf_nb, _ = mesh.sample_surface(num_narrowband, device=dev)
            noise = torch.randn_like(surf_nb) * band_width
            nb_pts = surf_nb + noise
            self._prof_toc("sample_narrow_band", t)

            if self.sdf is not None and hasattr(self.sdf, 'vertex_sdf'):
                nb_cp = self.sdf.closest_point(nb_pts.detach()).detach()
                nb_n = self.sdf.normal(nb_cp).detach()
                nb_sdf = self.sdf.f(nb_pts.detach()).detach().squeeze(-1)
            elif self.sdf is not None:
                nb_cp = self.sdf.closest_point(nb_pts.detach()).detach()
                nb_n = self.sdf.normal(nb_cp).detach()
                nb_sdf = self.sdf.f(nb_pts.detach()).detach().squeeze(-1)
            else:
                nb_n = surf_n_placeholder  # fallback, won't reach here normally
                nb_sdf = torch.zeros(num_narrowband, device=dev)

            if epoch == 1:
                self._nb_pts = nb_pts.detach().cpu().numpy()
                self._nb_normals = nb_n.detach().cpu().numpy()
                self._nb_sdf = nb_sdf.detach().cpu().numpy()

            # ======== Losses ========

            # 1) Surface Laplacian smoothness
            t = self._prof_tic()
            phi_s, grad_phi_s, g_phi_s, lap_phi, n_surf = self._surface_ops(
                self.model_phi, surf_pts, code_val, surf_n, prof_tag="phi"
            )
            self._prof_toc("surface_ops_phi", t)

            t = self._prof_tic()
            psi_s, grad_psi_s, g_psi_s, lap_psi, _ = self._surface_ops(
                self.model_psi, surf_pts, code_val, surf_n, prof_tag="psi"
            )
            self._prof_toc("surface_ops_psi", t)

            L_smooth_phi = (lap_phi ** 2).mean()
            L_smooth_psi = (lap_psi ** 2).mean()
            L_smooth = (lambda_smooth_phi * L_smooth_phi
                        + lambda_smooth_psi * L_smooth_psi)

            # 2) CPM: narrow band only (detached normal = partial d_x)
            t = self._prof_tic()
            cpm_phi = self._cpm_loss(self.model_phi, nb_pts, nb_n, code_val)
            self._prof_toc("cpm_phi", t)

            t = self._prof_tic()
            cpm_psi = self._cpm_loss(self.model_psi, nb_pts, nb_n, code_val)
            self._prof_toc("cpm_psi", t)

            L_cpm = cpm_phi.mean() + cpm_psi.mean()

            # 3) Gauge
            L_mean = phi_s.mean() ** 2 + psi_s.mean() ** 2

            # 4) Data fidelity
            t = self._prof_tic()
            (_, _, _, _, c_u_pot, c_u_stream, c_u) = self._forward_u(
                c_pts.detach(), c_normals_geom, code_val
            )
            self._prof_toc("forward_u", t)

            L_data = ((c_u - c_target_rot) ** 2).sum(-1).mean()

            # 5) Orthogonality
            u_pot_surf = g_phi_s
            u_stream_surf = torch.cross(n_surf, g_psi_s, dim=-1)
            L_orth = (u_pot_surf * u_stream_surf).sum(-1).mean() ** 2

            loss = (lambda_data * L_data
                    + L_smooth
                    + lambda_cpm * L_cpm
                    + lambda_mean * L_mean
                    + lambda_orth * L_orth)

            self.optimizer.zero_grad(set_to_none=True)

            t = self._prof_tic()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.model_phi.parameters()) + list(self.model_psi.parameters()),
                clip_grad,
            )
            self._prof_toc("backward", t)

            t = self._prof_tic()
            self.optimizer.step()
            scheduler.step()
            self._prof_toc("opt_step", t)

            self._prof_toc("epoch_total", t_epoch)
            self._prof_count += 1

            if epoch % log_every == 0 or epoch == 1:
                with torch.no_grad():
                    u_total = u_pot_surf + u_stream_surf
                    u_mag = u_total.norm(dim=-1).mean().item()
                    pot_mag = u_pot_surf.norm(dim=-1).mean().item()
                    str_mag = u_stream_surf.norm(dim=-1).mean().item()
                    data_err = torch.sqrt(L_data.detach()).item()
                print(
                    f"  [{epoch}/{num_epochs}]  theta={theta:.2f}  "
                    f"loss={loss.item():.4f}  "
                    f"data={L_data.item():.6f}(rmse={data_err:.4f})  "
                    f"sm_phi={L_smooth_phi.item():.4f}  "
                    f"sm_psi={L_smooth_psi.item():.4f}  "
                    f"cpm={L_cpm.item():.6f}  "
                    f"mean={L_mean.item():.6f}  "
                    f"orth={L_orth.item():.6f}  "
                    f"|u|={u_mag:.4f}  |pot|={pot_mag:.4f}  |str|={str_mag:.4f}  "
                    f"lr={scheduler.get_last_lr()[0]:.6f}"
                )
                if self._profile_enabled:
                    print(self._prof_report_window(prefix="profile"))
                    self._prof_reset_window()

            if save_dir and save_every > 0 and epoch % save_every == 0:
                self.save(os.path.join(save_dir, f"vecfield_epoch_{epoch}.pt"))

    # --------------------------------------------------------
    # Export
    # --------------------------------------------------------

    def _export_single(self, mesh, theta, batch_size=4096):
        self.model_phi.eval()
        self.model_psi.eval()
        dev = self.device
        fixed = getattr(self, '_fixed_theta', None)
        code_val = 0.0 if fixed is not None else theta

        V = mesh.vertices_np.astype(np.float32)
        Faces = mesh.faces_np.astype(np.int64)
        v0 = V[Faces[:, 0]]; v1 = V[Faces[:, 1]]; v2 = V[Faces[:, 2]]
        fc = ((v0 + v1 + v2) / 3.0).astype(np.float32)
        fn_mesh = mesh.face_normals_np.astype(np.float32)
        if self.sdf is not None:
            fn_eval = []
            with torch.enable_grad():
                for s in range(0, len(fc), batch_size):
                    e = min(s + batch_size, len(fc))
                    pts = torch.from_numpy(fc[s:e]).to(dev)
                    fn_eval.append(self.sdf.normal(pts).detach().cpu().numpy())
            fn = np.concatenate(fn_eval, axis=0).astype(np.float32)
        else:
            fn = fn_mesh

        all_phi, all_psi = [], []
        all_u_pot, all_u_stream, all_u = [], [], []

        with torch.enable_grad():
            for s in range(0, len(fc), batch_size):
                e = min(s + batch_size, len(fc))
                pts = torch.from_numpy(fc[s:e]).to(dev)
                nrm = torch.from_numpy(fn[s:e]).to(dev)
                phi, psi, _, _, u_pot, u_stream, u = self._forward_u(
                    pts, nrm, code_val
                )
                all_phi.append(phi.detach().cpu().numpy().flatten())
                all_psi.append(psi.detach().cpu().numpy().flatten())
                all_u_pot.append(u_pot.detach().cpu().numpy())
                all_u_stream.append(u_stream.detach().cpu().numpy())
                all_u.append(u.detach().cpu().numpy())

        c_ids = getattr(self, '_constraint_face_ids', np.array([], dtype=np.int64))
        c_vecs_0 = getattr(self, '_constraint_vectors', np.zeros((0, 3), dtype=np.float32))
        if len(c_ids) > 0 and len(c_vecs_0) > 0:
            c_normals = fn[c_ids]
            c_base = project_to_tangent(c_vecs_0, c_normals, preserve_norm=True)
            c_vecs_rot = rotate_tangent_vectors(c_base, c_normals, theta)
        else:
            c_vecs_rot = c_vecs_0

        result = {
            'mesh_vertices': V, 'mesh_faces': Faces,
            'theta': np.float32(theta),
            'constraint_face_ids': c_ids,
            'constraint_vectors_original': c_vecs_0,
            'constraint_vectors_rotated': c_vecs_rot,
            'face_centers': fc,
            'face_normals': fn,
            'face_normals_mesh': fn_mesh,
            'phi': np.concatenate(all_phi),
            'psi': np.concatenate(all_psi),
            'u_pot': np.concatenate(all_u_pot),
            'u_stream': np.concatenate(all_u_stream),
            'u': np.concatenate(all_u),
        }
        if hasattr(self, '_nb_pts'):
            result['nb_pts'] = self._nb_pts
            result['nb_normals'] = self._nb_normals
            result['nb_sdf'] = self._nb_sdf
        return result

    def export(self, mesh, save_path, theta=0.0, batch_size=4096):
        data = self._export_single(mesh, theta, batch_size)
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        np.savez(save_path, **data)
        print(f"Exported theta={theta:.4f} -> {save_path}  ({len(data['phi'])} faces)")
        return data

    def export_k(self, mesh, out_dir, k=8, batch_size=4096):
        os.makedirs(out_dir, exist_ok=True)
        fixed = getattr(self, '_fixed_theta', None)
        if fixed is not None:
            path = os.path.join(out_dir, "vecfield_000.npz")
            self.export(mesh, path, theta=fixed, batch_size=batch_size)
            print(f"\nFixed-angle mode: exported 1 angle to {out_dir}/")
        else:
            angles = np.linspace(0, 2 * np.pi, k, endpoint=False)
            for i, theta in enumerate(angles):
                path = os.path.join(out_dir, f"vecfield_{i:03d}.npz")
                self.export(mesh, path, theta=float(theta), batch_size=batch_size)
            print(f"\nExported {k} angles to {out_dir}/")

    # --------------------------------------------------------
    # Save / Load
    # --------------------------------------------------------

    def save(self, path):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        torch.save({
            'model_phi': self.model_phi.state_dict(),
            'model_psi': self.model_psi.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'input_size': self.input_size,
            'geo_size': self.geo_size,
            'hidden': self.hidden,
            'lr': self.lr,
            'constraint_face_ids': getattr(self, '_constraint_face_ids', None),
            'constraint_vectors': getattr(self, '_constraint_vectors', None),
            'fixed_theta': getattr(self, '_fixed_theta', None),
            'nb_pts': getattr(self, '_nb_pts', None),
            'nb_normals': getattr(self, '_nb_normals', None),
            'nb_sdf': getattr(self, '_nb_sdf', None),
        }, path)
        print(f"  saved -> {path}")

    @classmethod
    def load(cls, path, device='cpu', sdf=None):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        obj = cls(
            input_size=ckpt.get('input_size', 3),
            hidden=ckpt.get('hidden', 128),
            lr=ckpt.get('lr', 1e-3),
            device=device,
            sdf=sdf,
        )
        obj.model_phi.load_state_dict(ckpt['model_phi'])
        obj.model_psi.load_state_dict(ckpt['model_psi'])
        obj.optimizer.load_state_dict(ckpt['optimizer'])
        obj._constraint_face_ids = ckpt.get('constraint_face_ids')
        obj._constraint_vectors = ckpt.get('constraint_vectors')
        obj._fixed_theta = ckpt.get('fixed_theta', None)
        nb = ckpt.get('nb_pts', None)
        if nb is not None:
            obj._nb_pts = nb
            obj._nb_normals = ckpt['nb_normals']
            obj._nb_sdf = ckpt['nb_sdf']
        print(f"Loaded from {path}")
        return obj


# ============================================================
# Polyscope visualizer
# ============================================================

def visualize_polyscope(npz_path):
    import polyscope as ps

    data = np.load(npz_path, allow_pickle=True)
    V = data['mesh_vertices']
    F = data['mesh_faces']
    fc = data['face_centers']
    u = data['u']
    u_pot = data['u_pot']
    u_stream = data['u_stream']
    phi = data['phi']
    psi = data['psi']
    c_ids = data['constraint_face_ids']
    c_vecs_rot = data['constraint_vectors_rotated']

    ps.init()
    ps.set_up_dir("z_up")
    ps.set_ground_plane_mode("shadow_only")

    sm = ps.register_surface_mesh("mesh", V, F, smooth_shade=True)
    sm.add_scalar_quantity("phi", phi, defined_on='faces', cmap='viridis', enabled=False)
    sm.add_scalar_quantity("psi", psi, defined_on='faces', cmap='coolwarm', enabled=True)
    sm.add_scalar_quantity("|u|", np.linalg.norm(u, axis=1),
                           defined_on='faces', cmap='reds', enabled=False)
    sm.add_vector_quantity("u", u, defined_on='faces',
                           vectortype='ambient', enabled=True,
                           length=0.02, color=(0.2, 0.5, 0.9))
    sm.add_vector_quantity("u_pot", u_pot, defined_on='faces',
                           vectortype='ambient', enabled=False,
                           length=0.02, color=(0.9, 0.5, 0.1))
    sm.add_vector_quantity("u_stream", u_stream, defined_on='faces',
                           vectortype='ambient', enabled=False,
                           length=0.02, color=(0.1, 0.7, 0.3))

    if len(c_ids) > 0 and len(c_vecs_rot) > 0:
        cloud = ps.register_point_cloud("constraints", fc[c_ids],
                                        radius=0.008, color=(1, 0.2, 0.2))
        cloud.add_vector_quantity("target", c_vecs_rot,
                                  vectortype='ambient', length=0.03,
                                  color=(0.9, 0.1, 0.1), enabled=True)
        cloud.add_vector_quantity("predicted", u[c_ids],
                                  vectortype='ambient', length=0.03,
                                  color=(0.1, 0.1, 0.9), enabled=True)
    ps.show()


if __name__ == '__main__':
    import sys
    if '--vis' in sys.argv:
        visualize_polyscope(sys.argv[sys.argv.index('--vis') + 1])
    else:
        print("Usage: python SurfaceVectorFieldDesign.py --vis <out_dir>/vecfield_000.npz")
        print("Training: python train_vecfield.py --help")
