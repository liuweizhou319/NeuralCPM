"""
Mesh_geometry - Load OBJ triangle mesh, sample surface & narrow band
=====================================================================

Drop-in replacement for Torus_geometry / Sphere_geometry.
Provides the same function signatures so SurfaceDFVecLaplacianBasis
can use them directly.

Supports adaptive narrow-band sampling via anchor-based Voronoi ray
bounds.  Call  mesh.build_adaptive_band(...)  once after construction,
and subsequent  mesh.sample_narrow_band(...)  calls automatically use
per-anchor local safe radii instead of a global fixed band_width.

Usage:
    mesh = MeshGeometry('bunny.obj', device='cuda')

    # (optional) enable adaptive narrow band
    mesh.build_adaptive_band()

    # sample_surface: area-uniform (torch); sample_surface_per_face: one random point per face.
    pts, normals = mesh.sample_surface(N=3000, device='cuda')
    pts, normals, sdf = mesh.sample_narrow_band(N=2000, device='cuda')
"""

import math
import numpy as np
import matplotlib.pyplot as plt

import torch


# ============================================================
# OBJ Loader (self-contained, no dependencies)
# ============================================================

def load_obj(filepath):
    """
    Load triangle mesh from OBJ file.

    Returns
    -------
    vertices : (V, 3) numpy array
    faces : (F, 3) numpy array of int (0-indexed)
    """
    vertices = []
    faces = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if parts[0] == 'v' and len(parts) >= 4:
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == 'f':
                # Handle f v, f v/vt, f v/vt/vn, f v//vn
                face_verts = []
                for p in parts[1:]:
                    idx = int(p.split('/')[0]) - 1  # OBJ is 1-indexed
                    face_verts.append(idx)
                # Triangulate if polygon has more than 3 vertices (fan triangulation)
                for i in range(1, len(face_verts) - 1):
                    faces.append([face_verts[0], face_verts[i], face_verts[i + 1]])

    vertices = np.array(vertices, dtype=np.float32)
    faces = np.array(faces, dtype=np.int64)
    return vertices, faces


# ============================================================
# Farthest-point sampling (for anchor generation)
# ============================================================

def _farthest_point_sample(pts, M, seed=0):
    """
    Greedy farthest-point sampling.

    Parameters
    ----------
    pts : (N, 3) numpy array - candidate pool
    M   : int - number of anchors to select

    Returns
    -------
    indices : (M,) int array - indices into pts
    """
    N = len(pts)
    M = min(M, N)
    rng = np.random.RandomState(seed)
    indices = np.zeros(M, dtype=np.int64)
    indices[0] = rng.randint(N)

    # dist[i] = min distance from pts[i] to any already-selected anchor
    dist = np.full(N, np.inf, dtype=np.float64)
    for k in range(M):
        # update distances w.r.t. newly added anchor
        d = np.sqrt(((pts - pts[indices[k]]) ** 2).sum(axis=1))
        dist = np.minimum(dist, d)
        if k + 1 < M:
            indices[k + 1] = dist.argmax()

    return indices


# ============================================================
# Mesh Geometry Class
# ============================================================

class MeshGeometry:
    """
    Holds a triangle mesh and provides sampling functions
    with the same interface as the analytic geometry modules.
    """

    def __init__(self, obj_path, normalize=True, device='cpu'):
        """
        Parameters
        ----------
        obj_path : str
            Path to .obj file
        normalize : bool
            If True, center mesh and scale to fit in unit sphere
        device : str
        """
        verts_np, faces_np = load_obj(obj_path)
        print(f"Loaded {obj_path}: {len(verts_np)} vertices, {len(faces_np)} faces")

        if normalize:
            # Center
            center = (verts_np.max(axis=0) + verts_np.min(axis=0)) / 2
            verts_np = verts_np - center
            # Scale so bounding sphere radius = 1
            scale = np.sqrt((verts_np ** 2).sum(axis=1)).max()
            verts_np = verts_np / scale
            print(f"  Normalized: center={center}, scale={scale:.4f}")

        self.obj_path = obj_path
        self.vertices_np = verts_np
        self.faces_np = faces_np
        self.device = torch.device(device)

        # Precompute on CPU (will transfer to GPU as needed)
        self._precompute()

        # Adaptive band (built lazily by build_adaptive_band)
        self._adaptive_band = None

    def _precompute(self):
        """Precompute face normals, areas, vertex normals, cumulative area CDF."""
        V = self.vertices_np
        F = self.faces_np

        v0 = V[F[:, 0]]
        v1 = V[F[:, 1]]
        v2 = V[F[:, 2]]

        # Face normals (unnormalized = 2 * area * n_hat)
        cross = np.cross(v1 - v0, v2 - v0)  # (F, 3)
        face_area_2 = np.sqrt((cross ** 2).sum(axis=1) + 1e-30)
        self.face_normals_np = cross / face_area_2[:, None]  # (F, 3)
        self.face_areas_np = face_area_2 / 2.0  # (F,)

        # CDF for area-weighted sampling
        total_area = self.face_areas_np.sum()
        self.face_cdf_np = np.cumsum(self.face_areas_np) / total_area  # (F,)
        self.total_area = total_area

        # Vertex normals (area-weighted average of incident face normals)
        vertex_normals = np.zeros_like(V)  # (V, 3)
        for i in range(3):
            np.add.at(vertex_normals, F[:, i], self.face_normals_np * self.face_areas_np[:, None])
        norms = np.sqrt((vertex_normals ** 2).sum(axis=1, keepdims=True) + 1e-30)
        self.vertex_normals_np = vertex_normals / norms

        self.avg_edge = np.sqrt(total_area / len(F))

        print(f"  Precomputed: total area = {total_area:.4f}, "
              f"avg edge length ~ {self.avg_edge:.4f}")

    def to(self, device):
        """Move precomputed data to device (lazy, called on first sample)."""
        self.device = torch.device(device)
        return self

    # ================================================================
    # Adaptive narrow band - precompute
    # ================================================================

    def build_adaptive_band(
        self,
        n_anchors: int = 5000,
        knn_k: int = 16,
        band_width_max: float = 0.05,
        band_width_min: float = None,
        safety_beta: float = 0.7,
        u_min: float = 0.1,
    ):
        """
        Precompute anchor-based adaptive safe radii for narrow-band
        sampling.  After calling this, sample_narrow_band() will
        automatically use local radii instead of a fixed band_width.

        Algorithm
        ---------
        1. Farthest-point-sample M anchors on the mesh surface.
        2. For each anchor i, query its K nearest-neighbour anchors.
        3. Compute the Voronoi ray safe radius along +/-n_i:

               rho_raw^+ = min_{j: n_i.d_ij > 0}  |d_ij|^2 / (2 n_i.d_ij)
               rho_raw^- = min_{j: -n_i.d_ij > 0} |d_ij|^2 / (2 (-n_i.d_ij))

        4. Clip:  rho = clip(beta . rho_raw,  rho_min,  rho_max)

        Parameters
        ----------
        n_anchors      : number of anchors (~ mesh density)
        knn_k          : neighbours per anchor for Voronoi bound
        band_width_max : global upper clamp  (= old band_width)
        band_width_min : global lower clamp  (default 0.1 x avg_edge)
        safety_beta    : shrink factor in (0, 1)
        u_min          : minimum offset ratio to avoid near-zero offsets
        """
        if band_width_min is None:
            band_width_min = 0.1 * self.avg_edge

        dev = self.device

        # --- Step 1: generate anchor candidates via FPS ---
        pool_size = max(n_anchors * 10, 50000)
        pool_pts, pool_normals = self.sample_surface(pool_size)
        pool_pts_np = pool_pts.numpy()
        pool_normals_np = pool_normals.numpy()

        n_anchors = min(n_anchors, pool_size)
        sel = _farthest_point_sample(pool_pts_np, n_anchors)

        anchor_pts = torch.from_numpy(pool_pts_np[sel].copy()).float().to(dev)    # (M, 3)
        anchor_normals = torch.from_numpy(pool_normals_np[sel].copy()).float().to(dev)  # (M, 3)
        M = anchor_pts.shape[0]

        # --- Step 2: kNN among anchors (torch.cdist + topk) ---
        K = min(knn_k, M - 1)
        dists = torch.cdist(anchor_pts, anchor_pts)          # (M, M)
        # exclude self by setting diagonal to inf
        dists.fill_diagonal_(float('inf'))
        _, knn_idx = dists.topk(K, dim=1, largest=False)     # (M, K)

        # --- Step 3: Voronoi ray safe radii (fully vectorised) ---
        # d_ij = anchor_pts[knn_idx] - anchor_pts[:, None]
        neighbours = anchor_pts[knn_idx]                     # (M, K, 3)
        d = neighbours - anchor_pts.unsqueeze(1)             # (M, K, 3)
        dist_sq = (d * d).sum(dim=2)                         # (M, K)

        # a_ij = n_i . d_ij
        a = (anchor_normals.unsqueeze(1) * d).sum(dim=2)     # (M, K)

        INF = torch.tensor(float('inf'), device=dev)

        # rho^+: only where a > 0
        mask_plus = a > 1e-12
        candidate_plus = torch.where(mask_plus, dist_sq / (2.0 * a.clamp(min=1e-12)), INF)
        rho_plus, _ = candidate_plus.min(dim=1)              # (M,)

        # rho^-: only where a < 0
        neg_a = -a
        mask_minus = neg_a > 1e-12
        candidate_minus = torch.where(mask_minus, dist_sq / (2.0 * neg_a.clamp(min=1e-12)), INF)
        rho_minus, _ = candidate_minus.min(dim=1)            # (M,)

        # --- Step 4: clip ---
        rho_plus  = (safety_beta * rho_plus).clamp(band_width_min, band_width_max)
        rho_minus = (safety_beta * rho_minus).clamp(band_width_min, band_width_max)

        self._adaptive_band = {
            'anchor_pts': anchor_pts,          # (M, 3) on device
            'rho_plus': rho_plus,              # (M,)   on device
            'rho_minus': rho_minus,            # (M,)   on device
            'u_min': u_min,
        }

        print(f"  Adaptive band: {M} anchors, K={K}, beta={safety_beta}")
        print(f"    rho^+ range: [{rho_plus.min().item():.5f}, {rho_plus.max().item():.5f}]")
        print(f"    rho^- range: [{rho_minus.min().item():.5f}, {rho_minus.max().item():.5f}]")

    # ================================================================
    # Core: sample on surface
    # ================================================================

    def _sample_surface_barycentric(self, N, device):
        """Legacy: area-weighted random triangle + barycentric uniform."""
        r = np.random.rand(N)
        face_idx = np.searchsorted(self.face_cdf_np, r, side='right')
        face_idx = np.clip(face_idx, 0, len(self.faces_np) - 1)
        s = np.random.rand(N).astype(np.float32)
        t = np.random.rand(N).astype(np.float32)
        mask = s + t > 1.0
        s[mask] = 1.0 - s[mask]
        t[mask] = 1.0 - t[mask]
        w0 = 1.0 - s - t
        w1 = s
        w2 = t
        F = self.faces_np[face_idx]
        v0 = self.vertices_np[F[:, 0]]
        v1 = self.vertices_np[F[:, 1]]
        v2 = self.vertices_np[F[:, 2]]
        pts = w0[:, None] * v0 + w1[:, None] * v1 + w2[:, None] * v2
        n0 = self.vertex_normals_np[F[:, 0]]
        n1 = self.vertex_normals_np[F[:, 1]]
        n2 = self.vertex_normals_np[F[:, 2]]
        normals = w0[:, None] * n0 + w1[:, None] * n1 + w2[:, None] * n2
        normals = normals / (np.sqrt((normals ** 2).sum(axis=1, keepdims=True)) + 1e-30)
        return (torch.from_numpy(pts).float().to(device),
                torch.from_numpy(normals).float().to(device))

    def _sample_surface_torch(self, N, device):
        """Area-weighted random triangle + uniform barycentric (vectorized torch, GPU-friendly)."""
        dev = torch.device(device)
        cdf = torch.from_numpy(self.face_cdf_np).to(dev)
        verts = torch.from_numpy(self.vertices_np).float().to(dev)
        faces = torch.from_numpy(self.faces_np).long().to(dev)
        vnorm = torch.from_numpy(self.vertex_normals_np).float().to(dev)
        nf = faces.shape[0]

        r = torch.rand(N, device=dev, dtype=cdf.dtype)
        fi = torch.searchsorted(cdf, r, right=True).clamp_(0, nf - 1)

        s = torch.rand(N, device=dev)
        t = torch.rand(N, device=dev)
        mask = s + t > 1.0
        s = torch.where(mask, 1.0 - s, s)
        t = torch.where(mask, 1.0 - t, t)
        w0 = 1.0 - s - t
        w1 = s
        w2 = t

        tri = faces[fi]
        v0 = verts[tri[:, 0]]
        v1 = verts[tri[:, 1]]
        v2 = verts[tri[:, 2]]
        pts = w0.unsqueeze(1) * v0 + w1.unsqueeze(1) * v1 + w2.unsqueeze(1) * v2

        n0 = vnorm[tri[:, 0]]
        n1 = vnorm[tri[:, 1]]
        n2 = vnorm[tri[:, 2]]
        normals = w0.unsqueeze(1) * n0 + w1.unsqueeze(1) * n1 + w2.unsqueeze(1) * n2
        normals = normals / (normals.norm(dim=1, keepdim=True) + 1e-30)
        return pts, normals

    def sample_surface(self, N, device='cpu', **kwargs):
        """
        Area-uniform samples on the mesh (random triangle weighted by area, uniform
        barycentric in triangle). Vectorized **PyTorch** on ``device``; vertex normals
        interpolated. ``legacy=True`` uses the numpy implementation (same distribution).
        """
        N = int(N)
        if N <= 0:
            dev = torch.device(device)
            z = torch.empty(0, 3, device=dev, dtype=torch.float32)
            return z, z
        if kwargs.get("legacy", False):
            return self._sample_surface_barycentric(N, device)
        return self._sample_surface_torch(N, device)

    def sample_surface_per_face(self, device='cpu'):
        """
        One **uniform random** point per triangle (face order matches ``faces_np`` rows).

        Returns ``(pts, normals)`` each of shape ``(F, 3)``. Training pairs this with a **sparse
        face graph** (vertex one-ring: ``undirected_face_vertex_ring_edges`` in
        ``SurfaceFrameFieldSH``) instead of Euclidean kNN to avoid O(F^2) memory.
        """
        dev = torch.device(device)
        verts = torch.from_numpy(self.vertices_np).float().to(dev)
        faces = torch.from_numpy(self.faces_np).long().to(dev)
        vnorm = torch.from_numpy(self.vertex_normals_np).float().to(dev)
        F = faces.shape[0]
        if F == 0:
            z = torch.empty(0, 3, device=dev, dtype=torch.float32)
            return z, z

        s = torch.rand(F, device=dev)
        t = torch.rand(F, device=dev)
        mask = s + t > 1.0
        s = torch.where(mask, 1.0 - s, s)
        t = torch.where(mask, 1.0 - t, t)
        w0 = 1.0 - s - t
        w1 = s
        w2 = t

        tri = faces
        v0 = verts[tri[:, 0]]
        v1 = verts[tri[:, 1]]
        v2 = verts[tri[:, 2]]
        pts = w0.unsqueeze(1) * v0 + w1.unsqueeze(1) * v1 + w2.unsqueeze(1) * v2

        n0 = vnorm[tri[:, 0]]
        n1 = vnorm[tri[:, 1]]
        n2 = vnorm[tri[:, 2]]
        normals = w0.unsqueeze(1) * n0 + w1.unsqueeze(1) * n1 + w2.unsqueeze(1) * n2
        normals = normals / (normals.norm(dim=1, keepdim=True) + 1e-30)
        return pts, normals

    # # ================================================================
    # # Core: narrow band  (adaptive or fixed)
    # # ================================================================

    # def sample_narrow_band(self, N, band_width=0.05, device='cpu',
    #                        use_adaptive=True, **kwargs):
    #     """
    #     Sample points in narrow band around mesh surface.

    #     If build_adaptive_band() has been called AND use_adaptive=True,
    #     uses per-anchor local safe radii (band_width is ignored).
    #     Otherwise uses fixed band_width.

    #     Parameters
    #     ----------
    #     N : int
    #     band_width : float   (used in fixed-width mode)
    #     device : str or torch.device
    #     use_adaptive : bool  (set False to force fixed-width even if
    #                           adaptive band has been built)

    #     Returns
    #     -------
    #     pts : (N, 3)
    #     normals : (N, 3)  surface normals at closest point
    #     sdf_vals : (N,)   approximate signed distance (= offset)
    #     """
    #     if use_adaptive and self._adaptive_band is not None:
    #         return self._sample_narrow_band_adaptive(N, device)
    #     else:
    #         return self._sample_narrow_band_fixed(N, band_width, device)

    # def _sample_narrow_band_fixed(self, N, band_width, device):
    #     """Original fixed-width narrow band."""
    #     surface_pts, normals = self.sample_surface(N, device=device)

    #     offsets = (torch.rand(N, 1, device=device) - 0.5) * 2 * band_width
    #     offsets = torch.where(offsets.abs() < 0.01 * band_width,
    #                           torch.sign(offsets) * 0.01 * band_width, offsets)

    #     pts = surface_pts + offsets * normals
    #     sdf_vals = offsets.squeeze(1)
    #     return pts, normals, sdf_vals

    # def _sample_narrow_band_adaptive(self, N, device):
    #     """Adaptive narrow band using precomputed anchor safe radii (GPU)."""
    #     ab = self._adaptive_band
    #     u_min = ab['u_min']

    #     # 1. sample surface points
    #     surface_pts, normals = self.sample_surface(N, device=device)

    #     # 2. find nearest anchor for each surface sample via torch.cdist
    #     #    anchor_pts is already on self.device; move surface_pts there for cdist
    #     anchor_dev = ab['anchor_pts'].device
    #     surf_on_dev = surface_pts.to(anchor_dev)
    #     dists = torch.cdist(surf_on_dev.unsqueeze(0),
    #                         ab['anchor_pts'].unsqueeze(0)).squeeze(0)  # (N, M)
    #     idx = dists.argmin(dim=1)                                      # (N,)

    #     rho_plus  = ab['rho_plus'][idx]    # (N,) on anchor_dev
    #     rho_minus = ab['rho_minus'][idx]   # (N,)

    #     # 3. random sign: +1 or -1
    #     sign = torch.where(torch.rand(N, device=anchor_dev) < 0.5, 1.0, -1.0)

    #     # 4. random magnitude ratio in [u_min, 1]
    #     u = u_min + (1.0 - u_min) * torch.rand(N, device=anchor_dev)

    #     # 5. offset = sign * u * rho
    #     offsets = torch.where(sign > 0, u * rho_plus, -u * rho_minus)  # (N,)
    #     offsets = offsets.to(device).unsqueeze(1)                       # (N,1)

    #     pts = surface_pts + offsets * normals
    #     sdf_vals = offsets.squeeze(1)
    #     return pts, normals, sdf_vals

    # ================================================================
    # Closest point (brute force, for evaluation only - not for training)
    # ================================================================

    def closest_point(self, query_pts, device='cpu'):
        """
        Brute-force closest point projection onto mesh.
        Use only for small batches (evaluation/visualization), not in training loop.

        Parameters
        ----------
        query_pts : (N, 3) tensor

        Returns
        -------
        cp : (N, 3)  closest points on mesh
        normals : (N, 3)  interpolated vertex normals at closest points
        dists : (N,)  unsigned distances
        """
        Q = query_pts.detach().cpu().numpy()  # (N, 3)
        F = self.faces_np
        V = self.vertices_np

        v0 = V[F[:, 0]]  # (F, 3)
        v1 = V[F[:, 1]]
        v2 = V[F[:, 2]]

        # For each query point, find closest face (brute force)
        best_cp = np.zeros((len(Q), 3), dtype=np.float32)
        best_dist = np.full(len(Q), np.inf, dtype=np.float32)
        best_bary = np.zeros((len(Q), 3), dtype=np.float32)
        best_face = np.zeros(len(Q), dtype=np.int64)

        # Process in chunks to avoid memory explosion
        chunk_size = max(1, min(500, 50_000_000 // len(F)))
        for start in range(0, len(Q), chunk_size):
            end = min(start + chunk_size, len(Q))
            q = Q[start:end][:, None, :]  # (chunk, 1, 3)

            # Project onto each triangle
            e0 = (v1 - v0)[None, :, :]  # (1, F, 3)
            e1 = (v2 - v0)[None, :, :]
            w = q - v0[None, :, :]       # (chunk, F, 3)

            d00 = (e0 * e0).sum(axis=2)
            d01 = (e0 * e1).sum(axis=2)
            d11 = (e1 * e1).sum(axis=2)
            d20 = (w * e0).sum(axis=2)
            d21 = (w * e1).sum(axis=2)

            denom = d00 * d11 - d01 * d01 + 1e-30
            s = (d11 * d20 - d01 * d21) / denom
            t = (d00 * d21 - d01 * d20) / denom

            # Clamp to triangle
            s = np.clip(s, 0, 1)
            t = np.clip(t, 0, 1)
            excess = s + t - 1
            mask = excess > 0
            s[mask] -= excess[mask] / 2
            t[mask] -= excess[mask] / 2
            s = np.clip(s, 0, 1)
            t = np.clip(t, 0, 1)

            cp = v0[None, :, :] + s[:, :, None] * e0 + t[:, :, None] * e1  # (chunk, F, 3)
            dist = np.sqrt(((q - cp) ** 2).sum(axis=2) + 1e-30)  # (chunk, F)

            # Best face per query
            best_f_idx = dist.argmin(axis=1)  # (chunk,)
            for i, fi in enumerate(best_f_idx):
                gi = start + i
                if dist[i, fi] < best_dist[gi]:
                    best_dist[gi] = dist[i, fi]
                    best_cp[gi] = cp[i, fi]
                    best_bary[gi] = [1 - s[i, fi] - t[i, fi], s[i, fi], t[i, fi]]
                    best_face[gi] = fi

        # Interpolate normals at closest points
        n0 = self.vertex_normals_np[F[best_face, 0]]
        n1 = self.vertex_normals_np[F[best_face, 1]]
        n2 = self.vertex_normals_np[F[best_face, 2]]
        normals = (best_bary[:, 0:1] * n0 + best_bary[:, 1:2] * n1 + best_bary[:, 2:3] * n2)
        normals = normals / (np.sqrt((normals ** 2).sum(axis=1, keepdims=True)) + 1e-30)

        cp = torch.from_numpy(best_cp).float().to(device)
        normals = torch.from_numpy(normals).float().to(device)
        dists = torch.from_numpy(best_dist).float().to(device)
        return cp, normals, dists

    # ================================================================
    # Visualization
    # ================================================================

    def visualize(self, save_path=None):
        """Visualize mesh with surface samples, normals, narrow band."""
        fig = plt.figure(figsize=(18, 6))

        V = self.vertices_np
        F = self.faces_np

        # --- Panel 1: Mesh wireframe ---
        ax = fig.add_subplot(1, 3, 1, projection='3d')
        # Plot a subset of edges
        edges_set = set()
        for f in F[:min(len(F), 5000)]:
            for i in range(3):
                e = tuple(sorted([f[i], f[(i + 1) % 3]]))
                edges_set.add(e)
        for e in list(edges_set)[:3000]:
            ax.plot3D(*zip(V[e[0]], V[e[1]]), color='gray', linewidth=0.3, alpha=0.5)
        ax.set_title(f'Mesh ({len(V)} verts, {len(F)} faces)')
        ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
        _set_axes_equal(ax)

        # --- Panel 2: Surface samples + normals ---
        ax = fig.add_subplot(1, 3, 2, projection='3d')
        pts_s, normals_s = self.sample_surface(800)
        pts_np = pts_s.numpy()
        n_np = normals_s.numpy()
        ax.scatter(pts_np[:, 0], pts_np[:, 1], pts_np[:, 2], s=3, c='steelblue', alpha=0.6)
        step = max(1, len(pts_np) // 200)
        arrow_len = 0.05
        ax.quiver(pts_np[::step, 0], pts_np[::step, 1], pts_np[::step, 2],
                  n_np[::step, 0] * arrow_len, n_np[::step, 1] * arrow_len,
                  n_np[::step, 2] * arrow_len,
                  color='red', alpha=0.5, arrow_length_ratio=0.3, linewidth=0.5)
        ax.set_title('Surface samples + normals')
        ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
        _set_axes_equal(ax)

        # --- Panel 3: Narrow band ---
        ax = fig.add_subplot(1, 3, 3, projection='3d')
        nb_pts, _, nb_sdf = self.sample_narrow_band(1500, band_width=0.05)
        nb_np = nb_pts.numpy()
        nb_sdf_np = nb_sdf.numpy()
        sc = ax.scatter(nb_np[:, 0], nb_np[:, 1], nb_np[:, 2],
                        c=nb_sdf_np, cmap='RdBu_r', s=2, alpha=0.6)
        fig.colorbar(sc, ax=ax, shrink=0.5, label='SDF (approx)')
        ax.set_title('Narrow band (colored by offset)')
        ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
        _set_axes_equal(ax)

        fig.suptitle(f'Mesh Geometry', fontsize=14, fontweight='bold')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()


def _set_axes_equal(ax):
    """Make 3D axes have equal scale."""
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    center = limits.mean(axis=1)
    radius = (limits[:, 1] - limits[:, 0]).max() / 2
    ax.set_xlim3d([center[0] - radius, center[0] + radius])
    ax.set_ylim3d([center[1] - radius, center[1] + radius])
    ax.set_zlim3d([center[2] - radius, center[2] + radius])


# ============================================================
# Quick test
# ============================================================

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python Mesh_geometry.py <mesh.obj>")
        sys.exit(1)

    mesh = MeshGeometry(sys.argv[1], normalize=True)

    # Test surface sampling
    pts, normals = mesh.sample_surface(5000)
    print(f"Surface samples: pts {pts.shape}, normals {normals.shape}")
    dot = (normals * normals).sum(dim=1)
    print(f"  Normal norms: mean={dot.mean():.6f}")

    # Test fixed narrow band
    nb_pts, nb_n, nb_sdf = mesh.sample_narrow_band(3000, band_width=0.05)
    print(f"Fixed narrow band: pts {nb_pts.shape}, "
          f"sdf range [{nb_sdf.min():.4f}, {nb_sdf.max():.4f}]")

    # Test adaptive narrow band
    mesh.build_adaptive_band()
    nb_pts2, nb_n2, nb_sdf2 = mesh.sample_narrow_band(3000)
    print(f"Adaptive narrow band: pts {nb_pts2.shape}, "
          f"sdf range [{nb_sdf2.min():.4f}, {nb_sdf2.max():.4f}]")

    mesh.visualize(save_path='mesh_geometry.png')
