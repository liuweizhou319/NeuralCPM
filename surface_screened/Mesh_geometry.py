"""
Mesh_geometry - load an OBJ triangle mesh, normalise it, sample its surface
===========================================================================

Role in the pipeline
--------------------
The scripts that train or evaluate the network (train_screen.py,
evaluate_dirichlet_generalization.py, export_split_samples.py,
export_worst_rel_l2.py) load their mesh through MeshGeometry.  The class
therefore defines the normalised coordinate frame in which the network, the
surface / narrow-band samples and the SDF lattice of
SparseGridFDM_SDF.from_mesh() live; lengths such as --band-width and the SDF
voxel size are expressed in these units.  The FEM reference (fem_screen.py)
has its own loader that applies the same normalisation rule in float64, and
sdf_spacing.py reuses load_obj.

OBJ parsing (load_obj)
----------------------
Only 'v' lines (their first three numbers) and 'f' lines are read.  Face
corners may be written as  v,  v/vt,  v/vt/vn  or  v//vn ; polygons are
fan-triangulated.  'vn' and 'vt' lines are ignored, and negative (relative)
indices are not supported.  Vertex ids are 0-based, in the order of the 'v'
lines.

Normalisation (normalize=True, which every script of this repository passes)
----------------------------------------------------------------------------
    center = (bbox_min + bbox_max) / 2    # bounding-box centre, not the centroid
    V      = V - center
    scale  = max_i |V_i|                  # maximum vertex norm after the shift
    V      = V / scale

Everything is computed in float32.  Afterwards the mesh lies inside the unit
sphere centred at the origin, with its farthest vertex on the sphere.  center
and scale are printed at construction time; original coordinates are recovered
with  x_orig = x_norm * scale + center.

Normals
-------
Normals stored in the OBJ file ('vn' lines) are ignored.  Face normals are
recomputed from the faces (cross product of two edges) and vertex normals are
the area-weighted average of the incident face normals, so their orientation
follows the face winding: consistently outward-wound faces give outward
normals.

Sampling
--------
    mesh = MeshGeometry('data/Bunny.obj', normalize=True, device='cuda')
    pts, normals = mesh.sample_surface(N=3000, device='cuda')

sample_surface returns two float32 torch tensors of shape (N, 3) on `device`:
points drawn uniformly with respect to surface area, and unit normals obtained
by barycentric interpolation of the vertex normals.  It draws from the global
NumPy random generator (np.random.rand), so np.random.seed() makes it
reproducible.

Exposed attributes: vertices_np (V, 3) float32, faces_np (F, 3) int64,
face_normals_np (F, 3), face_areas_np (F,), face_cdf_np (F,),
vertex_normals_np (V, 3), total_area, avg_edge.
"""

import numpy as np

import torch


# ============================================================
# OBJ Loader (self-contained, no dependencies)
# ============================================================

def load_obj(filepath):
    """
    Load triangle mesh from OBJ file.

    Only 'v' and 'f' lines are read; 'vn', 'vt' and all other lines are
    ignored.  Polygons with more than three corners are fan-triangulated.

    Returns
    -------
    vertices : (V, 3) numpy array, float32
    faces : (F, 3) numpy array of int64 (0-indexed)
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
# Mesh Geometry Class
# ============================================================

class MeshGeometry:
    """
    Holds a triangle mesh (optionally normalised to the unit sphere) together
    with precomputed face / vertex normals and an area CDF, and provides
    area-uniform surface sampling.
    """

    def __init__(self, obj_path, normalize=True, device='cpu'):
        """
        Parameters
        ----------
        obj_path : str
            Path to .obj file
        normalize : bool
            If True, shift the vertices by the bounding-box centre and divide
            them by the maximum vertex norm (in float32), so that the mesh
            fits in the unit sphere centred at the origin
        device : str or torch.device
            Stored as self.device.  It has no effect on the methods of this
            class: sample_surface takes its own device argument.
        """
        verts_np, faces_np = load_obj(obj_path)
        print(f"Loaded {obj_path}: {len(verts_np)} vertices, {len(faces_np)} faces")

        if verts_np.size == 0 or faces_np.size == 0:
            hint = ""
            if str(obj_path).lower().endswith(".json"):
                hint = (
                    " This path looks like JSON (source spec): use it as "
                    "--source-spec and pass a .obj mesh as the first argument."
                )
            raise ValueError(
                f"No vertices or faces loaded from {obj_path!r}.{hint} "
                "Check that the file is a valid Wavefront OBJ with 'v' and 'f' lines."
            )

        if normalize:
            # Center: shift by the bounding-box centre
            center = (verts_np.max(axis=0) + verts_np.min(axis=0)) / 2
            verts_np = verts_np - center
            # Scale by the maximum vertex norm so bounding sphere radius = 1
            scale = np.sqrt((verts_np ** 2).sum(axis=1)).max()
            verts_np = verts_np / scale
            print(f"  Normalized: center={center}, scale={scale:.4f}")

        self.obj_path = obj_path
        self.vertices_np = verts_np
        self.faces_np = faces_np
        self.device = torch.device(device)

        # Precompute on CPU with NumPy (sample_surface moves its samples to
        # the requested device)
        self._precompute()

        # Unused placeholder: adaptive narrow-band sampling is not part of
        # this release, so this is always None.
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

        # Characteristic length sqrt(mean face area): a rough proxy for the
        # edge length, only reported in the message below
        self.avg_edge = np.sqrt(total_area / len(F))

        print(f"  Precomputed: total area = {total_area:.4f}, "
              f"avg edge length ~ {self.avg_edge:.4f}")

    # ================================================================
    # Core: sample on surface
    # ================================================================

    def sample_surface(self, N, device='cpu', **kwargs):
        """
        Area-uniform sampling on mesh surface.

        Faces are picked with probability proportional to their area, then a
        point is drawn uniformly inside each picked triangle.  The three
        np.random.rand(N) calls consume the global NumPy random stream.

        Parameters
        ----------
        N : int
            Number of samples
        device : str or torch.device
            Device of the returned tensors
        **kwargs
            Accepted and ignored

        Returns
        -------
        pts : (N, 3) float32 tensor
            Sample positions on the surface
        normals : (N, 3) float32 tensor
            Unit normals, barycentric interpolation of the vertex normals
        """
        # Pick random faces, area-weighted
        r = np.random.rand(N)
        face_idx = np.searchsorted(self.face_cdf_np, r, side='right')
        face_idx = np.clip(face_idx, 0, len(self.faces_np) - 1)

        # Random barycentric coordinates (uniform in triangle)
        s = np.random.rand(N).astype(np.float32)
        t = np.random.rand(N).astype(np.float32)
        mask = s + t > 1.0
        s[mask] = 1.0 - s[mask]
        t[mask] = 1.0 - t[mask]
        w0 = 1.0 - s - t
        w1 = s
        w2 = t

        F = self.faces_np[face_idx]  # (N, 3)
        v0 = self.vertices_np[F[:, 0]]  # (N, 3)
        v1 = self.vertices_np[F[:, 1]]
        v2 = self.vertices_np[F[:, 2]]

        pts = w0[:, None] * v0 + w1[:, None] * v1 + w2[:, None] * v2

        # Normals: interpolate vertex normals at barycentric coords
        n0 = self.vertex_normals_np[F[:, 0]]
        n1 = self.vertex_normals_np[F[:, 1]]
        n2 = self.vertex_normals_np[F[:, 2]]
        normals = w0[:, None] * n0 + w1[:, None] * n1 + w2[:, None] * n2
        normals = normals / (np.sqrt((normals ** 2).sum(axis=1, keepdims=True)) + 1e-30)

        pts = torch.from_numpy(pts).float().to(device)
        normals = torch.from_numpy(normals).float().to(device)
        return pts, normals
