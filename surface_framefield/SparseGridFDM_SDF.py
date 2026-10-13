"""
SparseGridFDM_SDF - sparse narrow-band signed distance field (SDF) on an fvdb
lattice, with finite-difference derivatives.

Precomputed per lattice vertex: f, grad f, H_f, J_n, dJ_n/dx
All queried at arbitrary points via trilinear interpolation.

Role in the pipeline
--------------------
train_frame_field.py builds a SparseGridFDM_SDF from the normalised mesh
(MeshGeometry) and passes it to SurfaceFrameFieldSH as the fixed geometry
oracle: normal n(x), Hessian (principal curvature directions), normal
Jacobian J_n and closest point.  Nothing is cached (the grid is rebuilt by
every script run) and the grid parameters are not saved with the model, so a
script that reloads a checkpoint must build the grid with the same resolution,
padding and halo as training.

Sign convention
---------------
f is POSITIVE outside the mesh and NEGATIVE inside, so n = grad f / |grad f|
is the outward normal.

Construction (SparseGridFDM_SDF.from_mesh)
------------------------------------------
1. fvdb voxelises the mesh surface on a lattice with spacing

       voxel = (1 + 2*padding) * L_max / (resolution - 1)

   where L_max is the longest bounding-box edge of the normalised mesh.  The
   lattice origin is bbox_min - padding * L_max.
2. The surface voxels are dilated by fd_halo voxel layers in every direction
   (a cube of (2*fd_halo + 1)^3 voxels around each surface voxel).  The
   lattice vertices are the 8 corners of every dilated voxel.
3. fcpw gives the unsigned distance at every lattice vertex.  The sign comes
   from the first available source among: libigl fast winding number, fcpw
   Scene.contains (CPU scene only), sign of the closest-point normal.  If
   none is available the distance stays unsigned and a warning is printed.
4. Finite differences with step `voxel` give grad f, the Hessian H, the
   normal Jacobian J_n = (I - n n^T) H / |grad f| and dJ_n/dx at every
   lattice vertex.

Parameters
----------
resolution : number of lattice nodes along the longest edge of the padded
             bounding box; it enters only through the voxel-size formula.
padding    : fraction of L_max added on each side of the bounding box.  The
             grid is sparse, so padding adds no voxels: it scales the voxel
             size by (1 + 2*padding) and shifts the lattice origin.
fd_halo    : halo of the band, in voxel layers added on every side of the
             surface voxels (integer; a minimum of 2 is enforced).  The
             physical half-width of the band is about fd_halo * voxel.

Query API (x is a (B, 3) tensor in the normalised mesh frame)
-------------------------------------------------------------
    f(x)                     -> (B, 1)        signed distance
    f_and_grad(x)            -> (B, 1), (B, 3)
    normal(x)                -> (B, 3)        grad f / |grad f|
    hessian(x)               -> (B, 3, 3)
    normal_jacobian(x)       -> (B, 3, 3)     J[k, i] = d n_k / d x_i
    normal_jacobian_deriv(x) -> (B, 3, 3, 3)  D[k, i, j] = d J[k, i] / d x_j
    closest_point(x)         -> (B, 3)        x - f(x) * n(x), a single step

Queries outside the band return 0: missing lattice vertices contribute zero,
so f, normal, hessian and normal_jacobian vanish there and closest_point
returns x unchanged.  The interpolation weights are not renormalised, so in
cells that are only partly inside the band the values fade towards zero.

Requirements
------------
CUDA, fvdb and fcpw are hard requirements (the lattice is always built on
'cuda').  libigl is optional and only used for the winding-number sign.
With the fcpw wheel from pip the two messages

    [fcpw] GPU path failed: ...
    [fcpw] falling back to CPU scene

are expected: the fcpw GPU path needs a GPU-enabled fcpw build plus the
environment variable FCPW_DIR; otherwise the CPU BVH is used.
"""

import os
import time
import numpy as np
import torch

try:
    import fvdb
    _HAS_FVDB = True
except ImportError:
    fvdb = None
    _HAS_FVDB = False

try:
    import fcpw
    _HAS_FCPW = True
except ImportError:
    fcpw = None
    _HAS_FCPW = False


# ----------------------------------------------------------
# fcpw compatibility shims
#
# The fcpw Python bindings exist with snake_case names (scene_3D, ...) and
# with CamelCase names (Scene3D, ...), and some method signatures differ
# between builds.  Each helper below tries the known variants in a fixed
# order.
# ----------------------------------------------------------

def _fcpw_make_scene():
    """Create an empty fcpw scene (snake_case: scene_3D; CamelCase: Scene3D)."""
    if not _HAS_FCPW:
        raise ImportError("fcpw")
    if hasattr(fcpw, "scene_3D"):
        return fcpw.scene_3D()
    return fcpw.Scene3D()


def _fcpw_aggregate_bvh_surface_area():
    """Return the surface-area-heuristic BVH aggregate type of either binding."""
    if hasattr(fcpw, "aggregate_type"):
        return fcpw.aggregate_type.bvh_surface_area
    return fcpw.AggregateType.Bvh_SurfaceArea


def _fcpw_scene_build(scene, aggregate_type, build_vectorized):
    """scene.build API varies: 2 args vs 4 args (print_stats, reduce_memory)."""
    try:
        scene.build(aggregate_type, build_vectorized)
    except TypeError:
        scene.build(aggregate_type, build_vectorized, False, False)


def _fcpw_make_interaction_list():
    """Create the (CPU) interaction list that receives closest-point results."""
    if hasattr(fcpw, "interaction_3D_list"):
        return fcpw.interaction_3D_list()
    return fcpw.Interaction3DList()


def _fcpw_make_gpu_scene(fcpw_dir):
    """Create a GPU scene; fcpw_dir is the fcpw source directory (FCPW_DIR)."""
    if hasattr(fcpw, "gpu_scene_3D"):
        try:
            return fcpw.gpu_scene_3D(fcpw_dir, False)
        except TypeError:
            return fcpw.gpu_scene_3D(fcpw_dir)
    return fcpw.GPUScene3D(fcpw_dir)


def _fcpw_transfer_to_gpu(gpu_scene, cpu_scene, device_backend="default"):
    """gpu_scene.transfer_to_gpu API varies: with or without a backend argument."""
    try:
        gpu_scene.transfer_to_gpu(cpu_scene, device_backend)
    except TypeError:
        gpu_scene.transfer_to_gpu(cpu_scene)


def _fcpw_make_gpu_interaction_list():
    """Create the GPU interaction list that receives closest-point results."""
    if hasattr(fcpw, "gpu_interaction_list"):
        return fcpw.gpu_interaction_list()
    return fcpw.GPUInteractionList()


def _fcpw_interaction_positions(interactions):
    """Closest-point positions as an (N, 3) float32 array."""
    if hasattr(interactions, "get_positions"):
        return np.asarray(interactions.get_positions(), dtype=np.float32)
    return np.array([getattr(i, "p", i) for i in interactions], dtype=np.float32)


def _fcpw_interaction_normals(interactions):
    """Closest-point normals as an (N, 3) float32 array, or None if unavailable."""
    if hasattr(interactions, "get_normals"):
        return np.asarray(interactions.get_normals(), dtype=np.float32)
    if not hasattr(interactions, "__iter__") or len(interactions) == 0:
        return None
    first = interactions[0]
    for attr in ("n", "ng", "normal"):
        if hasattr(first, attr):
            return np.array([getattr(i, attr) for i in interactions], dtype=np.float32)
    return None


def _fcpw_interaction_distances(interactions):
    """Unsigned distances as a float32 array, or None if there is no bulk getter."""
    if hasattr(interactions, "get_distances"):
        return np.asarray(interactions.get_distances(), dtype=np.float32)
    return None


def _fcpw_find_closest_points(scene, q, sqr_max, interactions, want_normals=True):
    """Try (kw record_normal) -> (kw record_normals) -> (positional flag) -> (no flag at all)."""
    try:
        scene.find_closest_points(q, sqr_max, interactions, record_normal=want_normals)
        return
    except TypeError:
        pass
    try:
        scene.find_closest_points(q, sqr_max, interactions, record_normals=want_normals)
        return
    except TypeError:
        pass
    try:
        scene.find_closest_points(q, sqr_max, interactions, want_normals)
        return
    except TypeError:
        pass
    scene.find_closest_points(q, sqr_max, interactions)


def _fcpw_contains(scene, q):
    """Returns int32 array (1 inside, 0 outside) using Scene.contains; None if unavailable."""
    if not hasattr(scene, "contains"):
        return None
    try:
        result = np.zeros(len(q), dtype=np.int32)
        scene.contains(q, result)
        return result
    except Exception as e:
        print(f"  [fcpw] scene.contains failed ({e}); will use normal-based sign")
        return None


def _winding_inside(V, F, q, threshold=0.5):
    """
    Generalized winding-number inside test (works on non-watertight / messy meshes).
    Returns bool array (True = inside), or None if libigl is not installed,
    does not provide fast_winding_number_for_meshes, or the call fails.
    """
    try:
        import igl
    except ImportError:
        return None
    if not hasattr(igl, "fast_winding_number_for_meshes"):
        return None
    Vd = np.ascontiguousarray(V.astype(np.float64))
    Fd = np.ascontiguousarray(F.astype(np.int32))
    qd = np.ascontiguousarray(q.astype(np.float64))
    try:
        w = igl.fast_winding_number_for_meshes(Vd, Fd, qd)
    except Exception as e:
        print(f"  [igl] fast_winding_number failed ({e})")
        return None
    return np.asarray(w).reshape(-1) > threshold


def _build_fcpw_scene(vertices, faces, use_gpu=True):
    """
    Build one FCPW scene once, then reuse it for all query batches.

    GPU mode requires:
      - a Python build of fcpw with GPU support
      - non-vectorized CPU BVH build
      - env var FCPW_DIR pointing to the fcpw source directory
        (the directory passed to GPUScene3D(...))
    It raises if one of these is missing; the caller then retries with
    use_gpu=False, which builds a vectorized CPU BVH.
    """
    if not _HAS_FCPW:
        raise ImportError("fcpw is not installed. Please `pip install fcpw` or build it with GPU support.")

    positions = vertices.astype(np.float32, copy=False)
    indices = faces.astype(np.int32, copy=False)

    scene = _fcpw_make_scene()
    scene.set_object_count(1)
    scene.set_object_vertices(positions, 0)
    scene.set_object_triangles(indices, 0)

    agg = _fcpw_aggregate_bvh_surface_area()

    if use_gpu:
        # IMPORTANT: the fcpw README says GPU transfer requires non-vectorized CPU BVH
        build_vectorized_cpu_bvh = False
        _fcpw_scene_build(scene, agg, build_vectorized_cpu_bvh)

        fcpw_dir = os.environ.get("FCPW_DIR", None)
        if fcpw_dir is None:
            raise RuntimeError(
                "GPU FCPW requires env var FCPW_DIR to point to the fcpw source directory "
                "(the path passed to gpu_scene_3D / GPUScene3D)."
            )

        gpu_scene = _fcpw_make_gpu_scene(fcpw_dir)
        _fcpw_transfer_to_gpu(gpu_scene, scene)
        return gpu_scene
    else:
        build_vectorized_bvh = True
        _fcpw_scene_build(scene, agg, build_vectorized_bvh)
        return scene


def _query_fcpw_signed_distance(scene, query_points, V=None, F=None):
    """
    Query signed distance using closest points from FCPW.

    Sign convention:  POSITIVE outside, NEGATIVE inside

    Sign source priority:
      1. libigl `fast_winding_number_for_meshes` (robust on non-watertight)
      2. FCPW `Scene.contains` (only valid for watertight meshes)
      3. closest-point normal sign  dot(q - cp, n)
      4. UNSIGNED with loud warning
    """
    q = query_points.astype(np.float32, copy=False)
    squared_max_radii = np.full((len(q),), np.inf, dtype=np.float32)

    cname = scene.__class__.__name__
    is_gpu = ("GPUScene" in cname) or cname.lower().startswith("gpu_scene")
    if is_gpu:
        interactions = _fcpw_make_gpu_interaction_list()
    else:
        interactions = _fcpw_make_interaction_list()

    # IMPORTANT: ask FCPW to fill normals (default is False -> all-zero normals
    # would silently fall through and yield UNSIGNED distance).
    _fcpw_find_closest_points(scene, q, squared_max_radii, interactions,
                              want_normals=True)

    # Prefer the bulk distance getter; fall back to recomputing from positions.
    d = _fcpw_interaction_distances(interactions)
    cp = None
    if d is None:
        cp = _fcpw_interaction_positions(interactions)
        d = np.linalg.norm(q - cp, axis=1).astype(np.float32)

    # --- Sign recovery (priority order) ---

    # 1) Generalized winding number (robust to non-watertight, holes, flips).
    if V is not None and F is not None:
        inside_w = _winding_inside(V, F, q)
        if inside_w is not None:
            return d * np.where(inside_w, -1.0, 1.0).astype(np.float32)

    # 2) FCPW Scene.contains  (CPU-only; *must* be watertight).
    inside = _fcpw_contains(scene, q) if not is_gpu else None
    if inside is not None:
        n_in = int(inside.sum())
        if 0 < n_in < len(inside):
            return d * np.where(inside.astype(bool), -1.0, 1.0).astype(np.float32)
        # All-inside or all-outside -> almost certainly a non-watertight mesh.
        # Fall through to normal-based sign and warn.
        print(f"  [warning] Scene.contains degenerate ({n_in}/{len(inside)} inside); "
              "mesh likely not watertight. Falling back to normal-based sign.")

    # 3) Normal-based sign:  outward normals -> outside => dot(q - cp, n) > 0
    if cp is None:
        cp = _fcpw_interaction_positions(interactions)
    n_cp = _fcpw_interaction_normals(interactions)
    if n_cp is not None and np.any(np.linalg.norm(n_cp, axis=1) > 0):
        dot = np.sum((q - cp) * n_cp, axis=1)
        sgn = np.where(dot >= 0, 1.0, -1.0).astype(np.float32)
        return d * sgn

    # 4) Last resort: unsigned (warn loudly so it's not silent).
    print("  [warning] FCPW: cannot recover sign (no winding, no contains, no normals). "
          "Returning UNSIGNED distance.")
    return d


def _compute_sdf_batched(V, F, pts, bs=50000):
    """
    Build FCPW scene ONCE, then query in batches.
    Falls back to CPU FCPW if GPU path is unavailable.

    A GPU scene is attempted whenever CUDA is available; it needs a
    GPU-enabled fcpw build and the environment variable FCPW_DIR (see
    _build_fcpw_scene).  While FCPW_DIR is unset - the default, since
    installing fcpw with pip does not set it - the attempt fails, so the
    'GPU path failed ... falling back to CPU scene' messages are expected.
    """
    use_gpu = torch.cuda.is_available()
    t0 = time.time()

    try:
        scene = _build_fcpw_scene(V, F, use_gpu=use_gpu)
        mode = "GPU" if use_gpu else "CPU"
        print(f"  [fcpw] scene built on {mode}")
    except Exception as e:
        print(f"  [fcpw] GPU path failed: {e}")
        print("  [fcpw] falling back to CPU scene")
        scene = _build_fcpw_scene(V, F, use_gpu=False)
        mode = "CPU"

    t1 = time.time()
    print(f"  [fcpw] scene built on {mode} in {t1 - t0:.3f}s")

    out = np.empty(len(pts), dtype=np.float32)
    for i in range(0, len(pts), bs):
        j = min(i + bs, len(pts))
        tb0 = time.time()
        out[i:j] = _query_fcpw_signed_distance(scene, pts[i:j], V=V, F=F)
        tb1 = time.time()
        print(f"  [fcpw] query {i}:{j} / {len(pts)}  batch={j-i}  time={tb1 - tb0:.3f}s")

    t2 = time.time()
    print(f"  [fcpw] total sdf query time: {t2 - t1:.3f}s   overall: {t2 - t0:.3f}s")
    return out


class SparseGridFDM_SDF:
    """
    Narrow-band signed distance field of a triangle mesh, stored on a sparse
    voxel lattice and queried by trilinear interpolation.

    Build it with SparseGridFDM_SDF.from_mesh(mesh, resolution=..., padding=...,
    fd_halo=..., device=...); the constructor only wraps precomputed arrays.

    Sign convention: f > 0 outside the mesh, f < 0 inside; normal(x) is the
    outward normal grad f / |grad f|.

    Lattice:
        voxel = (1 + 2*padding) * L_max / (resolution - 1)
    with L_max the longest bounding-box edge of the normalised mesh.
      - resolution : lattice nodes along the longest edge of the padded
                     bounding box.
      - padding    : fraction of L_max added on each side of the bounding box
                     (scales the voxel size and shifts the lattice origin; it
                     adds no voxels because the grid is sparse).
      - fd_halo    : halo of the band, in voxel layers added on every side of
                     the surface voxels; a minimum of 2 is enforced.

    Query API (x: (B, 3) tensor in the normalised mesh frame):
        f(x)                     -> (B, 1)
        f_and_grad(x)            -> (B, 1), (B, 3)
        normal(x)                -> (B, 3)
        hessian(x)               -> (B, 3, 3)
        normal_jacobian(x)       -> (B, 3, 3)     J[k, i] = d n_k / d x_i
        normal_jacobian_deriv(x) -> (B, 3, 3, 3)  D[k, i, j] = d J[k, i] / d x_j
        closest_point(x)         -> (B, 3)
    Queries outside the band return 0 (closest_point then returns x itself).

    Hard requirements: CUDA, fvdb and fcpw; libigl is optional (winding-number
    sign).  The fcpw message 'GPU path failed ... falling back to CPU scene'
    is expected with the fcpw wheel from pip.

    Stored fields (one extra all-zero sentinel row is appended to each):
    vertex_sdf (N+1,), vertex_grad (N+1, 3), vertex_hess (N+1, 3, 3),
    vertex_Jn (N+1, 3, 3), vertex_dJn (N+1, 3, 3, 3); voxel_size, origin,
    device.
    """

    def __init__(self, sdf_vals, grad_vals, hess_vals, Jn_vals, dJn_vals,
                 lut, ijk_min, origin, voxel_size, device='cpu'):
        self.device = torch.device(device)
        self.voxel_size = float(voxel_size)
        self.origin = torch.tensor(origin, dtype=torch.float32, device=self.device)
        self._ijk_min = torch.tensor(ijk_min, dtype=torch.long, device=self.device)
        self._lut = torch.from_numpy(lut.astype(np.int64)).to(self.device)
        self._lut_shape = torch.tensor(lut.shape, dtype=torch.long, device=self.device)
        n = len(sdf_vals)
        self._sentinel = n

        def _pad(arr):
            z = np.zeros((1,) + arr.shape[1:], dtype=np.float32)
            return torch.from_numpy(np.concatenate([arr, z], axis=0)).to(self.device)

        self.vertex_sdf  = _pad(sdf_vals.reshape(-1, 1)).squeeze(-1)  # (N+1,)
        self.vertex_grad = _pad(grad_vals)    # (N+1, 3)
        self.vertex_hess = _pad(hess_vals)    # (N+1, 3, 3)
        self.vertex_Jn   = _pad(Jn_vals)      # (N+1, 3, 3)
        self.vertex_dJn  = _pad(dJn_vals)     # (N+1, 3, 3, 3)

        print(f"SparseGridFDM_SDF: {n} verts, lut={tuple(lut.shape)}, "
              f"voxel={voxel_size:.5f}, device={self.device}")

    # ----------------------------------------------------------
    # Build
    # ----------------------------------------------------------

    @classmethod
    def from_mesh(cls, mesh, resolution=96, padding=0.15, fd_halo=2, device='cpu'):
        """
        Build the SDF of `mesh` (any object with vertices_np and faces_np,
        e.g. MeshGeometry).

        voxel = (1 + 2*padding) * L_max / (resolution - 1), with L_max the
        longest bounding-box edge of the mesh; fd_halo is the halo in voxel
        layers (at least 2).  The lattice is always built on CUDA; `device`
        is where the resulting tensors are stored and queried.
        """
        assert _HAS_FVDB, "fvdb required"
        assert torch.cuda.is_available(), "CUDA required for fvdb"
        V = mesh.vertices_np.astype(np.float32)
        F = mesh.faces_np.astype(np.int32)
        mn, mx = V.min(0), V.max(0)
        pad = padding * (mx - mn).max()
        bb_min = mn - pad
        bb_max = mx + pad
        vs = float((bb_max - bb_min).max() / (resolution - 1))
        return cls._build(V, F, bb_min, vs, max(2, int(fd_halo)), device)

    @classmethod
    def _build(cls, V, F, bb_min, vs, fd_halo, device):
        t0 = time.time()
        dev = torch.device('cuda')

        origin_t = torch.tensor(bb_min.tolist(), dtype=torch.float32, device=dev)
        v_t = torch.from_numpy(V).float().to(dev)
        f_t = torch.from_numpy(F).long().to(dev)

        # 1) mesh voxels
        # NOTE: this call works with fvdb-core 0.3.x - 0.4.x.  Newer releases
        # changed the GridBatch constructor; there the equivalent call is
        # fvdb.GridBatch.from_mesh(...) with the same arguments (untested).
        mesh_grid = fvdb.GridBatch(impl='hash').from_mesh(
            fvdb.JaggedTensor([v_t]), fvdb.JaggedTensor([f_t]),
            voxel_sizes=vs, origins=origin_t)
        surf_ijk = mesh_grid.ijk.jdata.to(dev).int()
        print(f"  [fvdb] surface voxels: {surf_ijk.shape[0]}")

        # 2) dilate
        r = fd_halo
        offs = torch.arange(-r, r+1, device=dev, dtype=torch.int32)
        dx, dy, dz = torch.meshgrid(offs, offs, offs, indexing='ij')
        d = torch.stack([dx.reshape(-1), dy.reshape(-1), dz.reshape(-1)], 1)
        dilated = torch.unique((surf_ijk[:, None, :] + d[None, :, :]).reshape(-1, 3), dim=0)
        print(f"  [fvdb] dilated voxels: {dilated.shape[0]} (halo={fd_halo})")

        # 3) vertex set = voxel corners
        corners = torch.tensor(
            [[0,0,0],[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,0,1],[0,1,1],[1,1,1]],
            dtype=torch.int32, device=dev)
        verts = torch.unique(
            (dilated[:, None, :] + corners[None, :, :]).reshape(-1, 3), dim=0)
        verts_np = verts.cpu().numpy().astype(np.int32)
        n_vert = len(verts_np)
        print(f"  [fvdb] vertices: {n_vert}")

        # 4) SDF
        # Lattice vertex (i, j, k) sits at bb_min + (i, j, k) * vs, i.e. voxel
        # ijk is treated as the cell with corners ijk .. ijk + 1.  Keep this
        # convention: the published results depend on it.
        world = bb_min[None, :] + verts_np.astype(np.float32) * vs
        sdf_vals = _compute_sdf_batched(V, F, world)

        # 5) FDM: grad, hess, Jn, dJn
        grad_v, hess_v, Jn_v, dJn_v, lut, ijk_min = \
            cls._fdm_all(verts_np, sdf_vals, vs)

        print(f"  [fvdb] build done in {time.time()-t0:.1f}s")
        return cls(sdf_vals, grad_v, hess_v, Jn_v, dJn_v,
                   lut, ijk_min, bb_min.astype(np.float32), vs, device)

    @staticmethod
    def _fdm_all(vertex_ijk, sdf_vals, h):
        """Compute grad, hess, J_n, dJ_n/dx via finite differences."""
        n = len(vertex_ijk)
        ijk_min = vertex_ijk.min(axis=0)
        ijk_max = vertex_ijk.max(axis=0)
        lut_dims = (ijk_max - ijk_min + 1).astype(np.int64)
        rel = (vertex_ijk - ijk_min).astype(np.int64)

        lut = np.full(tuple(lut_dims), n, dtype=np.int64)
        lut[rel[:, 0], rel[:, 1], rel[:, 2]] = np.arange(n, dtype=np.int64)

        phi = np.concatenate([sdf_vals.astype(np.float32),
                              np.zeros(1, dtype=np.float32)])
        phi0 = sdf_vals.astype(np.float32)
        h2 = h * h

        def gather(offset):
            q = rel + np.asarray(offset, dtype=np.int64)[None, :]
            ok = ((q[:, 0] >= 0) & (q[:, 0] < lut_dims[0]) &
                  (q[:, 1] >= 0) & (q[:, 1] < lut_dims[1]) &
                  (q[:, 2] >= 0) & (q[:, 2] < lut_dims[2]))
            idx = np.full(n, n, dtype=np.int64)
            idx[ok] = lut[q[ok, 0], q[ok, 1], q[ok, 2]]
            return phi[idx], idx < n

        # --- finite-difference helpers on the SDF samples (phi via gather, phi0) ---
        def first_deriv(axis):
            e = np.zeros(3, dtype=np.int64); e[axis] = 1
            p1, vp = gather(e); m1, vm = gather(-e)
            out = np.zeros(n, dtype=np.float32)
            c = vp & vm
            out[c] = (p1[c] - m1[c]) / (2*h)
            f1 = (~c) & vp; out[f1] = (p1[f1] - phi0[f1]) / h
            b1 = (~c) & (~vp) & vm; out[b1] = (phi0[b1] - m1[b1]) / h
            return out

        def second_diag(axis):
            e = np.zeros(3, dtype=np.int64); e[axis] = 1
            p1, vp = gather(e); m1, vm = gather(-e)
            out = np.zeros(n, dtype=np.float32)
            c = vp & vm
            out[c] = (p1[c] - 2*phi0[c] + m1[c]) / h2
            return out

        def mixed(a, b):
            ea = np.zeros(3, dtype=np.int64); ea[a] = 1
            eb = np.zeros(3, dtype=np.int64); eb[b] = 1
            pp, v1 = gather(ea+eb); pm, v2 = gather(ea-eb)
            mp, v3 = gather(-ea+eb); mm, v4 = gather(-ea-eb)
            ok = v1 & v2 & v3 & v4
            out = np.zeros(n, dtype=np.float32)
            out[ok] = (pp[ok] - pm[ok] - mp[ok] + mm[ok]) / (4*h2)
            return out

        # --- grad ---
        grad_v = np.stack([first_deriv(i) for i in range(3)], axis=1)

        # --- hess ---
        hess_v = np.zeros((n, 3, 3), dtype=np.float32)
        for i in range(3):
            hess_v[:, i, i] = second_diag(i)
        for a, b in [(0,1),(0,2),(1,2)]:
            v = mixed(a, b)
            hess_v[:, a, b] = hess_v[:, b, a] = v

        # --- J_n = (1/|g|) (I - nn^T) H  at each vertex ---
        gn = np.linalg.norm(grad_v, axis=1, keepdims=True).clip(1e-8)  # (n,1)
        nv = grad_v / gn  # (n, 3)
        # P = I - nn^T applied to H: P_H = H - n (n^T H)
        nTH = np.einsum('ni,nij->nj', nv, hess_v)  # (n, 3)
        P_H = hess_v - nv[:, :, None] * nTH[:, None, :]  # (n,3,3)
        Jn_v = P_H / gn[:, :, None]  # (n, 3, 3)

        # --- dJ_n/dx via FDM on Jn components ---
        # Pad Jn for sentinel lookup
        Jn_padded = np.concatenate([Jn_v, np.zeros((1, 3, 3), dtype=np.float32)])

        def fd_on_field(field_padded, axis):
            """Central FD of per-vertex field along axis.
            field_padded: (n+1, ...), returns (n, ...)
            """
            e = np.zeros(3, dtype=np.int64); e[axis] = 1
            # forward neighbor index
            qf = rel + e[None, :]
            okf = ((qf[:, 0] >= 0) & (qf[:, 0] < lut_dims[0]) &
                   (qf[:, 1] >= 0) & (qf[:, 1] < lut_dims[1]) &
                   (qf[:, 2] >= 0) & (qf[:, 2] < lut_dims[2]))
            idx_f = np.full(n, n, dtype=np.int64)
            idx_f[okf] = lut[qf[okf, 0], qf[okf, 1], qf[okf, 2]]
            # backward neighbor index
            qb = rel - e[None, :]
            okb = ((qb[:, 0] >= 0) & (qb[:, 0] < lut_dims[0]) &
                   (qb[:, 1] >= 0) & (qb[:, 1] < lut_dims[1]) &
                   (qb[:, 2] >= 0) & (qb[:, 2] < lut_dims[2]))
            idx_b = np.full(n, n, dtype=np.int64)
            idx_b[okb] = lut[qb[okb, 0], qb[okb, 1], qb[okb, 2]]

            shape = field_padded.shape[1:]
            out = np.zeros((n,) + shape, dtype=np.float32)
            c = (idx_f < n) & (idx_b < n)
            out[c] = (field_padded[idx_f[c]] - field_padded[idx_b[c]]) / (2*h)
            f1 = (~c) & (idx_f < n)
            out[f1] = (field_padded[idx_f[f1]] - field_padded[np.arange(n)[f1]]) / h
            b1 = (~c) & (~f1) & (idx_b < n)
            out[b1] = (field_padded[np.arange(n)[b1]] - field_padded[idx_b[b1]]) / h
            return out

        # dJn_v[:, :, :, axis] = d J_n / d x_axis
        dJn_v = np.stack([fd_on_field(Jn_padded, ax) for ax in range(3)],
                         axis=-1)  # (n, 3, 3, 3)

        return grad_v, hess_v, Jn_v, dJn_v, lut, ijk_min

    # ----------------------------------------------------------
    # Trilinear interpolation
    # ----------------------------------------------------------

    def _trilinear(self, field, x):
        """Trilinear interpolation of per-vertex field.
        field: (N+1, ...) tensor.  x: (B, 3) tensor.
        Returns: (B, ...) tensor.
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
        x = x.to(self.device)

        gc = (x - self.origin) / self.voxel_size
        base = gc.floor().long()
        frac = gc - base.float()
        rel = base - self._ijk_min
        dims = self._lut_shape
        sent = self._sentinel

        def gv(ox, oy, oz):
            q = rel + torch.tensor([ox, oy, oz], dtype=torch.long,
                                   device=self.device)
            ok = ((q[:, 0] >= 0) & (q[:, 0] < dims[0]) &
                  (q[:, 1] >= 0) & (q[:, 1] < dims[1]) &
                  (q[:, 2] >= 0) & (q[:, 2] < dims[2]))
            qs = torch.stack([q[:, 0].clamp(0, dims[0]-1),
                              q[:, 1].clamp(0, dims[1]-1),
                              q[:, 2].clamp(0, dims[2]-1)], 1)
            idx = self._lut[qs[:, 0], qs[:, 1], qs[:, 2]]
            return torch.where(ok, idx,
                               torch.full_like(idx, sent))

        idx = torch.stack([gv(i, j, k)
                           for i in (0,1) for j in (0,1) for k in (0,1)], 1)
        # order: 000,001,010,011,100,101,110,111
        tx, ty, tz = frac[:, 0], frac[:, 1], frac[:, 2]
        w = torch.stack([
            (1-tx)*(1-ty)*(1-tz), (1-tx)*(1-ty)*tz,
            (1-tx)*ty*(1-tz),     (1-tx)*ty*tz,
            tx*(1-ty)*(1-tz),     tx*(1-ty)*tz,
            tx*ty*(1-tz),         tx*ty*tz], 1)

        vals = field[idx]  # (B, 8, ...)
        if vals.dim() == 2:
            return (vals * w).sum(1)
        es = [x.shape[0], 8] + [1] * (vals.dim() - 2)
        return (vals * w.view(*es)).sum(1)

    # ----------------------------------------------------------
    # Query API
    # ----------------------------------------------------------

    def f(self, x):
        """Signed distance at x (positive outside).  Returns (B, 1)."""
        return self._trilinear(self.vertex_sdf, x).unsqueeze(-1)

    def f_and_grad(self, x):
        """Signed distance (B, 1) and its (unnormalised) gradient (B, 3)."""
        return self.f(x), self._trilinear(self.vertex_grad, x)

    def normal(self, x):
        """Unit normal grad f / |grad f| at x (outward).  Returns (B, 3)."""
        g = self._trilinear(self.vertex_grad, x)
        return g / (g.norm(dim=-1, keepdim=True) + 1e-8)

    def hessian(self, x):
        """Hessian of f at x.  Returns (B, 3, 3)."""
        return self._trilinear(self.vertex_hess, x)

    def normal_jacobian(self, x):
        """Jacobian of the normal at x.  Returns (B, 3, 3).
        result[:, k, i] = d n_k / d x_i
        """
        return self._trilinear(self.vertex_Jn, x)

    def normal_jacobian_deriv(self, x):
        """d J_n / d x at query points.  Returns (B, 3, 3, 3).
        result[:, k, i, j] = d (J_n)_{ki} / d x_j
        """
        return self._trilinear(self.vertex_dJn, x)

    def closest_point(self, x):
        """One-step projection x - f(x) * n(x) onto the surface.  Returns (B, 3)."""
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
        val = self.f(x)
        n = self.normal(x)
        return x - val * n
