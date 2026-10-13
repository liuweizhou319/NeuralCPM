"""
Two analytic families of surfaces parameterised by a time t: a wavy torus and
a radial icosphere.  train_basis_time.py uses them as shape families; run this
file to view a few values of t with polyscope.

Usage:
    python surfaces.py --shape torus     --t 0 0.5 1.0 1.5
    python surfaces.py --shape icosphere --t 0 0.5 1.0 1.5
"""

import argparse
import numpy as np


# ---------- wavy torus ----------
def wavy_torus(t, nu=128, nv=64, R=1.6, r0=0.35, eps=0.25, k=3, omega=1.0):
    u = np.linspace(0, 2 * np.pi, nu, endpoint=False)
    v = np.linspace(0, 2 * np.pi, nv, endpoint=False)
    U, V = np.meshgrid(u, v, indexing="ij")

    r = r0 * (1 + eps * np.sin(k * U + omega * t))
    X = (R + r * np.cos(V)) * np.cos(U)
    Y = (R + r * np.cos(V)) * np.sin(U)
    Z = r * np.sin(V)
    verts = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)

    # triangles with periodic wrap-around
    I, J = np.meshgrid(np.arange(nu), np.arange(nv), indexing="ij")
    a = I * nv + J
    b = ((I + 1) % nu) * nv + J
    c = ((I + 1) % nu) * nv + (J + 1) % nv
    d = I * nv + (J + 1) % nv
    faces = np.concatenate(
        [np.stack([a, b, c], axis=-1).reshape(-1, 3),
         np.stack([a, c, d], axis=-1).reshape(-1, 3)],
        axis=0,
    )
    return verts, faces


# ---------- icosphere ----------
def _icosahedron():
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [-1,  phi, 0], [ 1,  phi, 0], [-1, -phi, 0], [ 1, -phi, 0],
        [ 0, -1,  phi], [ 0,  1,  phi], [ 0, -1, -phi], [ 0,  1, -phi],
        [ phi, 0, -1], [ phi, 0,  1], [-phi, 0, -1], [-phi, 0,  1],
    ], dtype=float)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    faces = np.array([
        [0,11,5],[0,5,1],[0,1,7],[0,7,10],[0,10,11],
        [1,5,9],[5,11,4],[11,10,2],[10,7,6],[7,1,8],
        [3,9,4],[3,4,2],[3,2,6],[3,6,8],[3,8,9],
        [4,9,5],[2,4,11],[6,2,10],[8,6,7],[9,8,1],
    ])
    return verts, faces


def icosphere(subdiv=4):
    verts, faces = _icosahedron()
    verts = list(verts)
    for _ in range(subdiv):
        cache, new_faces = {}, []

        def mid(i, j):
            key = (i, j) if i < j else (j, i)
            if key in cache:
                return cache[key]
            m = verts[i] + verts[j]
            m /= np.linalg.norm(m)
            verts.append(m)
            cache[key] = len(verts) - 1
            return cache[key]

        for a, b, c in faces:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            new_faces += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        faces = np.array(new_faces)
    return np.array(verts), faces


def radial_icosphere(t, subdiv=4):
    omega, faces = icosphere(subdiv)
    x, y, z = omega[:, 0], omega[:, 1], omega[:, 2]
    rho = (1
           + 0.15 * np.sin(3 * x + 2 * t)
           + 0.10 * np.cos(4 * y - t)
           + 0.08 * np.sin(5 * z + t))
    verts = rho[:, None] * omega
    return verts, faces


# ---------- main ----------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shape", choices=["torus", "icosphere"], required=True)
    p.add_argument("--t", type=float, nargs="+", required=True,
                   help="list of time values, e.g. --t 0 0.5 1.0 1.5")
    p.add_argument("--spacing", type=float, default=4.0,
                   help="spacing along the x axis between the shapes")
    args = p.parse_args()

    # imported here so that train_basis_time.py can use the shape functions
    # on a machine without polyscope
    import polyscope as ps

    ps.init()
    for k, t in enumerate(args.t):
        if args.shape == "torus":
            v, f = wavy_torus(t)
        else:
            v, f = radial_icosphere(t)
        v = v + np.array([k * args.spacing, 0.0, 0.0])
        ps.register_surface_mesh(f"{args.shape}_t={t:.3f}", v, f)
    ps.show()


if __name__ == "__main__":
    main()
