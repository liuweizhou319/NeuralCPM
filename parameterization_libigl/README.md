# surfaceframefield_igl

Takes a triangle mesh and a per-face frame field, computes a seamless UV parametrization with libigl's MIQ (Mixed-Integer Quadrangulation), writes the result as an OBJ with UVs, and shows it in polyscope.

Pipeline:

1. `igl::frame_field_deformer` deforms the frame field into a nearly orthogonal one.
2. `igl::frame_to_cross_field` converts it to a cross field.
3. `igl::copyleft::comiso::miq` solves for the integer-constrained parametrization. Every boundary edge of the mesh is passed in as a hard feature. Closed meshes have no boundary and work as well.

## Build

Requires [xmake](https://xmake.io) and MSVC.

```bash
xmake f -m release
xmake
```

The executable is written to `build/windows/x64/release/surfaceframefield_igl.exe`.

Dependency paths are set in `xmake.lua`. libigl is located through the `LIBIGL_ROOT` environment variable. polyscope, glfw, imgui and the other packages are looked up under `%LOCALAPPDATA%\.xmake\packages` at fixed version directories. If the build fails on a new machine, check these two first.

## Usage

From Git Bash:

```bash
./run.sh                                          # use the default inputs in the script
./run.sh <mesh.obj> <framefield.txt>              # choose the inputs
./run.sh <mesh.obj> <framefield.txt> <out.obj>    # also choose the output path
./run.sh --help
```

Example:

```bash
./run.sh tilted_squares_graph_global.obj tilted_squares_graph_global_framefield.txt
```

- A bare file name that does not exist in the current directory is looked up in `input/`.
- The default inputs are the `MESH` and `FIELD` variables at the top of `run.sh`. Edit them as needed.
- Without an output path, the result is written next to the mesh as `<mesh name>_uv.obj`. An existing file with that name is overwritten.
- The MIQ solve is slow. The terminal may print nothing for a long time after `INITIALIZED VECTOR ...`, which is expected. For reference, `tilted_squares_graph_global` (1760 faces) takes about 70 seconds.

From PowerShell, use `run.ps1`. It runs a fixed pair of inputs; edit `$Obj` and `$Cross` at the top of the file to change them.

The executable can also be called directly with the same arguments:

```bash
build/windows/x64/release/surfaceframefield_igl.exe <mesh.obj> <framefield.txt> [out.obj]
```

## Input format

**Mesh**: an OBJ file. Every face must be a triangle.

**Frame field**: a plain text file with one line per face and 6 floating-point numbers per line:

```
x1 y1 z1 x2 y2 z2
```

The first 3 numbers are the face's first frame direction and the last 3 are its second direction. Lines follow the face order of the OBJ. Blank lines and lines starting with `#` are ignored. The number of data lines must equal the number of faces, otherwise the program exits with an error.

## Output

- An OBJ file. `v` holds the original mesh vertices, `vt` holds the UV coordinates, and faces are written as `f v/vt`. The parametrization cuts the mesh along seams, so there are usually more `vt` entries than `v` entries.
- A polyscope window showing the input mesh, both frame directions (red for the first, blue for the second), and the seamless UV as a checkerboard texture. The program exits when the window is closed.

The OBJ is written before the window opens.

## Parameters

The MIQ parameters are hard-coded in the `miq(...)` call in [src/main.cpp](src/main.cpp) (around line 270). There are no command-line options for them.

| Parameter | Current value | Meaning |
|---|---|---|
| gradient size | 60.0 | Global scale of the UV grid. Larger values give a denser grid |
| stiffness | 5.0 | Reserved but unused in libigl. Changing it has no effect |
| direct round | false | Round all integer variables at once (faster, lower quality) |
| stiffening iterations | 5 | Maximum number of full solves. Stops early once a solve has no flipped triangles |
| local iterations | 5 | Local iterations during integer rounding |
| do round | true | Enable integer rounding |
| singularity round | true | Round singularity coordinates to integers |

Run `xmake` again after changing them.

## Common errors

| Error | Cause |
|---|---|
| `File not found: ...` | `run.sh` found the file neither in the current directory nor in `input/` |
| `Executable not found: ...` | The project has not been built. Run `xmake f -m release && xmake` |
| `frame field row count mismatch: expected N, got M` | The frame field line count does not match the mesh face count, usually because the mesh and field files are not a pair |
| `invalid frame field line K, expected 6 floating-point values` | Line K has fewer than 6 numbers or contains non-numeric text |
| `the input OBJ must be a triangular mesh` | The mesh contains non-triangular faces |
