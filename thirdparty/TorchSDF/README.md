# TorchSDF

This is a custom version of the signed distance field(SDF) computation from the [Kaolin library](https://github.com/NVIDIAGameWorks/kaolin). It supports SDF computation for **manifold** meshes with PyTorch on GPU.

## Purpose

Why don't I use the original Kaolin API?

- I just want to compute SDF. Kaolin is too large and redundant. I want a lighter package.
- With Kaolin, I should use `kaolin.metrics.trianglemesh.point_to_mesh_distance` and `kaolin.ops.mesh.check_sign` for SDF computation. But in a simple but not so precise definition, I can get the value in a single cuda kernel function. This is a potential acceleration.
- I can learn knowledge about the cpp interface of PyTorch.

## Installation

Require PyTorch installed.

```bash
bash install.sh
```

## Usage

The code provides two functions:

- `compute_sdf(pointclouds, face_vertices)`
    - input
        - unbatched points with shape (num_point , 3)
        - unbatched face_vertices with shape (num_face , 3, 3)
    - returns 
        - squared distance
        - normals defined by gradient
        - distance signs (inside -1 and outside 1)
        - closest points
- `index_vertices_by_faces(vertices_features, faces)`: return face_verts reqired by `compute_sdf(pointclouds, face_vertices)`.

## Note

- The sign is defined by `sign((p - closest_point).dot(face_normal))`, **check your mesh has perfect normal information**.
  - **This definition sometimes causes wrong results.** For example, there is an acute angle between two faces.
  - So it is not so precise so far.
- Returned normal is defined by `(p - closest_point).normalized()` or equally $\frac{\partial d}{\partial p}$, not face normal.
- The code only runs on cuda.
- Scripts in `tests` cannot run independently (require Kaolin API).
