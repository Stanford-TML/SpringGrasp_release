import trimesh
import kaolin
from torchsdf import index_vertices_by_faces, compute_sdf
import os
import torch
import numpy as np
import skimage

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = "cuda"

os.makedirs("tests/outputs", exist_ok=True)

# Ns = N * N * N
N = 256

voxel_origin = [-1, -1, -1]
voxel_size = 2.0 / (N - 1)
overall_index = torch.arange(0, N**3, 1, dtype=torch.long)
samples = torch.zeros((N**3, 3))

samples[:, 2] = overall_index % N
samples[:, 1] = (overall_index / N) % N
samples[:, 0] = ((overall_index / N) / N) % N

samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[0]
samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[2]

# (Ns, 3)
samples = samples.to(device)
values = None
# api = "Kaolin"
api = "TorchSDF"

print("====Marching cube test====")
for model in os.listdir("tests/models"):
    print("Test:", model[:-4])
    model_path = os.path.join("tests/models", model)
    mesh = trimesh.load(model_path, force="mesh", process=False)
    # (Ns, 3)
    x = samples.clone().requires_grad_()
    # (Nv, 3)
    verts = torch.Tensor(mesh.vertices.copy()).to(device)
    # (Nf, 3)
    faces = torch.Tensor(mesh.faces.copy()).long().to(device)
    # (1, Nf, 3, 3)
    face_verts = kaolin.ops.mesh.index_vertices_by_faces(
        verts.unsqueeze(0), faces)
    face_verts_ts = index_vertices_by_faces(verts, faces)

    distances, signs = None, None
    # Kaolin
    if api == "Kaolin":
        # (1, Ns)
        distances, face_indexes, types = kaolin.metrics.trianglemesh.point_to_mesh_distance(
            x.unsqueeze(0), face_verts)
        # (1, Ns)
        signs_ = kaolin.ops.mesh.check_sign(
            verts.unsqueeze(0), faces, x.unsqueeze(0))
        # (1, Ns)
        signs = torch.where(signs_, -torch.ones_like(
            signs_).int(), torch.ones_like(signs_).int())
    # TorchSDF
    elif api == "TorchSDF":
        distances, signs, normals_ts, clst_points_ts = compute_sdf(
            x, face_verts_ts)

    values = distances.sqrt() * signs
    values = values.detach().cpu().numpy().reshape(N, N, N)
    verts, faces, _, _ = skimage.measure.marching_cubes(
        values, level=0.0, spacing=[voxel_size] * 3)
    verts[:, 0] += voxel_origin[0]
    verts[:, 1] += voxel_origin[1]
    verts[:, 2] += voxel_origin[2]

    trimesh.Trimesh(vertices=verts, faces=faces,
                    process=False).export(os.path.join("tests/outputs", model))

print("====Done====")
