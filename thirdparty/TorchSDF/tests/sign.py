import trimesh
import kaolin
from torchsdf import index_vertices_by_faces, compute_sdf
import os
import torch
from time import time

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = "cuda"
# Ns
num_sample = 1000000
samples = torch.rand((num_sample, 3)).to(device).detach()
samples = samples * 2 - 1

all_pass = True

print("====Sign test====")
for model in os.listdir("tests/models"):
    print("Test:", model[:-4], end=" ")
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
    # (Nf, 3, 3)
    face_verts_ts = index_vertices_by_faces(verts, faces)

    # Kaolin
    # (1, Ns)
    signs = kaolin.ops.mesh.check_sign(
        verts.unsqueeze(0), faces, x.unsqueeze(0))
    signs = torch.where(signs, -1*torch.ones_like(signs, dtype=torch.int32),
                        torch.ones_like(signs, dtype=torch.int32))

    # TorchSDF
    # (Ns)
    distances_ts, signs_ts, normals_ts, clst_points_ts = compute_sdf(
        x, face_verts_ts)
    equal_num = (signs_ts == signs).sum().item()
    equal_ratio = equal_num/num_sample
    sign_fit = (equal_ratio > 0.98)
    if (sign_fit):
        print("\x1B[32mPass\x1B[0m")
    else:
        all_pass = False
        print("\x1B[31mSign wrong!\x1B[0m")
    print(f"Ratio: {equal_ratio:.3f} ({equal_num:d}/{num_sample:d})")

if (all_pass):
    print("====\x1B[32mAll pass\x1B[0m====")
else:
    print("====\x1B[31mWrong\x1B[0m====")
