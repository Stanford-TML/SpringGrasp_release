import trimesh
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

print("====Normal test====")
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
    # (Nf, 3, 3)
    face_verts = index_vertices_by_faces(verts, faces)

    # TorchSDF
    # (Ns)
    distances, dist_sign, normals, clst_points = compute_sdf(x, face_verts)
    gradient = torch.autograd.grad([distances.sum()], [x], create_graph=True,
                                   retain_graph=True)[0]

    normal_direct = normals * 2 * distances.unsqueeze(1).sqrt()
    normal_from_grad = torch.autograd.grad([distances.sum()], [x], create_graph=True,
                                           retain_graph=True)[0]
    normal_fit = torch.allclose(normal_direct, normal_from_grad, atol=5e-7)
    if normal_fit:
        print("\x1B[32mPass\x1B[0m")
    else:
        all_pass = False
        print("\x1B[31mNormal wrong!\x1B[0m")
    print("Max abs:", (normal_direct - normal_from_grad).abs().max().item())

if (all_pass):
    print("====\x1B[32mAll pass\x1B[0m====")
else:
    print("====\x1B[31mWrong\x1B[0m====")
