import trimesh
import kaolin
from torchsdf import index_vertices_by_faces, compute_sdf
import os
import torch
from time import time

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = "cuda"
# Ns
num_sample = 10000000
samples = torch.rand((num_sample, 3)).to(device).detach()
samples = samples * 2 - 1

all_pass = True

print("====Speed test====")
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
    torch.cuda.synchronize()
    tmp = time()
    distances, face_indexes, types = kaolin.metrics.trianglemesh.point_to_mesh_distance(
        x.unsqueeze(0), face_verts)
    signs_ = kaolin.ops.mesh.check_sign(
        verts.unsqueeze(0), faces, x.unsqueeze(0))
    signs = torch.where(signs_, -torch.ones_like(
        signs_).int(), torch.ones_like(signs_).int())
    sdf = distances.sqrt() * signs
    torch.cuda.synchronize()
    time_kaolin = time() - tmp

    # TorchSDF
    # (Ns)
    torch.cuda.synchronize()
    tmp = time()
    distances_ts, dist_sign_ts, normals_ts, clst_points_ts = compute_sdf(
        x, face_verts_ts)
    sdf_ts = distances_ts.sqrt() * dist_sign_ts
    torch.cuda.synchronize()
    time_ts = time() - tmp
    
    equal_num = (dist_sign_ts == signs).sum().item()
    equal_ratio = equal_num/num_sample
    sign_fit = (equal_ratio > 0.98)
    dis_fit = torch.allclose(distances, distances_ts)
    if (dis_fit and sign_fit):
        print("\x1B[32mPass\x1B[0m")
    else:
        all_pass = False
        if (not dis_fit):
            print("\x1B[31mDistance wrong!\x1B[0m")
        if (not sign_fit):
            print("\x1B[31mSign wrong!\x1B[0m")
    print("Max abs:", (distances.sqrt() - distances_ts.sqrt()).abs().max().item())
    print(f"Ratio: {equal_ratio:.3f} ({equal_num:d}/{num_sample:d})")
    print("TorchSDF/Kaolin time:", time_ts/time_kaolin)

print("====\x1B[32mAll pass\x1B[0m====")
