import trimesh
import kaolin
from torchsdf import index_vertices_by_faces, compute_sdf
import os
import torch
from time import time

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = "cuda"

os.makedirs("tests/outputs", exist_ok=True)

# Ns
num_sample = 1000000
samples = torch.rand((num_sample, 3)).to(device).detach()
samples = samples * 2 - 1

all_pass = True


def write_points(finename, points, color):
    point_count = points.shape[0]
    ply_file = open(finename, 'w')
    ply_file.write("ply\n")
    ply_file.write("format ascii 1.0\n")
    ply_file.write("element vertex " + str(point_count) + "\n")
    ply_file.write("property float x\n")
    ply_file.write("property float y\n")
    ply_file.write("property float z\n")
    ply_file.write("property uchar red\n")
    ply_file.write("property uchar green\n")
    ply_file.write("property uchar blue\n")
    ply_file.write("end_header\n")

    for i in range(point_count):
        ply_file.write(str(points[i, 0]) + " " +
                       str(points[i, 1]) + " " +
                       str(points[i, 2]))
        ply_file.write(" "+str(int(color[i, 0])) + " " +
                       str(int(color[i, 1])) + " " +
                       str(int(color[i, 2])) + " ")
        ply_file.write("\n")
    ply_file.close()


print("====Sign check====")
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
    # (1, Ns)
    dif = signs_ts != signs
    dif = dif.reshape(-1)
    miss_points = x[dif, :]
    color = torch.zeros_like(miss_points).int()
    color[:, 0] = 255
    write_points(os.path.join("tests/outputs",
                 model[:-4]+".ply"), points=miss_points.detach().cpu().numpy(), color=color)
