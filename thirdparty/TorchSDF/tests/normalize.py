import trimesh
import os
import numpy as np

for model in os.listdir("tests/models"):
    model_path = os.path.join("tests/models", model)
    mesh = trimesh.load(model_path, force="mesh", process=False)
    # print(mesh.is_watertight)
    # verts = np.array(mesh.vertices)
    # xcenter = (np.max(verts[:, 0]) + np.min(verts[:, 0])) / 2
    # ycenter = (np.max(verts[:, 1]) + np.min(verts[:, 1])) / 2
    # zcenter = (np.max(verts[:, 2]) + np.min(verts[:, 2])) / 2
    # verts_ = verts - np.array([xcenter, ycenter, zcenter])
    # dmax = np.max(np.sqrt(np.sum(np.square(verts_), axis=1))) * 1.03
    # verts_ /= dmax
    # mesh_ = trimesh.Trimesh(vertices=verts_, faces=mesh.faces, process=False)
    # mesh_.export(model_path)
    print(mesh.is_watertight)
