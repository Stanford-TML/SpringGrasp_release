import torch
from torchsdf import _C


def index_vertices_by_faces(vertices_features, faces):
    r"""Index vertex features to convert per vertex tensor to per vertex per face tensor.

    Args:
        vertices_features (torch.FloatTensor):
            vertices features, of shape
            :math:`(\text{batch_size}, \text{num_points}, \text{knum})`,
            ``knum`` is feature dimension, the features could be xyz position,
            rgb color, or even neural network features.
        faces (torch.LongTensor):
            face index, of shape :math:`(\text{num_faces}, \text{num_vertices})`.
    Returns:
        (torch.FloatTensor):
            the face features, of shape
            :math:`(\text{batch_size}, \text{num_faces}, \text{num_vertices}, \text{knum})`.
    """
    assert vertices_features.ndim == 2, \
        "vertices_features must have 2 dimensions of shape (batch_sizenum_points, knum)"
    assert faces.ndim == 2, "faces must have 2 dimensions of shape (num_faces, num_vertices)"
    # input = vertices_features.unsqueeze(2).expand(-1, -1, faces.shape[-1], -1)
    # indices = faces[None, ..., None].expand(
    #     vertices_features.shape[0], -1, -1, vertices_features.shape[-1])
    # return torch.gather(input=input, index=indices, dim=1)
    input = vertices_features.reshape(-1, 1, 3).expand(-1, faces.shape[-1], -1)
    indices = faces[..., None].expand(
        -1, -1, vertices_features.shape[-1])
    return torch.gather(input=input, index=indices, dim=0)


def compute_sdf(pointclouds, face_vertices):
    return _UnbatchedTriangleDistanceCuda.apply(pointclouds, face_vertices)


class _UnbatchedTriangleDistanceCuda(torch.autograd.Function):
    @staticmethod
    def forward(ctx, points, face_vertices):
        num_points = points.shape[0]
        min_dist = torch.zeros(
            (num_points), device=points.device, dtype=points.dtype)
        dist_sign = torch.zeros(
            (num_points), device=points.device, dtype=torch.int32)
        normals = torch.zeros(
            (num_points, 3), device=points.device, dtype=points.dtype)
        clst_points = torch.zeros(
            (num_points, 3), device=points.device, dtype=points.dtype)
        _C.unbatched_triangle_distance_forward_cuda(
            points, face_vertices, min_dist, dist_sign, normals, clst_points)
        ctx.save_for_backward(points.contiguous(), clst_points)
        ctx.mark_non_differentiable(dist_sign, normals, clst_points)
        return min_dist, dist_sign, normals, clst_points

    @staticmethod
    def backward(ctx, grad_dist, grad_dist_sign, grad_normals, grad_clst_points):
        points, clst_points = ctx.saved_tensors
        grad_dist = grad_dist.contiguous()
        grad_points = torch.zeros_like(points)
        grad_face_vertices = None
        _C.unbatched_triangle_distance_backward_cuda(
            grad_dist, points, clst_points, grad_points)
        return grad_points, grad_face_vertices
