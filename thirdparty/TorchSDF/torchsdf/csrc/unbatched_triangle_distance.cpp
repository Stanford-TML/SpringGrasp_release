// Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <ATen/ATen.h>

#include "check.h"

namespace kaolin {

#ifdef WITH_CUDA

void unbatched_triangle_distance_forward_cuda_impl(
    at::Tensor points,
    at::Tensor face_vertices,
    at::Tensor dist,
    at::Tensor dist_sign,
    at::Tensor normals,
    at::Tensor clst_points);

void unbatched_triangle_distance_backward_cuda_impl(
    at::Tensor grad_dist,
    at::Tensor points,
    at::Tensor clst_points,
    at::Tensor grad_points);

#endif  // WITH_CUDA


void unbatched_triangle_distance_forward_cuda(
    at::Tensor points,
    at::Tensor face_vertices,
    at::Tensor dist,
    at::Tensor dist_sign,
    at::Tensor normals,
    at::Tensor clst_points) {
  CHECK_CUDA(points);
  CHECK_CUDA(face_vertices);
  CHECK_CUDA(dist);
  CHECK_CUDA(dist_sign);
  CHECK_CUDA(normals);
  CHECK_CUDA(clst_points);
  CHECK_CONTIGUOUS(points);
  CHECK_CONTIGUOUS(face_vertices);
  CHECK_CONTIGUOUS(dist);
  CHECK_CONTIGUOUS(dist_sign);
  CHECK_CONTIGUOUS(normals);
  CHECK_CONTIGUOUS(clst_points);
  const int num_points = points.size(0);
  const int num_faces = face_vertices.size(0);
  CHECK_SIZES(points, num_points, 3);
  CHECK_SIZES(face_vertices, num_faces, 3, 3);
  CHECK_SIZES(dist, num_points);
  CHECK_SIZES(dist_sign, num_points);
  CHECK_SIZES(normals, num_points, 3);
  CHECK_SIZES(clst_points, num_points, 3);
#if WITH_CUDA
  unbatched_triangle_distance_forward_cuda_impl(
      points, face_vertices, dist, dist_sign, normals, clst_points);
#else
  AT_ERROR("unbatched_triangle_distance not built with CUDA");
#endif
}

void unbatched_triangle_distance_backward_cuda(
    at::Tensor grad_dist,
    at::Tensor points,
    at::Tensor clst_points,
    at::Tensor grad_points) {
  CHECK_CUDA(grad_dist);
  CHECK_CUDA(points);
  CHECK_CUDA(clst_points);
  CHECK_CUDA(grad_points);
  CHECK_CONTIGUOUS(grad_dist);
  CHECK_CONTIGUOUS(points);
  CHECK_CONTIGUOUS(clst_points);
  CHECK_CONTIGUOUS(grad_points);

  const int num_points = points.size(0);
  CHECK_SIZES(grad_dist, num_points);
  CHECK_SIZES(points, num_points, 3);
  CHECK_SIZES(clst_points, num_points, 3);
  CHECK_SIZES(grad_points, num_points, 3);

#if WITH_CUDA
  unbatched_triangle_distance_backward_cuda_impl(
      grad_dist, points, clst_points, grad_points);
#else
  AT_ERROR("unbatched_triangle_distance_backward not built with CUDA");
#endif
}

}  // namespace kaolin
