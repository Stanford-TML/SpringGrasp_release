// Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License")
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

#include <math.h>

#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <THC/THCAtomics.cuh>

#include "utils.h"

#define PRIVATE_CASE_TYPE_AND_VAL(ENUM_TYPE, TYPE, TYPE_NAME, VAL, ...) \
  case ENUM_TYPE: { \
    using TYPE_NAME = TYPE; \
    const int num_threads = VAL; \
    return __VA_ARGS__(); \
  }


#define DISPATCH_INPUT_TYPES(TYPE, TYPE_NAME, SCOPE_NAME, ...) \
  [&] { \
    switch(TYPE) \
    { \
      PRIVATE_CASE_TYPE_AND_VAL(at::ScalarType::Float, float, TYPE_NAME, 1024, __VA_ARGS__) \
      PRIVATE_CASE_TYPE_AND_VAL(at::ScalarType::Double, double, TYPE_NAME, 512, __VA_ARGS__) \
      default: \
        AT_ERROR(#SCOPE_NAME, " not implemented for '", toString(TYPE), "'"); \
    } \
  }()

namespace kaolin {

template<typename T>
struct ScalarTypeToVec3 { using type = void; };
template <> struct ScalarTypeToVec3<float> { using type = float3; };
template <> struct ScalarTypeToVec3<double> { using type = double3; };

template<typename T>
struct Vec3TypeToScalar { using type = void; };
template <> struct Vec3TypeToScalar<float3> { using type = float; };
template <> struct Vec3TypeToScalar<double3> { using type = double; };

__device__ __forceinline__ float3 make_vector(float x, float y, float z) {
  return make_float3(x, y, z);
}

__device__ __forceinline__ double3 make_vector(double x, double y, double z) {
  return make_double3(x, y, z);
}

template <typename vector_t>
__device__ __forceinline__ typename Vec3TypeToScalar<vector_t>::type dot(vector_t a, vector_t b) {
  return a.x * b.x + a.y * b.y + a.z * b.z ;
}

template<typename scalar_t, typename vector_t>
__device__ __forceinline__ scalar_t dot2(vector_t v) {
  return dot<scalar_t, vector_t>(v, v);
}

template<typename scalar_t>
__device__ __forceinline__ scalar_t clamp(scalar_t x, scalar_t a, scalar_t b) {
  return max(a, min(b, x));
}

template<typename vector_t>
__device__ __forceinline__ vector_t cross(vector_t a, vector_t b) {
  return make_vector(a.y * b.z - a.z * b.y,
                     a.z * b.x - a.x * b.z,
                     a.x * b.y - a.y * b.x);
}

template<typename scalar_t>
__device__ __forceinline__ int sign(scalar_t a) {
  if (a <= 0) {return -1;}
  else {return 1;}
}

template<typename scalar_t, typename vector_t>
__device__ __forceinline__ vector_t operator* (vector_t a, scalar_t b) {
  return make_vector(a.x * b, a.y * b, a.z * b);
}

template<typename vector_t>
__device__ __forceinline__ vector_t operator* (vector_t a, vector_t b) {
  return make_vector(a.x * b.x, a.y * b.y, a.z * b.z);
}

template<typename scalar_t, typename vector_t>
__device__ __forceinline__ vector_t operator+ (vector_t a, scalar_t b) {
  return make_vector(a.x + b, a.y + b, a.z + b);
}

template<typename vector_t>
__device__ __forceinline__ vector_t operator+ (vector_t a, vector_t b) {
  return make_vector(a.x + b.x, a.y + b.y, a.z + b.z);
}

template<typename scalar_t, typename vector_t>
__device__ __forceinline__ vector_t operator- (vector_t a, scalar_t b) {
  return make_vector(a.x - b, a.y - b, a.z - b);
}

template<typename vector_t>
__device__ __forceinline__ vector_t operator- (vector_t a, vector_t b) {
  return make_vector(a.x - b.x, a.y - b.y, a.z - b.z);
}

template<typename scalar_t, typename vector_t>
__device__ __forceinline__ vector_t operator/ (vector_t a, scalar_t b) {
  return make_vector(a.x / b, a.y / b, a.z / b);
}

template<typename vector_t>
__device__ __forceinline__ vector_t operator/ (vector_t a, vector_t b) {
  return make_vector(a.x / b.x, a.y / b.y, a.z / b.z);
}

template<typename vector_t>
__device__ __forceinline__ typename Vec3TypeToScalar<vector_t>::type project_edge(
    vector_t vertex, vector_t edge, vector_t point) {
  typedef typename Vec3TypeToScalar<vector_t>::type scalar_t;
  vector_t point_vec = point - vertex;
  scalar_t length = dot(edge, edge);
  return dot(point_vec, edge) / length;
}

template<typename vector_t>
__device__ __forceinline__ vector_t project_plane(vector_t vertex, vector_t normal, vector_t point) {
  typedef typename Vec3TypeToScalar<vector_t>::type scalar_t;
  scalar_t inv_len = rsqrt(dot(normal, normal));
  vector_t unit_normal = normal * inv_len;
  scalar_t dist = (point.x - vertex.x) * unit_normal.x + \
                  (point.y - vertex.y) * unit_normal.y + \
                  (point.z - vertex.z) * unit_normal.z;
  return point - (unit_normal * dist);
}

template<typename scalar_t>
__device__ __forceinline__ bool in_range(scalar_t a) {
  return (a <= 1 && a >= 0);
}

template<typename vector_t>
__device__ __forceinline__ bool is_above(vector_t vertex, vector_t edge, vector_t normal, vector_t point) {
  vector_t edge_normal = cross(normal, edge);
  return dot(edge_normal, point - vertex) > 0;
}

template<typename vector_t>
__device__ __forceinline__ bool is_not_above(vector_t vertex, vector_t edge, vector_t normal,
                                     vector_t point) {
  vector_t edge_normal = cross(normal, edge);
  return dot(edge_normal, point - vertex) <= 0;
}


template<typename vector_t>
__device__ __forceinline__ vector_t point_at(vector_t vertex, vector_t edge, float t) {
  return vertex + (edge * t);
}


template<typename scalar_t, typename vector_t, int BLOCK_SIZE>
__global__ void unbatched_triangle_distance_forward_cuda_kernel(
    const vector_t* points,
    const vector_t* vertices,
    int num_points,
    int num_faces,
    scalar_t* out_dist,
    int* out_dist_sign,
    vector_t* out_normals,
    vector_t* clst_points) {
  __shared__ vector_t shm[BLOCK_SIZE * 3];

  for (int start_face_idx = 0; start_face_idx < num_faces; start_face_idx += BLOCK_SIZE) {
    int num_faces_iter = min(num_faces - start_face_idx, BLOCK_SIZE);
    for (int j = threadIdx.x; j < num_faces_iter * 3; j += blockDim.x) {
      shm[j] = vertices[start_face_idx * 3 + j];
    }
    __syncthreads();
    for (int point_idx = threadIdx.x + blockDim.x * blockIdx.x; point_idx < num_points;
         point_idx += blockDim.x * gridDim.x) {
      vector_t p = points[point_idx];
      scalar_t best_dist = INFINITY;
      int best_dist_sign = 0;
      vector_t best_normal;
      vector_t best_clst_point;
      for (int sub_face_idx = 0; sub_face_idx < num_faces_iter; sub_face_idx++) {
        vector_t closest_point;

        vector_t v1 = shm[sub_face_idx * 3];
        vector_t v2 = shm[sub_face_idx * 3 + 1];
        vector_t v3 = shm[sub_face_idx * 3 + 2];

        vector_t e12 = v2 - v1;
        vector_t e23 = v3 - v2;
        vector_t e31 = v1 - v3;
        vector_t normal = cross(v1 - v2, e31);
        scalar_t uab = project_edge(v1, e12, p);
        scalar_t uca = project_edge(v3, e31, p);
        if (uca > 1 && uab < 0) {
          closest_point = v1;
        } else {
          scalar_t ubc = project_edge(v2, e23, p);
          if (uab > 1 && ubc < 0) {
            closest_point = v2;
          } else if (ubc > 1 && uca < 0) {
            closest_point = v3;
          } else {
            if (in_range(uab) && (is_not_above(v1, e12, normal, p))) {
              closest_point = point_at(v1, e12, uab);
            } else if (in_range(ubc) && (is_not_above(v2, e23, normal, p))) {
              closest_point = point_at(v2, e23, ubc);
            } else if (in_range(uca) && (is_not_above(v3, e31, normal, p))) {
              closest_point = point_at(v3, e31, uca);
            } else {
              closest_point = project_plane(v1, normal, p);
            }
          }
        }
        vector_t dist_vec = p - closest_point;
        vector_t grad_normal = dist_vec * rsqrt(1e-16f + dot(dist_vec, dist_vec));
        int dist_sign = (dot(dist_vec, normal)>=0)? 1 : -1;
        float dist = dot(dist_vec, dist_vec);
        if (sub_face_idx == 0 || best_dist > dist) {
          best_dist = dist;
          best_dist_sign = dist_sign;
          best_normal = grad_normal;
          best_clst_point = closest_point;
        }
      }
      if (start_face_idx == 0 || out_dist[point_idx] > best_dist) {
        out_dist[point_idx] = best_dist;
        out_dist_sign[point_idx] = best_dist_sign;
        out_normals[point_idx] = best_normal;
        clst_points[point_idx] = best_clst_point;
      }
    }
    __syncthreads();
  }
}

template<typename scalar_t, typename vector_t>
__global__ void unbatched_triangle_distance_backward_cuda_kernel(
    const scalar_t* grad_dist,
    const vector_t* points,
    const vector_t* clst_points,
    int num_points,
    vector_t* grad_points) {
  for (int point_id = threadIdx.x + blockIdx.x * blockDim.x; point_id < num_points;
       point_id += blockDim.x * gridDim.x) {
    scalar_t grad_out = 2. * grad_dist[point_id];
    vector_t dist_vec = points[point_id] - clst_points[point_id];
    dist_vec = dist_vec * grad_out;
    grad_points[point_id] = dist_vec;
  }
}

void unbatched_triangle_distance_forward_cuda_impl(
    at::Tensor points,
    at::Tensor face_vertices,
    at::Tensor dist,
    at::Tensor dist_sign,
    at::Tensor normals,
    at::Tensor clst_points) {
  const int num_threads = 512;
  const int num_points = points.size(0);
  const int num_blocks = (num_points + num_threads - 1) / num_threads;
  AT_DISPATCH_FLOATING_TYPES(points.scalar_type(),
                             "unbatched_triangle_distance_forward_cuda", [&] {
    using vector_t = ScalarTypeToVec3<scalar_t>::type;
    const at::cuda::OptionalCUDAGuard device_guard(at::device_of(points));
    auto stream = at::cuda::getCurrentCUDAStream();
    unbatched_triangle_distance_forward_cuda_kernel<scalar_t, vector_t, 512><<<
      num_blocks, num_threads, 0, stream>>>(
        reinterpret_cast<vector_t*>(points.data_ptr<scalar_t>()),
        reinterpret_cast<vector_t*>(face_vertices.data_ptr<scalar_t>()),
        points.size(0),
        face_vertices.size(0),
        dist.data_ptr<scalar_t>(),
        dist_sign.data_ptr<int32_t>(),
        reinterpret_cast<vector_t*>(normals.data_ptr<scalar_t>()),
        reinterpret_cast<vector_t*>(clst_points.data_ptr<scalar_t>()));
    //CUDA_CHECK(cudaGetLastError());
  });
}

void unbatched_triangle_distance_backward_cuda_impl(
    at::Tensor grad_dist,
    at::Tensor points,
    at::Tensor clst_points,
    at::Tensor grad_points) {

  DISPATCH_INPUT_TYPES(points.scalar_type(), scalar_t,
                       "unbatched_triangle_distance_backward_cuda", [&] {
    const int num_points = points.size(0);
    const int num_blocks = (num_points + num_threads - 1) / num_threads;
    using vector_t = ScalarTypeToVec3<scalar_t>::type;
    const at::cuda::OptionalCUDAGuard device_guard(at::device_of(points));
    auto stream = at::cuda::getCurrentCUDAStream();
    unbatched_triangle_distance_backward_cuda_kernel<scalar_t, vector_t><<<
      num_blocks, num_threads, 0, stream>>>(
        grad_dist.data_ptr<scalar_t>(),
        reinterpret_cast<vector_t*>(points.data_ptr<scalar_t>()),
        reinterpret_cast<vector_t*>(clst_points.data_ptr<scalar_t>()),
        points.size(0),
        reinterpret_cast<vector_t*>(grad_points.data_ptr<scalar_t>()));
    //CUDA_CHECK(cudaGetLastError());
  });
}

}  // namespace kaolin

#undef PRIVATE_CASE_TYPE_AND_VAL
#undef DISPATCH_INPUT_TYPES