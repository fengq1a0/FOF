// Exact multi-layer rasterization of triangle meshes and Fourier Occupancy
// Field (FOF) integration.
//
// Conventions (the Python side, lib/fof/core.py, prepares the inputs):
//   * Vertices arrive already projected to pixel space as fixed-point integers
//     with SUBPIXEL_BITS fractional bits, plus a float depth z.  Pixel (i, j)
//     (column i, row j, rows growing downwards) has its centre at (i, j) in
//     pixel units, i.e. (i << SUBPIXEL_BITS, j << SUBPIXEL_BITS) in fixed point.
//   * Coverage is decided with exact int64 edge functions and a top-left tie
//     rule (equivalent to a symbolic perturbation of the sample point).  For a
//     closed, consistently connected mesh every pixel centre is claimed by
//     exactly one triangle of each layer: no cracks and no double hits, also
//     when pixel centres fall exactly on shared edges or vertices.
//   * Each covered (pixel, triangle) pair becomes a fragment.  Its 64-bit key
//     packs [order-preserving depth bits : 32][exit flag : 1][face index : 31]
//     so that sorting the keys of a pixel orders its fragments by depth
//     (ascending z, i.e. from the far plane z=-1 towards the camera at z=+1),
//     with entries before exits on exact depth ties and the face index as the
//     final deterministic tie-break.
//   * A fragment is an "exit" when the triangle faces the camera (+z normal for
//     counter-clockwise faces): a ray travelling towards +z leaves the solid
//     there.  Back-facing triangles are "entries".

// Wrap our copy of CUB in its own namespace so it cannot collide (ODR) with
// the CUB instantiations inside libtorch_cuda.
#ifndef CUB_WRAPPED_NAMESPACE
#define CUB_WRAPPED_NAMESPACE fof_cub
#endif

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <cub/device/device_segmented_sort.cuh>

#include <cstdint>
#include <limits>
#include <vector>

namespace fof {
namespace {

constexpr int kSubpixelBits = 8;
constexpr int kThreads = 256;
constexpr int64_t kMaxBlocks = 1 << 16;

// Fill rules turning a sorted fragment list into occupied depth intervals.
enum FillRule : int {
  // Original FOF rule: an interval starts at the first entry seen while
  // outside and ends at the last exit before the next entry.  Leading exits
  // and a trailing unmatched entry run are ignored.  Tolerant to duplicated
  // surfaces, missing hits and small holes; needs consistent orientation.
  kRobust = 0,
  // Even-odd rule, independent of face orientation.
  kParity = 1,
  // Nonzero rule: inside where the winding number (entries - exits) is not 0.
  kWinding = 2,
  // Winding number clamped at 0 (exact union of nested / interpenetrating parts), an exit run
  // after a closed interval extends it (duplicated layers, as in kRobust), an interval still
  // open at the end of the ray closes at its last exit, and if nothing could be matched the
  // even-odd rule is used (flipped orientation).
  kHybrid = 3,
};

struct Tri {
  int x[3], y[3];      // fixed-point pixel coordinates, counter-clockwise
  float z[3];          // depth of each vertex
  long long area;      // twice the signed area, > 0 after canonicalisation
  int i0, j0, bw;      // clipped pixel bounding box origin and width
  int exit;            // 1 if the triangle faces the camera
};

int64_t num_blocks(int64_t n) {
  return std::max<int64_t>(1, std::min<int64_t>((n + kThreads - 1) / kThreads, kMaxBlocks));
}

__device__ __forceinline__ long long edge_fn(int ax, int ay, int bx, int by, int px, int py) {
  return ((long long)bx - ax) * ((long long)py - ay) - ((long long)by - ay) * ((long long)px - ax);
}

// Top-left rule for a counter-clockwise triangle (positive area in a y-down
// frame): a sample exactly on edge a->b belongs to the triangle iff the edge
// direction (dx, dy) satisfies dy < 0 || (dy == 0 && dx > 0).  This equals
// testing the sample perturbed by (eps, eps^2): two triangles on opposite sides
// of a shared edge traverse it in opposite directions, so exactly one claims
// the sample (likewise exactly one triangle of a vertex fan).  On a silhouette
// fold both triangles lie on the same side, so both or neither claim it, which
// adds an entry/exit pair of equal depth and keeps the crossing parity.
__device__ __forceinline__ bool covers(long long e, int dx, int dy) {
  return e > 0 || (e == 0 && (dy < 0 || (dy == 0 && dx > 0)));
}

__device__ __forceinline__ unsigned long long make_key(float depth, int exit, int face) {
  if (depth == 0.0f) depth = 0.0f;  // fold -0 into +0 so ties stay ties
  const unsigned int u = __float_as_uint(depth);
  const unsigned int ord = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
  return ((unsigned long long)ord << 32) | ((unsigned long long)(exit & 1) << 31) |
         (unsigned long long)(unsigned int)face;
}

__device__ __forceinline__ float key_depth(unsigned long long key) {
  const unsigned int ord = (unsigned int)(key >> 32);
  const unsigned int u = (ord & 0x80000000u) ? (ord & 0x7FFFFFFFu) : ~ord;
  return __uint_as_float(u);
}

__device__ __forceinline__ int key_exit(unsigned long long key) { return (int)((key >> 31) & 1ull); }

__device__ __forceinline__ int key_face(unsigned long long key) { return (int)(key & 0x7FFFFFFFull); }

// ---------------------------------------------------------------------------
// Rasterization
// ---------------------------------------------------------------------------

__global__ void setup_kernel(const int* __restrict__ vxy, const float* __restrict__ vz,
                             const int* __restrict__ faces, int64_t B, int64_t V, int64_t F,
                             int H, int W, Tri* __restrict__ tris, long long* __restrict__ counts) {
  const int64_t n = B * F;
  for (int64_t t = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; t < n;
       t += (int64_t)gridDim.x * blockDim.x) {
    const int64_t b = t / F;
    Tri tri;
    long long count = 0;
    bool ok = true;
    for (int k = 0; k < 3; ++k) {
      const int idx = faces[t * 3 + k];
      if (idx < 0 || idx >= V) {
        ok = false;
        break;
      }
      const int64_t v = b * V + idx;
      tri.x[k] = vxy[v * 2 + 0];
      tri.y[k] = vxy[v * 2 + 1];
      tri.z[k] = vz[v];
      ok = ok && isfinite(tri.z[k]);  // invalid vertices carry z = NaN
    }
    if (ok) {
      long long area = edge_fn(tri.x[0], tri.y[0], tri.x[1], tri.y[1], tri.x[2], tri.y[2]);
      if (area != 0) {
        // Negative area in the y-down pixel frame <=> normal points to +z.
        tri.exit = area < 0 ? 1 : 0;
        if (area < 0) {
          int ti = tri.x[1]; tri.x[1] = tri.x[2]; tri.x[2] = ti;
          ti = tri.y[1]; tri.y[1] = tri.y[2]; tri.y[2] = ti;
          const float tz = tri.z[1]; tri.z[1] = tri.z[2]; tri.z[2] = tz;
          area = -area;
        }
        tri.area = area;
        const int min_x = min(tri.x[0], min(tri.x[1], tri.x[2]));
        const int max_x = max(tri.x[0], max(tri.x[1], tri.x[2]));
        const int min_y = min(tri.y[0], min(tri.y[1], tri.y[2]));
        const int max_y = max(tri.y[0], max(tri.y[1], tri.y[2]));
        constexpr int kOne = (1 << kSubpixelBits) - 1;
        // ceil / floor division by 2^bits (arithmetic shifts floor towards -inf).
        const int i0 = max((min_x + kOne) >> kSubpixelBits, 0);
        const int i1 = min(max_x >> kSubpixelBits, W - 1);
        const int j0 = max((min_y + kOne) >> kSubpixelBits, 0);
        const int j1 = min(max_y >> kSubpixelBits, H - 1);
        if (i1 >= i0 && j1 >= j0) {
          tri.i0 = i0;
          tri.j0 = j0;
          tri.bw = i1 - i0 + 1;
          count = (long long)(i1 - i0 + 1) * (long long)(j1 - j0 + 1);
        }
      }
    }
    counts[t] = count;
    if (count > 0) tris[t] = tri;
  }
}

// Depth on the segment a-b at sample p (p lies on the segment).  The endpoints
// are put in a canonical (lexicographic) order first, so both triangles sharing
// the edge produce bit-identical depths; otherwise their different barycentric
// rounding could swap a coincident entry/exit pair on a silhouette fold.
__device__ __forceinline__ float edge_depth(int ax, int ay, float az, int bx, int by, float bz,
                                            int px, int py) {
  if (ax > bx || (ax == bx && ay > by)) {
    int t = ax; ax = bx; bx = t;
    t = ay; ay = by; by = t;
    const float tz = az; az = bz; bz = tz;
  }
  const long long num = ax != bx ? (long long)px - ax : (long long)py - ay;
  const long long den = ax != bx ? (long long)bx - ax : (long long)by - ay;
  const double z = __ddiv_rn(__dadd_rn(__dmul_rn((double)az, (double)(den - num)),
                                       __dmul_rn((double)bz, (double)num)),
                             (double)den);
  return __double2float_rn(z);
}

// Depth of a covered sample.  Interior samples use the barycentric formula
// (explicitly rounded double arithmetic, no FMA contraction, so the PyTorch
// reference reproduces it bit for bit); samples exactly on an edge or vertex
// use only the shared edge / vertex data.
__device__ __forceinline__ float sample_depth(const Tri& tri, long long e0, long long e1, long long e2,
                                              int px, int py) {
  if (e0 == 0 && e1 == 0) return tri.z[2];
  if (e1 == 0 && e2 == 0) return tri.z[0];
  if (e2 == 0 && e0 == 0) return tri.z[1];
  if (e0 == 0) return edge_depth(tri.x[1], tri.y[1], tri.z[1], tri.x[2], tri.y[2], tri.z[2], px, py);
  if (e1 == 0) return edge_depth(tri.x[2], tri.y[2], tri.z[2], tri.x[0], tri.y[0], tri.z[0], px, py);
  if (e2 == 0) return edge_depth(tri.x[0], tri.y[0], tri.z[0], tri.x[1], tri.y[1], tri.z[1], px, py);
  const double num = __dadd_rn(__dadd_rn(__dmul_rn((double)e0, (double)tri.z[0]),
                                         __dmul_rn((double)e1, (double)tri.z[1])),
                               __dmul_rn((double)e2, (double)tri.z[2]));
  return __double2float_rn(__ddiv_rn(num, (double)tri.area));
}

// First index t in [0, n) with cend[t] > k (cend is an inclusive prefix sum).
__device__ __forceinline__ int64_t find_tri(const long long* __restrict__ cend, int64_t n, int64_t k) {
  int64_t lo = 0, hi = n;
  while (lo < hi) {
    const int64_t mid = (lo + hi) >> 1;
    if (cend[mid] > k) hi = mid;
    else lo = mid + 1;
  }
  return lo;
}

// One thread per candidate (triangle, pixel-in-bounding-box) pair, which keeps
// the work balanced whatever the triangle sizes are.  kWrite=false counts the
// fragments per pixel, kWrite=true scatters their keys.
template <bool kWrite>
__global__ void raster_kernel(const Tri* __restrict__ tris, const long long* __restrict__ cend,
                              int64_t n_tris, int64_t F, int H, int W,
                              int* __restrict__ pix_count, const int* __restrict__ pix_ptr,
                              unsigned long long* __restrict__ keys) {
  const int64_t n_cand = cend[n_tris - 1];  // read on the device: no host synchronization
  for (int64_t k = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; k < n_cand;
       k += (int64_t)gridDim.x * blockDim.x) {
    const int64_t t = find_tri(cend, n_tris, k);
    const int64_t local = k - (t > 0 ? cend[t - 1] : 0);
    const Tri tri = tris[t];
    const int i = tri.i0 + (int)(local % tri.bw);
    const int j = tri.j0 + (int)(local / tri.bw);
    const int px = i << kSubpixelBits;
    const int py = j << kSubpixelBits;
    const long long e0 = edge_fn(tri.x[1], tri.y[1], tri.x[2], tri.y[2], px, py);
    if (!covers(e0, tri.x[2] - tri.x[1], tri.y[2] - tri.y[1])) continue;
    const long long e1 = edge_fn(tri.x[2], tri.y[2], tri.x[0], tri.y[0], px, py);
    if (!covers(e1, tri.x[0] - tri.x[2], tri.y[0] - tri.y[2])) continue;
    const long long e2 = edge_fn(tri.x[0], tri.y[0], tri.x[1], tri.y[1], px, py);
    if (!covers(e2, tri.x[1] - tri.x[0], tri.y[1] - tri.y[0])) continue;

    const int64_t b = t / F;
    const int64_t pix = (b * H + j) * W + i;
    if (!kWrite) {
      atomicAdd(pix_count + pix, 1);
    } else {
      const float depth = sample_depth(tri, e0, e1, e2, px, py);
      const int slot = pix_ptr[pix] + atomicAdd(pix_count + pix, 1);
      keys[slot] = make_key(depth, tri.exit, (int)(t - b * F));
    }
  }
}

// ---------------------------------------------------------------------------
// FOF integration
// ---------------------------------------------------------------------------

// Contribution of the interval [t0, t1] (t = normalized depth in [0, 1]) to
// coefficient c, up to the factor applied in integrate_pixel():
//   2 * int_{t0}^{t1} cos(c*pi*t) dt = 2/(c*pi) * (sin(c*pi*t1) - sin(c*pi*t0))
//                                    = 4/(c*pi) * cos(c*pi*m) * sin(c*pi*h),
// with midpoint m and half width h.  The product form avoids the cancellation
// of subtracting two nearly equal sines for thin intervals.
__device__ __forceinline__ float interval_term(float t0, float t1, int c) {
  const float h = 0.5f * (t1 - t0);
  if (c == 0) return h;
  const float m = 0.5f * (t0 + t1);
  return cospif((float)c * m) * sinpif((float)c * h);
}

__device__ float integrate_pixel(const unsigned long long* __restrict__ keys, int n, int c,
                                 float z_min, float inv_range, int rule) {
  float acc = 0.f;
  auto to_t = [&](unsigned long long key) {
    return fminf(fmaxf((key_depth(key) - z_min) * inv_range, 0.f), 1.f);
  };
  if (rule == kParity) {
    for (int k = 0; k + 1 < n; k += 2) acc += interval_term(to_t(keys[k]), to_t(keys[k + 1]), c);
  } else if (rule == kWinding) {
    // Nonzero rule on the winding number (entries - exits).
    int w = 0;
    float t_in = 0.f;
    for (int k = 0; k < n; ++k) {
      const int w_new = w + (key_exit(keys[k]) ? -1 : 1);
      if (w == 0) t_in = to_t(keys[k]);
      else if (w_new == 0) acc += interval_term(t_in, to_t(keys[k]), c);
      w = w_new;
    }
  } else if (rule == kHybrid) {
    int w = 0, n_out = 0;
    float t_in = 0.f, t_last = 0.f, p0 = 0.f, p1 = 0.f;
    bool has_exit = false, just_closed = false, pending = false;
    for (int k = 0; k < n; ++k) {
      const float t = to_t(keys[k]);
      if (!key_exit(keys[k])) {
        if (w == 0) {
          if (pending) acc += interval_term(p0, p1, c);
          pending = false;
          t_in = t;
          has_exit = false;
        }
        ++w;
        just_closed = false;
      } else if (w > 0) {
        --w;
        has_exit = true;
        t_last = t;
        if (w == 0) {
          pending = true;
          just_closed = true;
          p0 = t_in;
          p1 = t;
          ++n_out;
        }
      } else if (just_closed) {
        p1 = t;  // exit run after a closed interval: extend it
      }
    }
    if (pending) acc += interval_term(p0, p1, c);
    if (w > 0 && has_exit) {
      acc += interval_term(t_in, t_last, c);
      ++n_out;
    }
    if (n_out == 0)  // no consistent interval: orientation is unreliable -> even-odd
      for (int k = 0; k + 1 < n; k += 2) acc += interval_term(to_t(keys[k]), to_t(keys[k + 1]), c);
  } else {  // kRobust
    int state = 0;  // 0: outside, 1: inside, 2: just exited (exit may still move)
    float t_in = 0.f, t_out = 0.f;
    for (int k = 0; k < n; ++k) {
      const int exit = key_exit(keys[k]);
      if (state == 0) {
        if (!exit) { state = 1; t_in = to_t(keys[k]); }
      } else if (state == 1) {
        if (exit) { state = 2; t_out = to_t(keys[k]); }
      } else if (exit) {
        t_out = to_t(keys[k]);
      } else {
        acc += interval_term(t_in, t_out, c);
        state = 1;
        t_in = to_t(keys[k]);
      }
    }
    if (state == 2) acc += interval_term(t_in, t_out, c);
  }
  // Unclosed trailing intervals are dropped by every rule.
  // a_0 = 2 * sum(t1 - t0) = 4 * sum(h);  a_c = 4 / (c*pi) * sum(cos(c*pi*m) * sin(c*pi*h)).
  return c == 0 ? 4.f * acc : acc * (4.f / ((float)c * 3.14159265358979323846f));
}

// One thread per output element of the [B, C, H, W] tensor (fully coalesced
// writes, every element written exactly once, empty pixels become 0).
__global__ void fof_kernel(const int* __restrict__ pix_ptr, const unsigned long long* __restrict__ keys,
                           int64_t B, int64_t C, int64_t HW, float z_min, float inv_range, int rule,
                           float* __restrict__ out) {
  const int64_t total = B * C * HW;
  for (int64_t t = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; t < total;
       t += (int64_t)gridDim.x * blockDim.x) {
    const int64_t hw = t % HW;
    const int64_t bc = t / HW;
    const int c = (int)(bc % C);
    const int64_t pix = (bc / C) * HW + hw;
    const int beg = pix_ptr[pix];
    const int end = pix_ptr[pix + 1];
    out[t] = beg < end ? integrate_pixel(keys + beg, end - beg, c, z_min, inv_range, rule) : 0.f;
  }
}

// ---------------------------------------------------------------------------
// Front / back surface lookup
// ---------------------------------------------------------------------------

// For every pixel, take the front-most (largest z) and back-most (smallest z)
// fragment and recover the face index, depth and barycentric coordinates
// (w.r.t. the face's original vertex order) at the pixel centre.
__global__ void front_back_kernel(const int* __restrict__ pix_ptr,
                                  const unsigned long long* __restrict__ keys,
                                  const int* __restrict__ vxy, const int* __restrict__ faces,
                                  int64_t P, int64_t HW, int W, int64_t V, int64_t F,
                                  int* __restrict__ face_out, float* __restrict__ depth_out,
                                  float* __restrict__ bary_out) {
  for (int64_t pix = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; pix < P;
       pix += (int64_t)gridDim.x * blockDim.x) {
    const int beg = pix_ptr[pix];
    const int end = pix_ptr[pix + 1];
    const int64_t b = pix / HW;
    const int64_t hw = pix % HW;
    const int px = (int)(hw % W) << kSubpixelBits;
    const int py = (int)(hw / W) << kSubpixelBits;
    for (int s = 0; s < 2; ++s) {  // s = 0: front, s = 1: back
      const int64_t o = s * P + pix;
      if (beg == end) {
        face_out[o] = -1;
        depth_out[o] = __int_as_float(0x7fc00000);  // NaN, filled on the Python side
        bary_out[o * 3 + 0] = bary_out[o * 3 + 1] = bary_out[o * 3 + 2] = 0.f;
        continue;
      }
      const unsigned long long key = keys[s == 0 ? end - 1 : beg];
      const int f = key_face(key);
      const int* fc = faces + (b * F + f) * 3;
      int x[3], y[3];
      for (int k = 0; k < 3; ++k) {
        const int64_t v = b * V + fc[k];
        x[k] = vxy[v * 2 + 0];
        y[k] = vxy[v * 2 + 1];
      }
      const double area = (double)edge_fn(x[0], y[0], x[1], y[1], x[2], y[2]);
      face_out[o] = f;
      depth_out[o] = key_depth(key);
      bary_out[o * 3 + 0] = (float)((double)edge_fn(x[1], y[1], x[2], y[2], px, py) / area);
      bary_out[o * 3 + 1] = (float)((double)edge_fn(x[2], y[2], x[0], y[0], px, py) / area);
      bary_out[o * 3 + 2] = (float)((double)edge_fn(x[0], y[0], x[1], y[1], px, py) / area);
    }
  }
}

void check_cuda(const at::Tensor& t, const char* name, at::ScalarType dtype, int64_t dim) {
  TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(t.scalar_type() == dtype, name, " must have dtype ", dtype, ", got ", t.scalar_type());
  TORCH_CHECK(t.dim() == dim, name, " must have ", dim, " dims, got ", t.dim());
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

}  // namespace

// vxy:   [B, V, 2] int32 fixed-point pixel coordinates
// vz:    [B, V]    float32 depth (NaN marks an invalid vertex)
// faces: [B, F, 3] int32 vertex indices (out-of-range indices skip the face)
// returns pix_ptr [B*H*W + 1] int32 (CSR offsets) and keys [N] int64 holding
// the uint64 fragment keys, sorted inside every pixel.
std::vector<at::Tensor> rasterize(const at::Tensor& vxy, const at::Tensor& vz,
                                  const at::Tensor& faces, int64_t H, int64_t W) {
  check_cuda(vxy, "vxy", at::kInt, 3);
  check_cuda(vz, "vz", at::kFloat, 2);
  check_cuda(faces, "faces", at::kInt, 3);
  const int64_t B = vz.size(0), V = vz.size(1), F = faces.size(1);
  TORCH_CHECK(vxy.size(0) == B && vxy.size(1) == V && vxy.size(2) == 2, "vxy must be [B, V, 2]");
  TORCH_CHECK(faces.size(0) == B && faces.size(2) == 3, "faces must be [B, F, 3]");
  TORCH_CHECK(vxy.device() == vz.device() && vz.device() == faces.device(),
              "all inputs must be on the same device");
  TORCH_CHECK(H > 0 && W > 0, "resolution must be positive");
  TORCH_CHECK(W - 1 <= (std::numeric_limits<int>::max() >> kSubpixelBits) &&
                  H - 1 <= (std::numeric_limits<int>::max() >> kSubpixelBits),
              "resolution too large");
  TORCH_CHECK(F < (int64_t(1) << 31), "too many faces per mesh");
  const int64_t P = B * H * W;
  TORCH_CHECK(P < std::numeric_limits<int>::max(), "B * H * W must be < 2^31");

  const at::cuda::OptionalCUDAGuard device_guard(vz.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const auto i32 = vz.options().dtype(at::kInt);
  const auto i64 = vz.options().dtype(at::kLong);
  if (P == 0) return {at::zeros({1}, i32), at::empty({0}, i64)};  // empty batch

  // 1. Triangle setup: canonical orientation, exit flag, clipped bounding box.
  at::Tensor tris = at::empty({std::max<int64_t>(B * F, 1) * (int64_t)sizeof(Tri)},
                              vz.options().dtype(at::kByte));
  at::Tensor counts = at::empty({B * F}, i64);
  if (B * F > 0) {
    setup_kernel<<<num_blocks(B * F), kThreads, 0, stream>>>(
        vxy.data_ptr<int>(), vz.data_ptr<float>(), faces.data_ptr<int>(), B, V, F, (int)H, (int)W,
        reinterpret_cast<Tri*>(tris.data_ptr<uint8_t>()),
        reinterpret_cast<long long*>(counts.data_ptr<int64_t>()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  const at::Tensor cend = at::cumsum(counts, 0);
  // The candidate total stays on the device; a fixed, device-filling grid walks it.
  const int64_t raster_blocks =
      std::min<int64_t>(kMaxBlocks, (int64_t)at::cuda::getCurrentDeviceProperties()->multiProcessorCount * 8);

  // 2. Count fragments per pixel.
  at::Tensor pix_count = at::zeros({P}, i32);
  if (B * F > 0) {
    raster_kernel<false><<<raster_blocks, kThreads, 0, stream>>>(
        reinterpret_cast<const Tri*>(tris.data_ptr<uint8_t>()),
        reinterpret_cast<const long long*>(cend.data_ptr<int64_t>()), B * F, F, (int)H, (int)W,
        pix_count.data_ptr<int>(), nullptr, nullptr);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  // 3. CSR offsets (the fragment total is the one value the host needs: allocation size).
  const at::Tensor csum = at::cumsum(pix_count, 0, at::kLong);
  const int64_t n_frag = csum[P - 1].item<int64_t>();
  TORCH_CHECK(n_frag < std::numeric_limits<int>::max(), "too many fragments (", n_frag, ")");
  at::Tensor pix_ptr = at::zeros({P + 1}, i32);
  pix_ptr.narrow(0, 1, P).copy_(csum);

  // 4. Scatter keys, 5. sort them inside every pixel.
  at::Tensor keys = at::empty({n_frag}, i64);
  at::Tensor sorted = at::empty({n_frag}, i64);
  if (n_frag > 0) {
    pix_count.zero_();
    raster_kernel<true><<<raster_blocks, kThreads, 0, stream>>>(
        reinterpret_cast<const Tri*>(tris.data_ptr<uint8_t>()),
        reinterpret_cast<const long long*>(cend.data_ptr<int64_t>()), B * F, F, (int)H, (int)W,
        pix_count.data_ptr<int>(), pix_ptr.data_ptr<int>(),
        reinterpret_cast<unsigned long long*>(keys.data_ptr<int64_t>()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    const auto* k_in = reinterpret_cast<const unsigned long long*>(keys.data_ptr<int64_t>());
    auto* k_out = reinterpret_cast<unsigned long long*>(sorted.data_ptr<int64_t>());
    const int* offsets = pix_ptr.data_ptr<int>();
    size_t temp_bytes = 0;
    C10_CUDA_CHECK(::CUB_WRAPPED_NAMESPACE::cub::DeviceSegmentedSort::SortKeys(nullptr, temp_bytes, k_in, k_out, (int)n_frag,
                                                      (int)P, offsets, offsets + 1, stream));
    at::Tensor temp = at::empty({(int64_t)std::max<size_t>(temp_bytes, 1)}, vz.options().dtype(at::kByte));
    C10_CUDA_CHECK(::CUB_WRAPPED_NAMESPACE::cub::DeviceSegmentedSort::SortKeys(temp.data_ptr(), temp_bytes, k_in, k_out,
                                                      (int)n_frag, (int)P, offsets, offsets + 1, stream));
  }
  return {pix_ptr, sorted};
}

// Integrates the cosine basis over the occupied intervals of every pixel.
// Returns [B, C, H, W] float32 coefficients.
at::Tensor fof_integrate(const at::Tensor& pix_ptr, const at::Tensor& keys, int64_t B, int64_t H,
                         int64_t W, int64_t C, double z_min, double z_max, int64_t rule) {
  check_cuda(pix_ptr, "pix_ptr", at::kInt, 1);
  check_cuda(keys, "keys", at::kLong, 1);
  TORCH_CHECK(pix_ptr.size(0) == B * H * W + 1, "pix_ptr must have B*H*W+1 entries");
  TORCH_CHECK(C > 0, "num_coeffs must be positive");
  TORCH_CHECK(z_max > z_min, "z_max must be larger than z_min");
  TORCH_CHECK(rule == kRobust || rule == kParity || rule == kWinding || rule == kHybrid, "unknown fill rule");
  const at::cuda::OptionalCUDAGuard device_guard(keys.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  at::Tensor out = at::empty({B, C, H, W}, keys.options().dtype(at::kFloat));
  const int64_t total = B * C * H * W;
  if (total > 0) {
    fof_kernel<<<num_blocks(total), kThreads, 0, stream>>>(
        pix_ptr.data_ptr<int>(), reinterpret_cast<const unsigned long long*>(keys.data_ptr<int64_t>()),
        B, C, H * W, (float)z_min, (float)(1.0 / (z_max - z_min)), (int)rule, out.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return out;
}

// Returns face [2, B, H, W] int32 (-1 = empty), depth [2, B, H, W] float32
// (NaN = empty) and barycentrics [2, B, H, W, 3] float32; index 0 is the
// front-most surface (largest z), index 1 the back-most (smallest z).
std::vector<at::Tensor> front_back(const at::Tensor& pix_ptr, const at::Tensor& keys,
                                   const at::Tensor& vxy, const at::Tensor& faces, int64_t H,
                                   int64_t W) {
  check_cuda(pix_ptr, "pix_ptr", at::kInt, 1);
  check_cuda(keys, "keys", at::kLong, 1);
  check_cuda(vxy, "vxy", at::kInt, 3);
  check_cuda(faces, "faces", at::kInt, 3);
  const int64_t B = vxy.size(0), V = vxy.size(1), F = faces.size(1);
  const int64_t P = B * H * W;
  TORCH_CHECK(pix_ptr.size(0) == P + 1, "pix_ptr must have B*H*W+1 entries");
  const at::cuda::OptionalCUDAGuard device_guard(keys.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  at::Tensor face = at::empty({2, B, H, W}, vxy.options());
  at::Tensor depth = at::empty({2, B, H, W}, vxy.options().dtype(at::kFloat));
  at::Tensor bary = at::empty({2, B, H, W, 3}, vxy.options().dtype(at::kFloat));
  if (P > 0) {
    front_back_kernel<<<num_blocks(P), kThreads, 0, stream>>>(
        pix_ptr.data_ptr<int>(), reinterpret_cast<const unsigned long long*>(keys.data_ptr<int64_t>()),
        vxy.data_ptr<int>(), faces.data_ptr<int>(), P, H * W, (int)W, V, F, face.data_ptr<int>(),
        depth.data_ptr<float>(), bary.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {face, depth, bary};
}

}  // namespace fof
