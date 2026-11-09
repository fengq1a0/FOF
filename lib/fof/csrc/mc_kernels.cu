// Crack-free marching cubes (see lib/fof/mc.py and lib/fof/mc_tables.py for the conventions).
//
//   1. order-preserving compaction of the active cells (configuration neither 0 nor 255),
//      without any per-cell buffer: per-tile counts -> scan -> per-tile writes;
//   2. per active cell: configuration + asymptotic-decider bits -> table variant;
//   3. per active cell: emit triangles as global vertex ids and the centre-vertex records.
// Deduplication of vertex ids and vertex positions are done with PyTorch ops (mc.py), so the
// CUDA and the PyTorch backends share them and produce identical meshes.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace fof {
namespace {

constexpr int kMcThreads = 256;
constexpr int kMcItems = 16;  // cells per thread per tile
constexpr int kMcTile = kMcThreads * kMcItems;
constexpr int64_t kMcMaxBlocks = 1 << 16;

int64_t mc_blocks(int64_t n) {
  return std::max<int64_t>(1, std::min<int64_t>((n + kMcThreads - 1) / kMcThreads, kMcMaxBlocks));
}

// The grid is the stored volume plus an optional virtual border of ``pad`` samples (0 or 1) of
// value ``pad_value`` on every side: identical results to padding the volume explicitly, without
// the copy (which would double the peak memory of large volumes).
struct Grid {
  int64_t D, H, W;     // grid samples, border included
  int64_t Hc, Wc;      // cells per row / column (H - 1, W - 1)
  int64_t n_cells;
  int64_t vD, vH, vW;  // stored volume
  int64_t pad;         // border width (0 or 1)
  float pad_value;     // value of the border samples
};

__device__ __forceinline__ float sample(const float* __restrict__ vol, const Grid& g, int64_t d, int64_t h,
                                        int64_t w) {
  d -= g.pad;
  h -= g.pad;
  w -= g.pad;
  if (d < 0 || h < 0 || w < 0 || d >= g.vD || h >= g.vH || w >= g.vW) return g.pad_value;
  return vol[(d * g.vH + h) * g.vW + w];
}

__device__ __forceinline__ void cell_coords(const Grid& g, int64_t cell, int64_t& d, int64_t& h, int64_t& w) {
  w = cell % g.Wc;
  const int64_t t = cell / g.Wc;
  h = t % g.Hc;
  d = t / g.Hc;
}

// Level of the classification: for a float sample v and the double level L, v > L exactly iff
// v > Lf, where Lf is the largest float not above L.  The hot loops therefore compare floats.
struct Level {
  double d;  // the level
  float f;   // largest float <= d
};

// 8-bit configuration (bit k: corner k above the level).  ``vals`` (optional) receives the corner
// values relative to the level, computed in double (exact widening, no overflow).
__device__ __forceinline__ int cell_config(const float* __restrict__ vol, const Grid& g, int64_t d, int64_t h,
                                           int64_t w, const Level& level, double* vals) {
  int cfg = 0;
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    const float v = sample(vol, g, d + ((k >> 2) & 1), h + ((k >> 1) & 1), w + (k & 1));
    cfg |= (v > level.f ? 1 : 0) << k;
    if (vals) vals[k] = __dsub_rn((double)v, level.d);
  }
  return cfg;
}

__device__ __forceinline__ bool cell_active(const float* __restrict__ vol, const Grid& g, int64_t cell,
                                            const Level& level) {
  int64_t d, h, w;
  cell_coords(g, cell, d, h, w);
  const int cfg = cell_config(vol, g, d, h, w, level, nullptr);
  return cfg != 0 && cfg != 255;
}

__global__ void mc_count_kernel(const float* __restrict__ vol, Grid g, Level level, int* __restrict__ tile_count) {
  for (int64_t tile = blockIdx.x; tile * kMcTile < g.n_cells; tile += gridDim.x) {
    int count = 0;
    for (int j = 0; j < kMcItems; ++j) {
      const int64_t cell = tile * kMcTile + (int64_t)j * kMcThreads + threadIdx.x;
      const bool act = cell < g.n_cells && cell_active(vol, g, cell, level);
      count += __syncthreads_count(act);
    }
    if (threadIdx.x == 0) tile_count[tile] = count;
  }
}

// Writes the active cells of every tile in increasing cell order.
__global__ void mc_compact_kernel(const float* __restrict__ vol, Grid g, Level level,
                                  const int64_t* __restrict__ tile_offset, int64_t* __restrict__ cells) {
  __shared__ int warp_sum[kMcThreads / 32];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  for (int64_t tile = blockIdx.x; tile * kMcTile < g.n_cells; tile += gridDim.x) {
    int64_t base = tile_offset[tile];
    for (int j = 0; j < kMcItems; ++j) {
      const int64_t cell = tile * kMcTile + (int64_t)j * kMcThreads + threadIdx.x;
      const bool act = cell < g.n_cells && cell_active(vol, g, cell, level);
      const unsigned ballot = __ballot_sync(0xffffffffu, act);
      if (lane == 0) warp_sum[warp] = __popc(ballot);
      __syncthreads();
      int before = 0, total = 0;
      for (int k = 0; k < kMcThreads / 32; ++k) {
        before += k < warp ? warp_sum[k] : 0;
        total += warp_sum[k];
      }
      if (act) cells[base + before + __popc(ballot & ((1u << lane) - 1u))] = cell;
      base += total;
      __syncthreads();
    }
  }
}

struct Tables {
  const int* amb;       // [256] ambiguous-face mask
  const int* base;      // [256] first variant of each configuration
  const int* ntri;      // [V] triangles per variant
  const int8_t* tris;   // [V, tri_stride] vertex codes (edges 0..11, centres 12..)
  const int* ncen;      // [V] centre vertices per variant
  const int* cloop;     // [V, max_centres] edge masks of the centre loops
  const int8_t* faces;  // [6, 4] cyclic corner order of each face
  const int8_t* edges;  // [12, 4] (dx, dy, dz, axis) of each edge's start corner
  int tri_stride, max_centres;
};

__global__ void mc_variant_kernel(const float* __restrict__ vol, Grid g, Level level, Tables tb,
                                  const int64_t* __restrict__ cells, int64_t n, int* __restrict__ var,
                                  int* __restrict__ ntri, int* __restrict__ ncen) {
  for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < n; i += (int64_t)gridDim.x * blockDim.x) {
    int64_t d, h, w;
    cell_coords(g, cells[i], d, h, w);
    double vals[8];
    const int cfg = cell_config(vol, g, d, h, w, level, vals);
    const int mask = tb.amb[cfg];
    int bits = 0, rank = 0;
    for (int f = 0; f < 6; ++f) {
      if (!((mask >> f) & 1)) continue;
      const double v0 = vals[tb.faces[4 * f]], v1 = vals[tb.faces[4 * f + 1]];
      const double v2 = vals[tb.faces[4 * f + 2]], v3 = vals[tb.faces[4 * f + 3]];
      // Asymptotic decider: the inside diagonal is connected iff the product of its (positive)
      // values exceeds the product of the outside diagonal.  Only the face's own values are used,
      // so both cells sharing the face agree.  Double precision cannot overflow here, and the
      // explicitly rounded operations reproduce the PyTorch backend bit for bit.
      const double p02 = __dmul_rn(v0, v2), p13 = __dmul_rn(v1, v3);
      const bool connect = v0 > 0.0 ? p02 > p13 : p13 > p02;
      bits |= (connect ? 1 : 0) << rank;
      ++rank;
    }
    const int v = tb.base[cfg] + bits;
    var[i] = v;
    ntri[i] = tb.ntri[v];
    ncen[i] = tb.ncen[v];
  }
}

__device__ __forceinline__ int64_t vertex_id(const Grid& g, const Tables& tb, int64_t cell, int64_t d, int64_t h,
                                             int64_t w, int code) {
  if (code >= 12) return 3 * g.D * g.H * g.W + cell * tb.max_centres + (code - 12);
  const int8_t* e = tb.edges + 4 * code;
  return (((d + e[2]) * g.H + (h + e[1])) * g.W + (w + e[0])) * 3 + e[3];
}

__global__ void mc_emit_kernel(Grid g, Tables tb, const int64_t* __restrict__ cells, int64_t n,
                               const int* __restrict__ var, const int64_t* __restrict__ tri_off,
                               const int64_t* __restrict__ cen_off, int64_t* __restrict__ tri_ids,
                               int64_t* __restrict__ centres) {
  for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < n; i += (int64_t)gridDim.x * blockDim.x) {
    const int64_t cell = cells[i];
    int64_t d, h, w;
    cell_coords(g, cell, d, h, w);
    const int v = var[i];
    const int nt = tb.ntri[v];
    int64_t* out = tri_ids + tri_off[i] * 3;
    for (int k = 0; k < 3 * nt; ++k) out[k] = vertex_id(g, tb, cell, d, h, w, tb.tris[(int64_t)v * tb.tri_stride + k]);
    const int nc = tb.ncen[v];
    for (int s = 0; s < nc; ++s) {
      int64_t* rec = centres + (cen_off[i] + s) * 13;
      rec[0] = vertex_id(g, tb, cell, d, h, w, 12 + s);
      const int loop = tb.cloop[v * tb.max_centres + s];
      int k = 1;
      for (int e = 0; e < 12; ++e)
        if ((loop >> e) & 1) rec[k++] = vertex_id(g, tb, cell, d, h, w, e);
      for (; k < 13; ++k) rec[k] = -1;
    }
  }
}

template <typename T>
const T* checked_ptr(const at::Tensor& t, at::ScalarType dtype, const char* name, const at::Device& dev) {
  TORCH_CHECK(t.device() == dev && t.scalar_type() == dtype && t.is_contiguous(), "marching_cubes: bad table ", name);
  return t.data_ptr<T>();
}

}  // namespace

// Returns {triangle vertex ids [F, 3] int64, centre records [C, 13] int64}; vertex ids refer to the
// grid with the virtual border (``pad`` = 0 or 1 samples of value ``pad_value`` on every side).
std::vector<at::Tensor> marching_cubes(const at::Tensor& vol, double level, int64_t pad, double pad_value,
                                       const at::Tensor& amb,
                                       const at::Tensor& base, const at::Tensor& ntri, const at::Tensor& tris,
                                       const at::Tensor& ncen, const at::Tensor& cloop, const at::Tensor& faces,
                                       const at::Tensor& edges, int64_t max_centres) {
  TORCH_CHECK(vol.is_cuda() && vol.scalar_type() == at::kFloat && vol.dim() == 3 && vol.is_contiguous(),
              "vol must be a contiguous float32 CUDA tensor [D, H, W]");
  const at::cuda::OptionalCUDAGuard device_guard(vol.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const auto dev = vol.device();
  const auto i64 = vol.options().dtype(at::kLong);
  const auto i32 = vol.options().dtype(at::kInt);
  TORCH_CHECK(pad == 0 || pad == 1, "pad must be 0 or 1");
  TORCH_CHECK(std::isfinite((float)pad_value), "pad_value must be finite in float32");
  Grid g;
  g.vD = vol.size(0);
  g.vH = vol.size(1);
  g.vW = vol.size(2);
  TORCH_CHECK(g.vD > 0 && g.vH > 0 && g.vW > 0, "volume must not be empty");
  g.pad = pad;
  g.pad_value = (float)pad_value;
  g.D = g.vD + 2 * pad;
  g.H = g.vH + 2 * pad;
  g.W = g.vW + 2 * pad;
  g.Hc = g.H - 1;
  g.Wc = g.W - 1;
  TORCH_CHECK(g.D >= 2 && g.H >= 2 && g.W >= 2, "grid must be at least 2x2x2");
  g.n_cells = (g.D - 1) * g.Hc * g.Wc;
  TORCH_CHECK(3 * g.D * g.H * g.W < std::numeric_limits<int64_t>::max() / 4, "volume too large");
  Tables tb;
  tb.amb = checked_ptr<int>(amb, at::kInt, "amb", dev);
  tb.base = checked_ptr<int>(base, at::kInt, "base", dev);
  tb.ntri = checked_ptr<int>(ntri, at::kInt, "ntri", dev);
  tb.tris = checked_ptr<int8_t>(tris, at::kChar, "tris", dev);
  tb.ncen = checked_ptr<int>(ncen, at::kInt, "ncen", dev);
  tb.cloop = checked_ptr<int>(cloop, at::kInt, "cloop", dev);
  tb.faces = checked_ptr<int8_t>(faces, at::kChar, "faces", dev);
  tb.edges = checked_ptr<int8_t>(edges, at::kChar, "edges", dev);
  tb.tri_stride = (int)tris.size(1);
  tb.max_centres = (int)max_centres;
  TORCH_CHECK(cloop.dim() == 2 && cloop.size(1) == max_centres, "cloop must be [V, max_centres]");
  Level lv;
  lv.d = level;
  lv.f = (float)level;
  if ((double)lv.f > level) lv.f = nextafterf(lv.f, -INFINITY);
  const float* vp = vol.data_ptr<float>();

  // 1. active cells
  const int64_t n_tiles = (g.n_cells + kMcTile - 1) / kMcTile;
  const int64_t tile_blocks = std::min<int64_t>(n_tiles, kMcMaxBlocks);
  at::Tensor tile_count = at::empty({n_tiles}, i32);
  mc_count_kernel<<<tile_blocks, kMcThreads, 0, stream>>>(vp, g, lv, tile_count.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  const at::Tensor tile_end = at::cumsum(tile_count, 0, at::kLong);
  const int64_t n_act = tile_end[n_tiles - 1].item<int64_t>();
  if (n_act == 0) return {at::empty({0, 3}, i64), at::empty({0, 13}, i64)};
  const at::Tensor tile_offset = tile_end - tile_count;
  at::Tensor cells = at::empty({n_act}, i64);
  mc_compact_kernel<<<tile_blocks, kMcThreads, 0, stream>>>(vp, g, lv, tile_offset.data_ptr<int64_t>(),
                                                            cells.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  // 2. table variants and output sizes
  at::Tensor var = at::empty({n_act}, i32), nt = at::empty({n_act}, i32), nc = at::empty({n_act}, i32);
  mc_variant_kernel<<<mc_blocks(n_act), kMcThreads, 0, stream>>>(vp, g, lv, tb, cells.data_ptr<int64_t>(), n_act,
                                                                 var.data_ptr<int>(), nt.data_ptr<int>(),
                                                                 nc.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  const at::Tensor tri_end = at::cumsum(nt, 0, at::kLong);
  const at::Tensor cen_end = at::cumsum(nc, 0, at::kLong);
  const at::Tensor totals = at::stack({tri_end[n_act - 1], cen_end[n_act - 1]}).cpu();
  const int64_t n_tri = totals[0].item<int64_t>(), n_cen = totals[1].item<int64_t>();

  // 3. triangles and centre records
  const at::Tensor tri_off = tri_end - nt, cen_off = cen_end - nc;
  at::Tensor tri_ids = at::empty({n_tri, 3}, i64);
  at::Tensor centres = at::empty({n_cen, 13}, i64);
  mc_emit_kernel<<<mc_blocks(n_act), kMcThreads, 0, stream>>>(
      g, tb, cells.data_ptr<int64_t>(), n_act, var.data_ptr<int>(), tri_off.data_ptr<int64_t>(),
      cen_off.data_ptr<int64_t>(), tri_ids.data_ptr<int64_t>(), centres.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {tri_ids, centres};
}

}  // namespace fof
