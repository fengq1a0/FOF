#include <torch/extension.h>

#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cmath>

#include <vector>

const float PI = acos(-1.0);

namespace{

static inline __device__ float atomicMax(float* addr, float value)
{
    unsigned int* const addr_as_ui = (unsigned int*)addr;
    unsigned int old = *addr_as_ui, assumed;
    do {
        assumed = old;
        if (__uint_as_float(assumed) >= value) break;
        old = atomicCAS(addr_as_ui, assumed, __float_as_uint(value));
    } while (assumed != old);
    return old;
}

static inline __device__ float atomicMin(float* addr, float value)
{
    unsigned int* const addr_as_ui = (unsigned int*)addr;
    unsigned int old = *addr_as_ui, assumed;
    do {
        assumed = old;
        if (__uint_as_float(assumed) <= value) break;
        old = atomicCAS(addr_as_ui, assumed, __float_as_uint(value));
    } while (assumed != old);
    return old;
}




__device__ __forceinline__ float max3f(float a, float b, float c) {
    return fmaxf(fmaxf(a,b),c);
}

__device__ __forceinline__ float min3f(float a, float b, float c) {
    return fminf(fminf(a,b),c);
}

__device__ __forceinline__ bool compare(float a0, int a1, float b0, int b1) {
    if (a0 < b0) return true;
    if (a0 > b0) return false;
    if (a1 < b1) return true;
    return false;
}

__global__ void fof_cuda_render_kernel0(
    torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> v,
    int res, int* cnt, int* ccnt)
{
    //batch index
    const int n = blockIdx.y;
    // column index
    const int c = blockIdx.x * blockDim.x + threadIdx.x;

    if (c >= v.size(1)) return;

    auto p1 = v[n][c][0];
    auto p2 = v[n][c][1];
    auto p3 = v[n][c][2];

    int iMax = floorf(max3f(p1[0],p2[0],p3[0])); iMax = min(iMax+1, res);
    int jMax = floorf(max3f(p1[1],p2[1],p3[1])); jMax = min(jMax+1, res); 
    int iMin =  ceilf(min3f(p1[0],p2[0],p3[0])); iMin = max(iMin, 0);     
    int jMin =  ceilf(min3f(p1[1],p2[1],p3[1])); jMin = max(jMin, 0);     
    
    for (int j=jMin;j<jMax;j++)
    for (int i=iMin;i<iMax;i++)
    {
        float w3 = (p2[0]-p1[0])*(j-p1[1]) - (p2[1]-p1[1])*(i-p1[0]);
        float w1 = (p3[0]-p2[0])*(j-p2[1]) - (p3[1]-p2[1])*(i-p2[0]);
        float w2 = (p1[0]-p3[0])*(j-p3[1]) - (p1[1]-p3[1])*(i-p3[0]);
        float ss = w1+w2+w3;
        if (ss==0) continue;
        if ((w1>=0 && w2>=0 && w3>=0) || (w1<=0 && w2<=0 && w3<=0))
        {
            atomicAdd(&cnt[n*res*res+j*res+i], 1);
            ccnt[n*res*res+j*res+i] = 1;
        }
    }
}

__global__ void fof_normal_cuda_render_kernel0(
    torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> v,
    float* depth_F, float* depth_B,
    int res, int* cnt, int* ccnt)
{
    //batch index
    const int n = blockIdx.y;
    // column index
    const int c = blockIdx.x * blockDim.x + threadIdx.x;

    if (c >= v.size(1)) return;

    auto p1 = v[n][c][0];
    auto p2 = v[n][c][1];
    auto p3 = v[n][c][2];

    int iMax = floorf(max3f(p1[0],p2[0],p3[0])); iMax = min(iMax+1, res);
    int jMax = floorf(max3f(p1[1],p2[1],p3[1])); jMax = min(jMax+1, res); 
    int iMin =  ceilf(min3f(p1[0],p2[0],p3[0])); iMin = max(iMin, 0);     
    int jMin =  ceilf(min3f(p1[1],p2[1],p3[1])); jMin = max(jMin, 0);     
    
    for (int j=jMin;j<jMax;j++)
    for (int i=iMin;i<iMax;i++)
    {
        float w3 = (p2[0]-p1[0])*(j-p1[1]) - (p2[1]-p1[1])*(i-p1[0]);
        float w1 = (p3[0]-p2[0])*(j-p2[1]) - (p3[1]-p2[1])*(i-p2[0]);
        float w2 = (p1[0]-p3[0])*(j-p3[1]) - (p1[1]-p3[1])*(i-p3[0]);
        float ss = w1+w2+w3;
        if (ss==0) continue;
        if ((w1>=0 && w2>=0 && w3>=0) || (w1<=0 && w2<=0 && w3<=0))
        {
            float d_tmp = (w1*p1[2]+w2*p2[2]+w3*p3[2])/ss;
            atomicMax(&depth_F[n*res*res+j*res+i], d_tmp);
            atomicMin(&depth_B[n*res*res+j*res+i], d_tmp);
            atomicAdd(&cnt[n*res*res+j*res+i], 1);
            ccnt[n*res*res+j*res+i] = 1;
        }
    }
}

__global__ void fof_cuda_render_kernel1(
    torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> v,
    int res, int* cnt_pre, float* buffer, int* direction)
{
    //batch index
    const int n = blockIdx.y;
    // column index
    const int c = blockIdx.x * blockDim.x + threadIdx.x;

    if (c >= v.size(1)) return;

    auto p1 = v[n][c][0];
    auto p2 = v[n][c][1];
    auto p3 = v[n][c][2];

    int iMax = floorf(max3f(p1[0],p2[0],p3[0])); iMax = min(iMax+1, res);
    int jMax = floorf(max3f(p1[1],p2[1],p3[1])); jMax = min(jMax+1, res); 
    int iMin =  ceilf(min3f(p1[0],p2[0],p3[0])); iMin = max(iMin, 0);     
    int jMin =  ceilf(min3f(p1[1],p2[1],p3[1])); jMin = max(jMin, 0);     
    
    for (int j=jMin;j<jMax;j++)
    for (int i=iMin;i<iMax;i++)
    {
        float w3 = (p2[0]-p1[0])*(j-p1[1]) - (p2[1]-p1[1])*(i-p1[0]);
        float w1 = (p3[0]-p2[0])*(j-p2[1]) - (p3[1]-p2[1])*(i-p2[0]);
        float w2 = (p1[0]-p3[0])*(j-p3[1]) - (p1[1]-p3[1])*(i-p3[0]);
        float ss = w1+w2+w3;
        if (ss==0) continue;
        if ((w1>=0 && w2>=0 && w3>=0) || (w1<=0 && w2<=0 && w3<=0))
        {
            int tmp = atomicAdd(&cnt_pre[n*res*res+j*res+i], 1);
            buffer[tmp] = (w1*p1[2]+w2*p2[2]+w3*p3[2])/ss;
            direction[tmp] = ss>0?0:1;
        }
    }
}
#接收光栅化后的深度点 buffer 和像素索引，然后计算每个像素的深度直方图。
__global__ void fof_cuda_histogram_kernel(
    torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> fof, // Output FOF (now histogram)
    int* ind, // Start index for each pixel's depth points in buffer
    int* pix, // Flattened pixel index
    float* buffer, // Raw depth values for all pixels
    int cnum, // Number of valid pixels (pixels with at least one depth point)
    int num_bins, // Number of depth bins (FOF channels)
    float min_depth, // Minimum depth for binning
    float max_depth, // Maximum depth for binning
    int res) // Resolution
{
    // Each thread processes one valid pixel
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= cnum) return; // Ensure thread ID is within bounds of valid pixels

    // Get pixel coordinates (n, h, w) from flattened pix[id]
    int tmp_pix_id = pix[id];
    int w = tmp_pix_id % res;
    tmp_pix_id = tmp_pix_id / res;
    int h = tmp_pix_id % res;
    int n = tmp_pix_id / res;

    // Get start and end indices of depth points for this pixel
    int start = ind[id];
    int end = ind[id+1];

    // Ensure valid depth range for binning
    float depth_range = max_depth - min_depth;
    if (depth_range <= 0) return; // Avoid division by zero

    // Iterate through all depth points for this pixel
    for (int i = start; i < end; ++i) {
        float d_tmp = buffer[i];

        // Normalize depth to [0, 1] relative to min/max depth
        float normalized_depth = (d_tmp - min_depth) / depth_range;

        // Calculate bin index
        // Clamp to [0, num_bins-1] to handle depths slightly out of range
        int bin_idx = (int)floorf(normalized_depth * num_bins);
        bin_idx = max(0, min(bin_idx, num_bins - 1));

        // Atomic increment the count for the corresponding bin
        // fof tensor access: fof[batch][height][width][bin_channel]
        atomicAdd(&fof[n][h][w][bin_idx], 1.0f); // Count occurrences in this bin
        // Optional: If you want to store sum of depths in bin (then normalize later):
        // atomicAdd(&fof[n][h][w][bin_idx], d_tmp);
    }
}

__global__ void fof_normal_cuda_render_kernel1(
    torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> v,
    torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> vn,
    float* norm_F, float* norm_B,
    float* depth_F, float* depth_B, int res)
{
    //batch index
    const int n = blockIdx.y;
    // column index
    const int c = blockIdx.x * blockDim.x + threadIdx.x;

    if (c >= v.size(1)) return;

    auto p1 = v[n][c][0];
    auto p2 = v[n][c][1];
    auto p3 = v[n][c][2];

    auto n1 = vn[n][c][0];
    auto n2 = vn[n][c][1];
    auto n3 = vn[n][c][2];

    int iMax = floorf(max3f(p1[0],p2[0],p3[0])); iMax = min(iMax+1, res);
    int jMax = floorf(max3f(p1[1],p2[1],p3[1])); jMax = min(jMax+1, res); 
    int iMin =  ceilf(min3f(p1[0],p2[0],p3[0])); iMin = max(iMin, 0);     
    int jMin =  ceilf(min3f(p1[1],p2[1],p3[1])); jMin = max(jMin, 0);     
    
    for (int j=jMin;j<jMax;j++)
    for (int i=iMin;i<iMax;i++)
    {
        float w3 = (p2[0]-p1[0])*(j-p1[1]) - (p2[1]-p1[1])*(i-p1[0]);
        float w1 = (p3[0]-p2[0])*(j-p2[1]) - (p3[1]-p2[1])*(i-p2[0]);
        float w2 = (p1[0]-p3[0])*(j-p3[1]) - (p1[1]-p3[1])*(i-p3[0]);
        float ss = w1+w2+w3;
        if (ss==0) continue;
        if ((w1>=0 && w2>=0 && w3>=0) || (w1<=0 && w2<=0 && w3<=0))
        {
            float tmp_depth = (w1*p1[2]+w2*p2[2]+w3*p3[2])/ss;
            
            if (tmp_depth == depth_F[n*res*res+j*res+i])
            for (int t=0;t<3;t++)
            norm_F[n*res*res*3 + j*res*3 + i*3 + t] = (w1*(n1[t]*1024)+w2*(n2[t]*1024)+w3*(n3[t]*1024))/ss;

            if (tmp_depth == depth_B[n*res*res+j*res+i])
            for (int t=0;t<3;t++)
            norm_B[n*res*res*3 + j*res*3 + i*3 + t] = (w1*(n1[t]*1024)+w2*(n2[t]*1024)+w3*(n3[t]*1024))/ss;
        }
    }
}


__global__ void compact(int* cnt, int* cnt_pre, int* ccnt_pre, int* ind, int* pix, int pnum)
{
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= pnum) return;
    if (cnt[id] == 0) return;
    ind[ccnt_pre[id]] = cnt_pre[id];
    pix[ccnt_pre[id]] = id; 
}


// lib/fof/fof_cuda.cu

torch::Tensor fof_cuda_dynamic(torch::Tensor v, int num_bins, int res, float min_depth, float max_depth) // Add min_depth, max_depth
{
    cudaSetDevice(v.device().index());
    // FOF output tensor now represents depth histogram with num_bins channels
    auto fof = torch::zeros({v.size(0), res, res, num_bins}, // num is now num_bins
                            torch::TensorOptions()
                                .dtype(torch::kFloat32)
                                .device(v.device().type(), v.device().index())
                                .requires_grad(false));

    // pixels occ (cnt, cnt_pre, ccnt, ccnt_pre remain the same for initial rasterization counts)
    int pnum = v.size(0)*res*res;
    int* cnt; cudaMalloc(&cnt, sizeof(int)*pnum);
    cudaMemset(cnt, 0, sizeof(int)*pnum);
    int* cnt_pre; cudaMalloc(&cnt_pre, sizeof(int)*pnum);
    // pixels num
    int* ccnt; cudaMalloc(&ccnt, sizeof(int)*pnum);
    cudaMemset(ccnt, 0, sizeof(int)*pnum);
    int* ccnt_pre; cudaMalloc(&ccnt_pre, sizeof(int)*pnum);

    const int threads = 1024;
    const dim3 blocks((v.size(1) + threads - 1) / threads, v.size(0));

    // Step 1: Initial rasterization (remains the same)
    fof_cuda_render_kernel0<<<blocks, threads>>>(
        v.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
        res, cnt, ccnt
    );

    int inum, cnum; // inum is total depth points, cnum is total valid pixels
    {
        void* d_temp_storage = NULL;
        size_t temp_storage_bytes = 0;
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, cnt, cnt_pre, pnum);
        cudaMalloc(&d_temp_storage, temp_storage_bytes);
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, cnt, cnt_pre, pnum);
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, ccnt, ccnt_pre, pnum);
        cudaFree(d_temp_storage);

        int tmp0, tmp1;
        cudaMemcpy(&tmp0, cnt+pnum-1, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&tmp1, cnt_pre+pnum-1, sizeof(int), cudaMemcpyDeviceToHost);
        inum = tmp0+tmp1; // Total number of depth points across all pixels
        cudaMemcpy(&tmp0, ccnt+pnum-1, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&tmp1, ccnt_pre+pnum-1, sizeof(int), cudaMemcpyDeviceToHost);
        cnum = tmp0+tmp1; // Total number of pixels that were hit by triangles
    }
    if (inum==0 || cnum==0)
        return fof.permute({0,3,1,2});

    float* buffer; cudaMalloc(&buffer, sizeof(float)*inum); // To store all depth values
    int* direction; cudaMalloc(&direction, sizeof(int)*inum); // To store directions (might not be needed for histogram, but kept for simplicity of buffer filling)

    cudaMemcpy(ccnt, cnt_pre, sizeof(int)*pnum, cudaMemcpyDeviceToDevice); // Re-use ccnt to pass starting indices for buffer filling

    // Step 2: Collect all depth values into 'buffer' (fof_cuda_render_kernel1 remains mostly the same)
    fof_cuda_render_kernel1<<<blocks, threads>>>(
        v.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
        res, ccnt, buffer, direction // direction might not be used later but is filled here
    );

    int* ind; cudaMalloc(&ind, sizeof(int)*(cnum+1)); // To store start/end indices for each valid pixel's depths
    cudaMemcpy(&ind[cnum], &inum, sizeof(int), cudaMemcpyHostToDevice); // Set last element of ind to total depth points
    int* pix; cudaMalloc(&pix, sizeof(int)*cnum); // To store flattened pixel indices
    
    // Step 3: Compact pixel info (remains the same)
    compact<<<(pnum+1023)/1024, 1024>>>(cnt, cnt_pre, ccnt_pre, ind, pix, pnum);

    // *******************************************************************
    // IMPORTANT: Remove or comment out the automata call
    // automata<<<(cnum+1023)/1024, 1024>>>(ind, buffer, direction, cnum);
    // *******************************************************************

    // *******************************************************************
    // IMPORTANT: Remove or comment out the intergral call
    // long long tmp_integral = num_bins; // num is now num_bins
    // tmp_integral = tmp_integral*cnum;
    // intergral<<<(tmp_integral+1023)/1024, 1024>>>(
    //     fof.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
    //     ind, pix, buffer, cnum, num_bins, res, PI
    // );
    // *******************************************************************

    // Step 4: Call the new histogram kernel
    // Pass ind, pix, buffer (containing all depths), cnum, num_bins, min_depth, max_depth, res
    fof_cuda_histogram_kernel<<<(cnum+1023)/1024, 1024>>>( // Each block/thread processes one valid pixel
        fof.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
        ind, pix, buffer, cnum, num_bins, min_depth, max_depth, res
    );


    // Free allocated memory (remains the same)
    cudaFree(ind);
    cudaFree(pix);
    cudaFree(buffer);
    cudaFree(direction);
    cudaFree(cnt);
    cudaFree(cnt_pre);
    cudaFree(ccnt);
    cudaFree(ccnt_pre);

    return fof.permute({0,3,1,2});
}


// lib/fof/fof_cuda.cu

torch::Tensor fof_cuda_static(
    torch::Tensor v, int num_bins, int res, int pre_size, // num is now num_bins
    torch::Tensor pix_cnt,
    torch::Tensor int_cnt,
    torch::Tensor pix_pre,
    torch::Tensor int_pre,
    torch::Tensor pix,
    torch::Tensor ind,
    torch::Tensor pre_tmp, // This is for cub::DeviceScan temp storage
    torch::Tensor int_bbb, // This buffer now holds all raw depth values (was 'buffer' in dynamic)
    torch::Tensor int_ddd, // This buffer now holds all raw direction values (was 'direction' in dynamic)
    float min_depth, float max_depth // Add min_depth, max_depth
)
{
    cudaSetDevice(v.device().index());
    auto fof = torch::zeros({v.size(0), res, res, num_bins}, // num is now num_bins
                            torch::TensorOptions()
                                .dtype(torch::kFloat32)
                                .device(v.device().type(), v.device().index())
                                .requires_grad(false));
    int pnum = v.size(0)*res*res;
    cudaMemset(pix_cnt.data_ptr<int>(), 0, sizeof(int)*pnum);
    cudaMemset(int_cnt.data_ptr<int>(), 0, sizeof(int)*pnum);


    const int threads = 1024;
    const dim3 blocks((v.size(1) + threads - 1) / threads, v.size(0));
    
    // Step 1: Initial rasterization
    fof_cuda_render_kernel0<<<blocks, threads>>>(
        v.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
        res, int_cnt.data_ptr<int>(), pix_cnt.data_ptr<int>()
    );

    // Prefix sum for counts (remains the same)
    void* tmp_ptr = (void*) pre_tmp.data_ptr<unsigned char>();
    size_t tmp_size = pre_size; // pre_size is the size of temp_storage_bytes from cub
    cub::DeviceScan::ExclusiveSum(tmp_ptr, tmp_size, pix_cnt.data_ptr<int>(), pix_pre.data_ptr<int>(), pnum);
    cub::DeviceScan::ExclusiveSum(tmp_ptr, tmp_size, int_cnt.data_ptr<int>(), int_pre.data_ptr<int>(), pnum);
    int inum = int_cnt[pnum-1].item<int>() + int_pre[pnum-1].item<int>();
    int cnum = pix_cnt[pnum-1].item<int>() + pix_pre[pnum-1].item<int>();

    if (inum==0 || cnum==0)
        return fof.permute({0,3,1,2});

    cudaMemset(pix_cnt.data_ptr<int>(), 0, sizeof(int)*pnum); // Reset for next kernel
    // Step 2: Collect all depth values into int_bbb and int_ddd (used as buffer and direction)
    fof_cuda_render_kernel1<<<blocks, threads>>>(
        v.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
        res, pix_cnt.data_ptr<int>(), int_bbb.data_ptr<float>(), // int_bbb is now our depth buffer
        int_ddd.data_ptr<int>() // int_ddd is now our direction buffer
    );

    // Step 3: Compact pixel info
    cudaMemcpy(&ind.data_ptr<int>()[cnum], &inum, sizeof(int), cudaMemcpyHostToDevice);
    compact<<<(pnum+1023)/1024, 1024>>>(int_cnt.data_ptr<int>(), int_pre.data_ptr<int>(), pix_pre.data_ptr<int>(),
                                         ind.data_ptr<int>(), pix.data_ptr<int>(), pnum);
    
    // *******************************************************************
    // IMPORTANT: Remove or comment out the automata call
    // automata<<<(cnum+1023)/1024, 1024>>>(ind.data_ptr<int>(), int_bbb.data_ptr<float>(), int_ddd.data_ptr<int>(), cnum);
    // *******************************************************************

    // *******************************************************************
    // IMPORTANT: Remove or comment out the intergral call
    // intergral<<<(num_bins*cnum+1023)/1024, 1024>>>( // num is now num_bins
    //     fof.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
    //     ind.data_ptr<int>(), pix.data_ptr<int>(),
    //     int_bbb.data_ptr<float>(), cnum, num_bins, res, PI
    // );
    // *******************************************************************

    // Step 4: Call the new histogram kernel
    fof_cuda_histogram_kernel<<<(cnum+1023)/1024, 1024>>>(
        fof.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
        ind.data_ptr<int>(), pix.data_ptr<int>(),
        int_bbb.data_ptr<float>(), cnum, num_bins, min_depth, max_depth, res
    );

    return fof.permute({0,3,1,2});
}

std::vector<torch::Tensor> fof_normal_cuda_static(
    torch::Tensor v, torch::Tensor vn,
    int num, int res, int pre_size,
    torch::Tensor pix_cnt,
    torch::Tensor int_cnt,
    torch::Tensor pix_pre,
    torch::Tensor int_pre,
    torch::Tensor pix,
    torch::Tensor ind,
    torch::Tensor pre_tmp,
    torch::Tensor int_bbb,
    torch::Tensor int_ddd
)
{
    cudaSetDevice(v.device().index());
    auto fof = torch::zeros({v.size(0), res, res, num},
                            torch::TensorOptions()
                                .dtype(torch::kFloat32)
                                .device(v.device().type(), v.device().index())
                                .requires_grad(false));
    auto norm_F = torch::zeros({v.size(0), res, res, 3},
                            torch::TensorOptions()
                                .dtype(torch::kFloat32)
                                .device(v.device().type(), v.device().index())
                                .requires_grad(false));
    auto norm_B = torch::zeros({v.size(0), res, res, 3},
                            torch::TensorOptions()
                                .dtype(torch::kFloat32)
                                .device(v.device().type(), v.device().index())
                                .requires_grad(false));
    auto depth_F = torch::ones({v.size(0), res, res},
                            torch::TensorOptions()
                                .dtype(torch::kFloat32)
                                .device(v.device().type(), v.device().index())
                                .requires_grad(false)) * -1;   // mul -1 here!!!!!!
    auto depth_B = torch::ones({v.size(0), res, res},
                            torch::TensorOptions()
                                .dtype(torch::kFloat32)
                                .device(v.device().type(), v.device().index())
                                .requires_grad(false));
    
    int pnum = v.size(0)*res*res;
    cudaMemset(pix_cnt.data_ptr<int>(), 0, sizeof(int)*pnum);
    cudaMemset(int_cnt.data_ptr<int>(), 0, sizeof(int)*pnum);


    const int threads = 1024;
    const dim3 blocks((v.size(1) + threads - 1) / threads, v.size(0));
    fof_normal_cuda_render_kernel0<<<blocks, threads>>>(
        v.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
        depth_F.data_ptr<float>(), depth_B.data_ptr<float>(),
        res, int_cnt.data_ptr<int>(), pix_cnt.data_ptr<int>()
    );

    
    void* tmp_ptr = (void*) pre_tmp.data_ptr<unsigned char>();
    size_t tmp_size = pre_size;
    cub::DeviceScan::ExclusiveSum(tmp_ptr, tmp_size, pix_cnt.data_ptr<int>(), pix_pre.data_ptr<int>(), pnum);
    cub::DeviceScan::ExclusiveSum(tmp_ptr, tmp_size, int_cnt.data_ptr<int>(), int_pre.data_ptr<int>(), pnum);
    int inum = int_cnt[pnum-1].item<int>() + int_pre[pnum-1].item<int>();
    int cnum = pix_cnt[pnum-1].item<int>() + pix_pre[pnum-1].item<int>();

    if (inum==0 || cnum==0)
        return {fof.permute({0,3,1,2}), depth_F, depth_B, norm_F.permute({0,3,1,2}), norm_B.permute({0,3,1,2})};


    cudaMemcpy(pix_cnt.data_ptr<int>(), int_pre.data_ptr<int>(), sizeof(int)*pnum, cudaMemcpyDeviceToDevice);

    fof_cuda_render_kernel1<<<blocks, threads>>>(
        v.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
        res, pix_cnt.data_ptr<int>(), int_bbb.data_ptr<float>(),
        int_ddd.data_ptr<int>()
    );
    fof_normal_cuda_render_kernel1<<<blocks, threads>>>(
        v.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
        vn.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
        norm_F.data_ptr<float>(), norm_B.data_ptr<float>(),
        depth_F.data_ptr<float>(), depth_B.data_ptr<float>(), res
    );


    cudaMemcpy(&ind.data_ptr<int>()[cnum], &inum, sizeof(int), cudaMemcpyHostToDevice);
    compact<<<(pnum+1023)/1024, 1024>>>(int_cnt.data_ptr<int>(), int_pre.data_ptr<int>(), pix_pre.data_ptr<int>(),
                                        ind.data_ptr<int>(), pix.data_ptr<int>(), pnum);
    automata<<<(cnum+1023)/1024, 1024>>>(ind.data_ptr<int>(), int_bbb.data_ptr<float>(), int_ddd.data_ptr<int>(), cnum);

    intergral<<<(num*cnum+1023)/1024, 1024>>>(
        fof.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
        ind.data_ptr<int>(), pix.data_ptr<int>(),
        int_bbb.data_ptr<float>(), cnum, num, res, PI
    );

    return {fof.permute({0,3,1,2}), depth_F, depth_B, norm_F.permute({0,3,1,2}), norm_B.permute({0,3,1,2})};
}
