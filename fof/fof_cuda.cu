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

__global__ void automata(int* ind, float* buffer, int* direction, int cnum)
{
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= cnum) return;

    // sort
    int start = ind[id]; 
    int end = ind[id+1];
    for (int i=start+1; i<end; i++)
    {
        float tmp_buffer = buffer[i];
        int tmp_direction = direction[i];
        int pre = i-1;
        while (pre>=start && compare(tmp_buffer, tmp_direction, buffer[pre], direction[pre]))
        {
            buffer[pre+1] = buffer[pre];
            direction[pre+1] = direction[pre];
            pre--;
        }
        buffer[pre+1] = tmp_buffer;
        direction[pre+1] = tmp_direction;
    }

    // automata
    int state = 0;
    int pre = 0;
    for (int i=start; i<end; i++)
    if (state == 0)
    {
        if (direction[i]==0){
            state = 1;
            buffer[start+pre] = buffer[i];
        }
    }
    else if (state == 1)
    {
        if (direction[i]==1){
            state = 2;
            buffer[start+pre+1] = buffer[i];
        }
    }
    else if (state == 2) 
    {
        if (direction[i]==1){
            buffer[start+pre+1] = buffer[i];
        }else{
            state = 1;
            pre = pre+2;
            buffer[start+pre] = buffer[i];
        }
    }

    if (state==2) pre = pre+2;
    for (int i=start+pre; i<end; i++) buffer[i] = 1024;
}

__global__ void intergral(
    torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> fof,
    int* ind, int* pix, float* buffer, int cnum, int num, int res, float PI)
{
    long long tid = blockIdx.x;
    tid = tid * blockDim.x + threadIdx.x;
    int id = tid / num;
    if (id >= cnum) return;
    int c = tid % num;
    
    int tmp = pix[id];
    int w = tmp % res;
    tmp = tmp/res;
    int h = tmp % res;
    int n = tmp/res;

    int start = ind[id]; 
    int end = ind[id+1];
    if (c==0){
        for (int i=start; i<end; i=i+2)
        {
            if (buffer[i]==1024) break;
            fof[n][h][w][0] += buffer[i+1]-buffer[i];
        }
    }else{
        for (int i=start; i<end; i=i+2)
        {
            if (buffer[i]==1024) break;
            float t2 = buffer[i+1]+1;
            float t1 = buffer[i]+1;
            fof[n][h][w][c] += sin(t2*0.5*c*PI)-sin(t1*0.5*c*PI);
        }
        fof[n][h][w][c] /= 0.5*c*PI;
    }
}
} // namespace

torch::Tensor fof_cuda_dynamic(torch::Tensor v, int num, int res)
{
    cudaSetDevice(v.device().index());
    auto fof = torch::zeros({v.size(0), res, res, num},
                            torch::TensorOptions()
                                .dtype(torch::kFloat32)
                                .device(v.device().type(), v.device().index())
                                .requires_grad(false));
    
    // pixels occ
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
    fof_cuda_render_kernel0<<<blocks, threads>>>(
        v.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
        res, cnt, ccnt
    );

    int inum, cnum;
    {
        void     *d_temp_storage = NULL;
        size_t   temp_storage_bytes = 0;
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, cnt, cnt_pre, pnum);
        cudaMalloc(&d_temp_storage, temp_storage_bytes);
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, cnt, cnt_pre, pnum);
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, ccnt, ccnt_pre, pnum);
        cudaFree(d_temp_storage);

        int tmp0, tmp1;
        cudaMemcpy(&tmp0, cnt+pnum-1, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&tmp1, cnt_pre+pnum-1, sizeof(int), cudaMemcpyDeviceToHost);
        inum = tmp0+tmp1;
        cudaMemcpy(&tmp0, ccnt+pnum-1, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&tmp1, ccnt_pre+pnum-1, sizeof(int), cudaMemcpyDeviceToHost);
        cnum = tmp0+tmp1;
    }
    if (inum==0 || cnum==0)
        return fof.permute({0,3,1,2});

    float* buffer; cudaMalloc(&buffer, sizeof(float)*inum);
    int* direction; cudaMalloc(&direction, sizeof(int)*inum);
    cudaMemcpy(ccnt, cnt_pre, sizeof(int)*pnum, cudaMemcpyDeviceToDevice);
    fof_cuda_render_kernel1<<<blocks, threads>>>(
        v.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
        res, ccnt, buffer, direction
    );

    int* ind; cudaMalloc(&ind, sizeof(int)*(cnum+1)); 
    cudaMemcpy(&ind[cnum], &inum, sizeof(int), cudaMemcpyHostToDevice);
    int* pix; cudaMalloc(&pix, sizeof(int)*cnum);
    compact<<<(pnum+1023)/1024, 1024>>>(cnt, cnt_pre, ccnt_pre, ind, pix, pnum);

    automata<<<(cnum+1023)/1024, 1024>>>(ind, buffer, direction, cnum);

    long long tmp = num;
    tmp = tmp*cnum;
    intergral<<<(tmp+1023)/1024, 1024>>>(
        fof.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
        ind, pix, buffer, cnum, num, res, PI
    );

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


int get_buffer_size(int pnum)
{
    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    int* tmp = NULL;
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, tmp, tmp, pnum);
    return temp_storage_bytes;
}

torch::Tensor fof_cuda_static(
    torch::Tensor v, int num, int res, int pre_size,
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
    int pnum = v.size(0)*res*res;
    cudaMemset(pix_cnt.data_ptr<int>(), 0, sizeof(int)*pnum);
    cudaMemset(int_cnt.data_ptr<int>(), 0, sizeof(int)*pnum);


    const int threads = 1024;
    const dim3 blocks((v.size(1) + threads - 1) / threads, v.size(0));
    fof_cuda_render_kernel0<<<blocks, threads>>>(
        v.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
        res, int_cnt.data_ptr<int>(), pix_cnt.data_ptr<int>()
    );

    
    void* tmp_ptr = (void*) pre_tmp.data_ptr<unsigned char>();
    size_t tmp_size = pre_size;
    cub::DeviceScan::ExclusiveSum(tmp_ptr, tmp_size, pix_cnt.data_ptr<int>(), pix_pre.data_ptr<int>(), pnum);
    cub::DeviceScan::ExclusiveSum(tmp_ptr, tmp_size, int_cnt.data_ptr<int>(), int_pre.data_ptr<int>(), pnum);
    int inum = int_cnt[pnum-1].item<int>() + int_pre[pnum-1].item<int>();
    int cnum = pix_cnt[pnum-1].item<int>() + pix_pre[pnum-1].item<int>();

    if (inum==0 || cnum==0)
        return fof.permute({0,3,1,2});


    cudaMemcpy(pix_cnt.data_ptr<int>(), int_pre.data_ptr<int>(), sizeof(int)*pnum, cudaMemcpyDeviceToDevice);
    fof_cuda_render_kernel1<<<blocks, threads>>>(
        v.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
        res, pix_cnt.data_ptr<int>(), int_bbb.data_ptr<float>(),
        int_ddd.data_ptr<int>()
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