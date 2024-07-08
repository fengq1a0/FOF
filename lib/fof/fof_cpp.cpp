#include <torch/extension.h>
#include <vector>

torch::Tensor fof_cuda_dynamic(torch::Tensor v, int num, int res);
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
);
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
);
int get_buffer_size(int pnum);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("static_render", &fof_cuda_static, "static");
    m.def("dynamic_render", &fof_cuda_dynamic, "dynamic");
    m.def("fof_normal_static_render", &fof_normal_cuda_static, "static");
    m.def("get_buffer_size", &get_buffer_size, "get_buffer_size");
}
