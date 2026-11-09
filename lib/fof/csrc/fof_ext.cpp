#include <torch/extension.h>

#include <vector>

namespace fof {
std::vector<at::Tensor> rasterize(const at::Tensor& vxy, const at::Tensor& vz,
                                  const at::Tensor& faces, int64_t H, int64_t W);
at::Tensor fof_integrate(const at::Tensor& pix_ptr, const at::Tensor& keys, int64_t B, int64_t H,
                         int64_t W, int64_t C, double z_min, double z_max, int64_t rule);
std::vector<at::Tensor> front_back(const at::Tensor& pix_ptr, const at::Tensor& keys,
                                   const at::Tensor& vxy, const at::Tensor& faces, int64_t H,
                                   int64_t W);
std::vector<at::Tensor> marching_cubes(const at::Tensor& vol, double level, int64_t pad, double pad_value,
                                       const at::Tensor& amb,
                                       const at::Tensor& base, const at::Tensor& ntri, const at::Tensor& tris,
                                       const at::Tensor& ncen, const at::Tensor& cloop, const at::Tensor& faces,
                                       const at::Tensor& edges, int64_t max_centres);
}  // namespace fof

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "Exact multi-layer mesh rasterization and Fourier Occupancy Field integration (CUDA)";
  m.attr("SUBPIXEL_BITS") = 8;
  m.def("rasterize", &fof::rasterize,
        "Rasterize all layers. Returns (pix_ptr[int32, B*H*W+1], keys[int64, N]) sorted per pixel.",
        py::arg("vxy"), py::arg("vz"), py::arg("faces"), py::arg("H"), py::arg("W"));
  m.def("fof_integrate", &fof::fof_integrate,
        "Integrate the cosine basis over the occupied intervals -> [B, C, H, W].",
        py::arg("pix_ptr"), py::arg("keys"), py::arg("B"), py::arg("H"), py::arg("W"),
        py::arg("C"), py::arg("z_min"), py::arg("z_max"), py::arg("rule"));
  m.def("front_back", &fof::front_back,
        "Front-most / back-most fragment per pixel -> (face, depth, barycentrics).",
        py::arg("pix_ptr"), py::arg("keys"), py::arg("vxy"), py::arg("faces"), py::arg("H"),
        py::arg("W"));
  m.def("marching_cubes", &fof::marching_cubes,
        "Crack-free marching cubes (optional virtual border) -> (triangle vertex ids [F, 3], centre records [C, 13]).",
        py::arg("vol"), py::arg("level"), py::arg("pad"), py::arg("pad_value"), py::arg("amb"), py::arg("base"), py::arg("ntri"), py::arg("tris"),
        py::arg("ncen"), py::arg("cloop"), py::arg("faces"), py::arg("edges"), py::arg("max_centres"));
}
