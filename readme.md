# FOF-X: Towards Real-time Detailed Human Reconstruction from a Single Image
[website](https://cic.tju.edu.cn/faculty/likun/projects/FOFX/index.html) | [arxiv](https://arxiv.org/pdf/2412.05961)

# FOF: Learning Fourier Occupancy Field for Monocular Real-time Human Reconstruction
[website](https://cic.tju.edu.cn/faculty/likun/projects/FOF/index.html) | [paper](https://cic.tju.edu.cn/faculty/likun/projects/FOF/imgs/FOF_paper.pdf)

## Updates
- 2026-11: the FOF toolkit (`lib/fof`) is rewritten: exact, crack-free mesh->FOF rasterization with a more robust
  fill rule, GPU marching cubes, a real-time FOF-X (lossless) decoder and orientation repair for the training data.
  The networks and the checkpoint format are unchanged. See [the FOF toolkit](#the-fof-toolkit-libfof).

## Environment
To run this code, follow the following instructions.
```
conda create -n fof python=3.10
conda activate fof

# visit https://pytorch.org/
# install latest pytorch
# for example:
# conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

pip install tqdm
pip install opencv-python
pip install open3d
pip install scikit-image
pip install ninja
pip install tensorboard      # train.py
pip install pytest trimesh   # tests
```

The CUDA extension (`lib/fof/csrc`: the mesh->FOF rasterizer and the GPU marching cubes) is compiled automatically
the first time it is used (this needs `nvcc` matching your PyTorch CUDA version, and `ninja`). Alternatively build
it ahead of time:
```
cd lib/fof
sh compile.sh      # python setup.py build_ext --inplace
cd ../..
```
Nothing else needs to be compiled (the Cython marching cubes and the C++ solver of the original code are gone).
Without the extension (e.g. on CPU) the encoder runs a pure PyTorch implementation of the same algorithm (slower,
bit-identical crossings), and `fof_to_mesh` uses scikit-image or the PyTorch version of the GPU marching cubes.

Check train.py and test.py, have fun with this repo!

## The FOF toolkit (`lib/fof`)
```python
from lib.fof import mesh_to_fof, render_fof, fof_to_mesh, orient_outward

# meshes -> FOF: verts [B, V, 3] + faces [F, 3] or [B, F, 3]; also a triangle soup [B, F, 3, 3] or lists of meshes
fof = mesh_to_fof(verts, faces, num_coeffs=128, resolution=512)            # [B, 128, 512, 512]
out = render_fof(verts, faces, normals, num_coeffs=16, resolution=512)     # FOF + front/back depth & normal maps
v, f = fof_to_mesh(fof[0], depth_res=512)                                  # marching cubes (GPU for CUDA tensors)
v, f = fof_to_mesh(fof[0], depth_res=512, lossless=True)                   # FOF-X decoder, real time on the GPU
faces_np = orient_outward(verts_np, faces_np)                              # offline repair of a ground-truth mesh
```
Conventions: normalized coordinates in [-1, 1], x right, y up, z towards the camera (orthographic).
Pixel (h, w) has its centre at `x = (w + 0.5) * 2 / W - 1`, `y = 1 - (h + 0.5) * 2 / H`. With `t = (z + 1) / 2`,
`a_0 = 2 * int o(t) dt` and `a_c = 2 * int o(t) cos(c*pi*t) dt`, so `o(t) = a_0 / 2 + sum_c a_c cos(c*pi*t)`.
Faces must be counter-clockwise seen from outside (`orient_outward` repairs meshes that are not).

**Mesh -> FOF.** Every ray/surface crossing of every pixel is kept (no depth peeling, no layer limit). Vertices
are snapped to 1/256 pixel and coverage uses exact integer edge functions with a top-left rule, so pixel centres
on shared edges or vertices are counted exactly once (the cracks of the original kernel are gone). The fill rule
(`fill=`) turns the sorted crossings of a ray into occupied intervals. Measured on a watertight human solid with
injected defects, 4 viewing directions (over-/under-filled volume, % of the true volume):

| defects (each with interpenetrating parts) | `hybrid` (default) | `robust` (original) | `winding` | `parity` |
|---|---|---|---|---|
| holes | 0.38 / 5.13 | 0.35 / 5.35 | 0.19 / 7.88 | 0.42 / 23.95 |
| flipped patches | 0.20 / 0.16 | 0.19 / 2.20 | 0.00 / 3.53 | 0.00 / 18.55 |
| duplicated inner layer | 0.14 / 0.02 | 0.01 / 0.24 | 0.17 / 5.05 | 0.30 / 21.48 |

`hybrid` = winding number clamped at 0 (exact union of interpenetrating parts) + the original rule's merging of
exit runs (duplicated layers) + closing a still-open interval at its last exit + even-odd fallback when nothing
matches (flipped orientation). All rules agree on a single clean, closed surface; nested or interpenetrating
parts are exact with `winding` and `hybrid` only (`parity` makes cavities of them). A single ray cannot always tell
defects apart (crossings `E X X E` are a duplicated layer plus a hole, or a second, flipped part), so no per-ray
rule is right everywhere: `hybrid` errs where a flipped patch lies on a ray that also crosses another part, and
no per-ray rule can repair holes. Both are mesh-level defects, best fixed offline: after `orient_outward`
(consistent winding propagated over manifold edges, then every part with a negative signed volume flipped; see
[data preprocess](#data-preprocess)), `hybrid` was exact on a flipped-patch test where it otherwise had 0.4-1.0%
over- and under-fill. `ray_diagnostics` reports how many rays of a mesh are affected.
Encoding 8 SMPL-X meshes at 512^2 with 128 coefficients takes 1.3 ms with `hybrid` (1.2 ms with `robust`), mostly
spent writing the 1 GiB output. Use `fill="robust"` to reproduce the ground truth of the released checkpoints
(`FOF`, `FOF_Normal`, `render` and `test.py` do).

**FOF -> mesh.** `fof_to_mesh` evaluates the cosine series on the GPU and extracts the surface with a crack-free
GPU marching cubes (`lib/fof/mc.py`): its tables are generated from first principles, ambiguous faces are resolved
with the asymptotic decider computed from the face's own values (neighbouring cells always agree), and loops are
triangulated without face diagonals, so the output is a closed, consistently oriented 2-manifold (except where
samples are exactly equal to the level, which can create zero-area triangles); the CUDA and the PyTorch
implementations produce identical meshes. `lossless=True` is the FOF-X decoder: vertices on depth edges
are kept (exact for each pixel's 1D occupancy) and the in-between vertices minimize the least-squares Laplacian
energy, now solved matrix-free with 25 PCG iterations on the GPU (early stopping matches or beats the exact
minimizer; free components without any depth anchor are left untouched). SMPL-X at 512^3 with 128 coefficients:

| decoder | time | mean distance to GT (voxels) | normal error |
|---|---|---|---|
| GPU marching cubes | 4 ms | 0.050 | 4.8 deg |
| GPU marching cubes + lossless | 11 ms | 0.044 | 4.1 deg |
| old: scikit-image + exact least squares (CPU) | 2.3 s | 0.045 | 4.0 deg |

With `close=True` (default) the kernels read a virtual empty border instead of a padded copy of the volume:
1024^3 decoding takes 21 ms (52 ms lossless) at a 4.8 GiB peak, most of it the float32 volume itself.
`backend="skimage"` keeps the CPU path (scikit-image Lewiner MC). `Recon.decode` in `lib/utils.py` reproduces the
original decoder bit for bit; `Recon.decode_lossless` keeps its algorithm with an exact solve.

Run the tests with `pytest tests/` (CPU; the CUDA backend is tested as well when a GPU is available).

## Checkpoints
You can get pretrained checkpoints [here](https://drive.google.com/drive/folders/1ocS0YND9vtdFN8Z99BoUdPu-ktSUwt5x?usp=sharing).
Put them in ```./ckpt```.  They should be organized as shown below:
```
FOF
├─lib
└─ckpt
    ├─model32.pth
    ├─model48.pth
    ├─netB.pth
    └─netF.pth
```
```model32.pth``` and ```model48.pth``` correspond to HR-Net-32 and HR-Net-48 version, respectively. Both of them are trained on THuman2.0 dataset only.

## Data preprocess
You will need triangle mesh and corresponding SMPL-X mesh to train the model. The texture is not needed, and the meshes don't have to be watertight (which is better)! However, we prefer the meshes are clean, without floating fragments. Check the data preprocess code ```preprocess.py``` for more details. You can see the steps we do in the file.

**Orientation and holes.** The FOF ground truth is read from the orientation of the faces: along each pixel's ray,
the body starts at a face that points away from the camera and ends at a face that points towards it.
`preprocess.py` therefore makes the faces of every scan consistent and outward-facing (`orient_outward` in
`lib/fof/repair.py`, about 0.1 s per 50k faces) and prints how many faces it had to flip; clean meshes are left
unchanged. Open3D's `orient_triangles()` alone is not enough: it only makes neighbouring faces agree (a part that
is inside out stays inside out), and it gives up on the whole mesh at the first non-manifold edge. Holes cannot be
repaired along a ray, so close large ones beforehand (e.g. Poisson reconstruction). To check processed meshes:
```python
import numpy as np, open3d as o3d, torch
from lib.fof import ray_diagnostics

m = o3d.io.read_triangle_mesh(base_out + "100k-180/0000.ply")  # base_out as in preprocess.py
d = ray_diagnostics(torch.tensor(np.asarray(m.vertices), dtype=torch.float32, device="cuda"),
                    torch.tensor(np.asarray(m.triangles), device="cuda"), 512)
print(d["frac_unbalanced"].item())  # share of rays through holes or flipped faces: 0 for closed, outward meshes
print(d["frac_disagree"].item())    # share of rays where the fill rules differ (also nested / overlapping parts)
```

Check the code, you can use the function ```work``` to process your data. If you are using THuman2.0 dataset, you can organize your data as shown below.

Your THuman2.0 mesh or your scanned mesh. (Only xxxx.obj is used)
```
$base_mesh        # base_mesh in preprocess.py
├───0000
|   ├─0000.obj
|   ├─material0.jpeg
|   └─material0.mtl
├───0001
├───0002
...
└───0525
```

Corresponding SMPL-X mesh. (SMPL is also OK, but you will need to modify the dataloader yourself. Just a few lines.)
```
$base_smpl        # base_smpl in preprocess.py
├───0000
|   ├─mesh_smplx.obj
|   └─smplx_param.pkl
├───0001
├───0002
...
└───0525
```

The path you want to store the processed data. Dataloader also read data from it. You don't need to create it. After running ```preprocess.py```, it looks like: 
```
$base_out         # base_out in preprocess.py
├───100k-180
|   ├─0000.ply
|   ├─0001.ply
|   ...
|   └─0525.ply
├───smpl-180
|   ├─0000.ply
|   ├─0001.ply
|   ...
|   └─0525.ply
└───train_th2.txt
```
```train_th2.txt``` lists the meshes used for training. It should look like:
```
0000
0001
0008
...
```
You can check the ```.ply``` meshes in meshlab. Make sure they are well-aligned.

Finally, you can change the default ```path``` and ```name_file``` of ```TrainSet``` in ```lib/dataset_mesh_only.py``` to the right path. Alternatively, you can identify them when using the TrainSet class. Now you can enjoy your training!

## Citation

If you find this code useful for your research, please use the following BibTeX entry.

```
@inproceedings{li2022neurips,
  author = {Qiao Feng and Yebin Liu and Yu-Kun Lai and Jingyu Yang and Kun Li},
  title = {FOF: Learning Fourier Occupancy Field for Monocular Real-time Human Reconstruction},
  booktitle = {NeurIPS},
  year={2022},
}
```
```
@inproceedings{fofx,
  author = {Qiao Feng and Yuanwang Yang and Yebin Liu and Yu-Kun Lai and Jingyu Yang and Kun Li},
  title = {FOF-X: Towards Real-time Detailed Human Reconstruction from a Single Image},
  booktitle = {ArXiv:2412.05961},
  year={2024},
}
```
