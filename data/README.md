## Overview
Basically, our goal is to generate some image-fof pairs. It's seperated into four parts: 

0. to_npz : convert meshes into some `.npz` files (obj is really not a good format) 
1. normalization : resize all meshes to the same scale
2. image : render images 
3. gen_mpi : generate mpis (for fof)

Get the sample from https://github.com/ytrock/THuman2.0-Dataset/tree/main/data_sample/0525, and put it into the `obj` directory.

## 0. to_npz
Run the following codes:
```bash
cd apps
python to_npz.py
```
Each mesh would be like:
+ mesh.npz : {"v" : xxx, "f" : xxx}
+ uv.npz : {"vt" : xxx, "ft" : xxx}
+ texture.jpg

## 1. normalize
```bash
cd ../lib/cpp
sh compile.sh
cd ../../apps
python normalization.py
```
## 2. image
Codes in this part are borrowed from [PIFu](https://github.com/shunsukesaito/PIFu/)! And I recommend you to render images with your own renderer.

> for training and data generation
>
> - [trimesh](https://trimsh.org/) with [pyembree](https://github.com/scopatz/pyembree)
> - [pyexr](https://github.com/tvogels/pyexr)
> - PyOpenGL
> - freeglut (use `sudo apt-get install freeglut3-dev` for ubuntu users)
> - (optional) egl related packages for rendering with headless machines. (use `apt install libgl1-mesa-dri libegl1-mesa libgbm1` for ubuntu users)
> 
> Warning: I found that outdated NVIDIA drivers may cause errors with EGL. If you want to try out the EGL version, please update your NVIDIA driver to the latest!!

1. prt : It's very time-consuming!
```
python prt_util.py
```
2. render : This code has a memory leak problem. So, I do not render meshes with a single python script.
```
python render_data.py -i ../raw/0525 -o ../img/0525 -e -s 512
```
You can get more details in [PIFu](https://github.com/shunsukesaito/PIFu/).
## 3. mpis
```
python gen_mpi.py
```
You can get FOF with the function `tri_occ` in `../lib/dataset/ceof_generator.py`.

## Acknowledge
The model is borrowed from https://github.com/ytrock/THuman2.0-Dataset. And you can get THuman2.0 from this link.