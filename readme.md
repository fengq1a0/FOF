# FOF-X: Towards Real-time Detailed Human Reconstruction from a Single Image
[website](https://cic.tju.edu.cn/faculty/likun/projects/FOFX/index.html) | [arxiv](https://arxiv.org/pdf/2412.05961)

# FOF: Learning Fourier Occupancy Field for Monocular Real-time Human Reconstruction
[website](https://cic.tju.edu.cn/faculty/likun/projects/FOF/index.html) | [paper](https://cic.tju.edu.cn/faculty/likun/projects/FOF/imgs/FOF_paper.pdf)

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
```


You will need the cuda corresponding to your pytorch to compile the tools below. 
```
pip install Cython
pip install pybind11
# 1. fof
cd lib/fof
sh compile.sh
cd ../..

# 2. mc
# You need Eigen to compile this.
cd lib/mc
sh compile.sh
cd ../..
```
Check train.py and test.py, have fun with this repo!
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
You will need triangle mesh and corresponding SMPL-X mesh to train the model. Texture is not needed and the meshes don't have to be watertight (which is better)! However, we prefer the meshes are clean, without floating fragments. Check the data preprocess code ```preprocess.py``` for more details. You can see the seven steps we do in the file. Before run it, you will need to
```
pip install cyminiball
```
Check the code, you can use the function ```work``` to process you data. If you are using THuman2.0 dataset, you can organize your data as shown below.

Your THuman2.0 mesh or your scanned mesh. (Only xxxx.obj is used)
```
$base_mesh        # line 139 in preprocess.py
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
$base_smpl        # line 140 in preprocess.py
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
$base_out         # line 141 in preprocess.py
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
You can check the ```.ply``` meshes in meshlab. Make sure them are well-aligned.

Finally, you can change the ```line 26``` and ```line 27``` in the ```lib\dataset_mesh_only.py``` to the right path. Alternatively, you can identify them when using the TrainSet class. Now you can enjoy your training!

>## Citation
>
>If you find this code useful for your research, please use the following BibTeX entry.
>
>```
>@inproceedings{li2022neurips,
>  author = {Qiao Feng and Yebin Liu and Yu-Kun Lai and Jingyu Yang and Kun Li},
>  title = {FOF: Learning Fourier Occupancy Field for Monocular Real-time Human Reconstruction},
>  booktitle = {NeurIPS},
>  year={2022},
>}
>```
>```
>@inproceedings{fofx,
>  author = {Qiao Feng and Yebin Liu and Yu-Kun Lai and Jingyu Yang and Kun Li},
>  title = {FOF-X: Towards Real-time Detailed Human Reconstruction from a Single Image},
>  booktitle = {ArXiv:2412.05961},
>  year={2024},
>}
>```
