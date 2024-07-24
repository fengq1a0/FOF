# FOF: Learning Fourier Occupancy Field for Monocular Real-time Human Reconstruction

Releasing...

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
You will need triangle mesh and corresponding SMPL mesh to train the model. Texture is not needed and the meshes don't have to be watertight(watertight is better)! However, we prefer the meshes are clean, without floating fragments. Check the data preprocess code for more details.
