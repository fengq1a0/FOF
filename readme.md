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
cd lib/mc
sh compile.sh
cd ../..
```
Check train.py and test.py, have fun with this repo!
## Checkpoints
Uploading...
