# DiffuStereo: High Quality Human Reconstruction via Diffusion-based Stereo Using Sparse Cameras

**News**

* `06/11/2022` The code is released. But it's not complete. I'm still updating it. 


### [Project Page](http://cic.tju.edu.cn/faculty/likun/projects/FOF/index.html) | [Paper](http://cic.tju.edu.cn/faculty/likun/projects/FOF/imgs/FOF_paper.pdf) 



> [FOF: Learning Fourier Occupancy Field for Monocular Real-time Human Reconstruction](http://cic.tju.edu.cn/faculty/likun/projects/FOF/imgs/FOF_paper.pdf)  
> Qiao Feng, Yebin Liu, Yu-Kun Lai, Jingyu Yang, Kun Li
> NeurIPS 2022

We plan to release the training and testing code of FOF in this repository as soon as possible. Any discussions or questions would be welcome!

## Dependencies

To run this code, the following packages need to be installed.

```
numpy
pytorch
cv2
lmdb
numba
skimage
```

## Pretrained model

You can download the pretrained model and put it into the `ckpt/base` directory.

[Download: pretrained model of FOF-base](https://pan.baidu.com/s/17xdfkT6UKtuX5w0nvSK6yw?pwd=89go)

## Notice
The input images should be `.png`s with 512*512 resolution in RGBA format. And the alpha channel is the mask.

## TODO



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
