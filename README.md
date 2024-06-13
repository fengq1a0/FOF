# DEPRECATED
This branch is deprecated! Please check out the [main branch](https://github.com/fengq1a0/FOF/), which offers several improvements, efficient implementation, and better performance. 
We have simplified the training process, making it easy to train and reproduce our results. Additionally, FOF-related codes are implemented as a CUDA extension. Use it in your code!

># FOF: Learning Fourier Occupancy Field for Monocular Real-time Human Reconstruction
>It seems that I updated a wrong version checkpoint, which may not work as well as in paper. In fact, the original project has been modified to a new one, and I'm still working on it. It's very struggling to rewrite and roll back to the old version (the version >in this repository). So I'll update the new version code in this repositary as soon as the new paper is completed. By then, you can get a model that is similar to the old version by changing some parameters.
>
>**News**
>* `07/01/2023` the code for `data preprocessing` is updated. 
>* `20/11/2022` The `speed_test.py` is added. `pip install git+https://github.com/tatsy/torchmcubes.git` for gpu marching cubes. We implement a faster mcubes, and I'm still cleaning the code.
>* `06/11/2022` The code is released. But it's not complete. I'm still updating it. 
>
>
>### [Project Page](http://cic.tju.edu.cn/faculty/likun/projects/FOF/index.html) | [Paper](http://cic.tju.edu.cn/faculty/likun/projects/FOF/imgs/FOF_paper.pdf) 
>
>
>
>> [FOF: Learning Fourier Occupancy Field for Monocular Real-time Human Reconstruction](http://cic.tju.edu.cn/faculty/likun/projects/FOF/imgs/FOF_paper.pdf)  
>> Qiao Feng, Yebin Liu, Yu-Kun Lai, Jingyu Yang, Kun Li
>> NeurIPS 2022
>
>We plan to release the training and testing code of FOF in this repository as soon as possible. Any discussions or questions would be welcome!
>
>## Dependencies
>
>To run this code, the following packages need to be installed.
>
>```
>numpy
>pytorch
>cv2
>lmdb
>numba
>skimage
>tqdm
>pip install git+https://github.com/tatsy/torchmcubes.git
>```
>
>## Pretrained model
>
>You can download the pretrained model and put it into the `ckpt/base` directory.
>
>[Download: pretrained model of FOF-base](https://pan.baidu.com/s/17xdfkT6UKtuX5w0nvSK6yw?pwd=89go)
>
>## Notice
>The input images should be `.png`s with 512*512 resolution in RGBA format. And the alpha channel is the mask.
>
>## TODO
>
>A new version with cosine series(make the code neater)
>
>Code for generating the training data
>
>FOF-smpl
>
>
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
