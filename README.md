Intro
-----

This repository provides the codes for the CVPR16 paper, “[Deep Region and Multi-Label Learning for Facial Action Unit Detection](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Zhao_Deep_Region_and_CVPR_2016_paper.pdf)".
This code aims for training a convolutional network that contains a *region layer* for specializing the learned kernels on different facial regions, and meanwhile utilizes a multi-label cross-entropy to jointly learn 12 AUs.
This implementation is based on [Caffe Toolbox](https://github.com/BVLC/caffe). 


Code structure
--------------

Based on the caffe toolbox, we organize the source files as follows:

`include/caffe/`: Header files that contains the declaration of our implemented layers

`prototxt/`: Network architecture we used to compuare and report in our paper

`src/caffe/layers/`: Source files of our implemented layers


More info
---------

- **Contact**:  Please send comments to Kaili Zhao (kailizhao@bupt.edu.cn)  
- **Citation**: If you use this code in your paper, please cite the following:
```
@inproceedings{zhao2016deep,
  title={Deep Region and Multi-Label Learning for Facial Action Unit Detection},
  author={Zhao, Kaili and Chu, Wen-Sheng and Zhang, Honggang},
  booktitle={CVPR},
  year={2016}
}
```
