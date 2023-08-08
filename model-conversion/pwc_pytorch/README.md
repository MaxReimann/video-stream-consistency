# Lite-PWCNet

This directory contains the lite optical flow network used in our stabilization implemented in pytorch. 

It is an adaption of the original PWC-Net introduced by Sun et al. [1], and implemented in PyTorch by S. Niklaus [2].
While the original network trained by Sun et al., uses a custom cupy layer for correlation in [2], we use the [spatial-correlation-sampler](https://pypi.org/project/spatial-correlation-sampler/) package for our custom trained variants - see model_spec for different variants we trained.


We provide our trained pytorch weights for the dense , light and light with seperated refinement trained on chairs-things-sintel in [google drive](https://drive.google.com/drive/folders/1im1_ZLnN7S5OZcWbh_ZPC03RM1Mh3i4o?usp=sharing). 
The models are the sources for the ONNX models provided in the repo models directory  (PWCNet-dense-w-prepoc.onnx = pwcnet-my-dense-chairs-things-sintel.pth, PWCNet-light-w-prepoc.onnx = pwcnet-my-light4dec-sepref-chairs-things-sintel.pth), the "w-preproc" means the preprocessing resizing to a multiple of 64 was traced as well.



## references
```
[1]  @inproceedings{Sun_CVPR_2018,
         author = {Deqing Sun and Xiaodong Yang and Ming-Yu Liu and Jan Kautz},
         title = {{PWC-Net}: {CNNs} for Optical Flow Using Pyramid, Warping, and Cost Volume},
         booktitle = {IEEE Conference on Computer Vision and Pattern Recognition},
         year = {2018}
     }
```

```
[2]  @misc{pytorch-pwc,
         author = {Simon Niklaus},
         title = {A Reimplementation of {PWC-Net} Using {PyTorch}},
         year = {2018},
         howpublished = {\url{https://github.com/sniklaus/pytorch-pwc}}
    }
```
