<div align="center">
   <img src="./docs/images/wolf.png" width="600"><br><br>
</div>

-----------------------------------------------

**Wolf** is an open source library for Invertible Generative (Normalizing) Flows.

This is the code we used in the following papers

>[Decoupling Global and Local Representations from/for Image Generation](https://vixra.org/abs/2004.0222)

>Xuezhe Ma, Xiang Kong, Shanghang Zhang and Eduard Hovy

>[MaCow: Masked Convolutional Generative Flow](https://arxiv.org/abs/1902.04208)

>Xuezhe Ma, Xiang Kong, Shanghang Zhang and Eduard Hovy

>NeurIPS 2019

## Requirements
* Python >= 3.6
* Pytorch >= 1.3.1
* apex
* lmdb >= 0.94
* overrides 


## Installation
1. Install [NVIDIA-apex](https://github.com/NVIDIA/apex).
2. Install [Pytorch and torchvision](https://pytorch.org/get-started/locally/)

## Decoupling Global and Local Representations from/for Image Generation

### Switch Operation
<img src="./docs/images/switch.png" width="600"/>

### CelebA-HQ Samples
<img src="./docs/images/celeba_main.png" width="600"/>

### Running Experiments
First go to the experiments directory:
```base
cd experiments
```
Training a new CIFAR-10 model:
```base
python -u train.py --dataset cifar10 \
    --config  configs/cifar10/glow/glow-gaussian-uni.json \
    --epochs 15000 --valid_epochs 10
    --batch_size 512 --batch_steps 2 --eval_batch_size 1000 --init_batch_size 2048 \
    --lr 0.001 --beta1 0.9 --beta2 0.999 --eps 1e-8 --warmup_steps 50 --weight_decay 1e-6 --grad_clip 0 \
    --image_size 32 --n_bits 8 \
    --data_path <data path> --model_path <model path>
```
The hyper-parameters for other datasets are provided in the paper.
#### Note:
 - Config files, including refined version of Glow and MaCow, are provided [here](https://github.com/XuezheMax/wolf/tree/master/experiments/configs).
 - The argument --batch_steps is used for accumulated gradients to trade speed for memory. The size of each segment of data batch is batch-size / (num_gpus * batch_steps).
 - For distributed training on multi-GPUs, please use ```distributed.py``` or ```slurm.py```, and 
refer to the pytorch distributed parallel training [tutorial](https://pytorch.org/tutorials/intermediate/dist_tuto.html).
 - Please check details of arguments [here](https://github.com/XuezheMax/wolf/blob/master/experiments/options.py).
 
## MaCow: Masked Convolutional Generative Flow
We also implement the MaCow model with distributed training supported. To train a new MaCow model, please use the MaCow config files for different datasets.

## References
```
@incollection{macow2019,
    title = {MaCow: Masked Convolutional Generative Flow},
    author = {Ma, Xuezhe and Kong, Xiang and Zhang, Shanghang and Hovy, Eduard},
    booktitle = {Advances in Neural Information Processing Systems 33, (NeurIPS-2019)},
    year = {2019},
    publisher = {Curran Associates, Inc.}
}
```