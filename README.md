# Image Caption Pytorch
Pytorch implementation of image caption problem.
## Introduction
This is an implementation of image caption, based on two different papers.
The two papers are:
1. [Show and Tell: A Neural Image Caption Generator](https://arxiv.org/pdf/1411.4555.pdf)
2. [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](https://arxiv.org/pdf/1502.03044v1.pdf)

The code is based on [a-PyTorch-Tutorial-to-Image-Captioning](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning).

## Model Training and Testing
In order to run the code, a file called "dataset_coco.json" need to be download and put into the data folder.
You can [download](http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip) the file here.

### Training
1. run   pip install -r requirement.txt
2. run   chmod +x download.sh
3. run   ./download.sh
4. run   python create_input_files.py
5. run   python train-traditional.py
   <br>This is for the paper "[Show and Tell: A Neural Image Caption Generator](https://arxiv.org/pdf/1411.4555.pdf)"
6. run   python train-attention.py
   <br>This is for the paper "[Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](https://arxiv.org/pdf/1502.03044v1.pdf)"

### Testing
The testing code is tested under pycharm environment.
1. run caption-traditional.py
2. run caption-attention.py

## Pretrained model
You can download the pretrained model here
1. The [traiditional model](https://pan.baidu.com/s/1mS6yE-HofDTcKLIZjp_GFQ), the password is `yl2u`.
2. The [attentaion model](https://pan.baidu.com/s/1dpG4djWAs9CpFlxJ_uJPYw), the password is `lsv7`.
