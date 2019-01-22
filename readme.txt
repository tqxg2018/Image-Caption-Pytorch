This is an implementation of image caption, based on two different papers.
The two papers are:
1. Show and Tell: A Neural Image Caption Generator
2. Show, Attend and Tell: Neural Image Caption Generation with Visual Attention

In order to run the code, a file called "dataset_coco.json" need to be download and put into the data folder

training steps
1. run   pip install -r requirement.txt
2. run   chmod +x download.sh
3. run   ./download.sh
4. run   python create_input_files.py
5. run   python train-traditional.py
   This is for paper "Show and Tell: A Neural Image Caption Generator"
6. run   python train-attention.py
   This is for paper "Show, Attend and Tell: Neural Image Caption Generation with Visual Attention"

test---under pycharm environment
1. run caption-traditional.py
2. run caption-attention.py

