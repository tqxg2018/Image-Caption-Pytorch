#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from utils import create_input_files_att, create_input_files_tra

if __name__ == '__main__':
    # Create input files (along with word map)
    create_input_files_att(dataset='coco',
                       karpathy_json_path='./data/dataset_coco.json',
                       image_folder='./data/',
                       captions_per_image=5,
                       min_word_freq=5,
                       output_folder='./data/',
                       max_len=50)
    
    create_input_files_tra(dataset='coco',
                       karpathy_json_path='./data/dataset_coco.json',
                       image_folder='./data/',
                       captions_per_image=5,
                       min_word_freq=5,
                       output_folder='./data/',
                       max_len=50)