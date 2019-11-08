#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND-revision/intropyproject-classify-pet-images/get_input_args.py
#                                                                             
# PROGRAMMER: Moustafa Shomer
# DATE CREATED: 02-10-2019                          
# REVISED DATE: 

import argparse
def get_input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='pet_images/', help='path to folder of images')
    parser.add_argument('--arch', default = 'vgg' )
    parser.add_argument('--dogfile', default = 'dognames.txt' )
 
    return parser.parse_args()
