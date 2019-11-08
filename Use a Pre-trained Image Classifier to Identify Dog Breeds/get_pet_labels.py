#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND-revision/intropyproject-classify-pet-images/get_pet_labels.py
#                                                                             
# PROGRAMMER: Moustafa Shomer   
# DATE CREATED: 04-10-2019                                 
# REVISED DATE: 

from os import listdir
def get_pet_labels(image_dir):
    
    in_files = listdir(image_dir)
    results_dic = dict()
    
    for idx in range( 0,len(in_files) ,1):
       if in_files[idx][0] != ".":
           pet_label = ""
           pet_images = in_files[idx].lower().split("_")
        
           for word in pet_images: 
               if word.isalpha():
                    pet_label+= word+" "
                
           if in_files[idx] not in results_dic:
              results_dic[in_files[idx]] = [ pet_label.strip() ]
           else:
               print("** Warning: Duplicate files exist in directory:", in_files[idx])
    return results_dic