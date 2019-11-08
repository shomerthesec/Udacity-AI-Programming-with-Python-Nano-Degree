#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND-revision/intropyproject-classify-pet-images/classify_images_hints.py
#                                                                             
# PROGRAMMER: moustafa shomer
# DATE CREATED:06-10-2019                                 
# REVISED DATE: 

from classifier import classifier 
def classify_images(images_dir, results_dic, model):
    for key in results_dic:
       model_label = classifier(images_dir+key, model).lower()
        
       truth = results_dic[key][0]
       if truth in model_label:
          results_dic[key].extend([model_label, 1 ])
       else:
            results_dic[key].extend([model_label, 0 ])

