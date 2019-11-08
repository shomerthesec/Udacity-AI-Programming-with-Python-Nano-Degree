import numpy as np
import time
import torch
from torch import nn
from torch import tensor
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
import argparse
import sys
from PIL import Image
import json
import os
os.environ['QT_QPA_PLATFORM']='offscreen'
import matplotlib.pyplot as plt

        

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--image_path', type= str, default="flowers/test/1/image_06752.jpg", help= 'insert the location of the image ')
    p.add_argument('--arc', default= models.densenet121(pretrained=True), help='if you want to change the model insert the model command like *models.densenet121(pretrained=True)*' )
    p.add_argument('--firstinputlayer', type= int, default=1024, help='insert the number of output layers from your model before freezing it' )
    p.add_argument('--outputlayer', type= int, default=102, help='insert the amount of class you need to train on' )
    p.add_argument('--lossf', default=nn.NLLLoss() , help='choose your loss function by using the command like *nn.NLLLoss()*' )
    p.add_argument('--learningrate', type= float, default=0.001, help='choose your learning rate' )
    p.add_argument('--nepoch', type= int, default=35, help= 'insert the number of epochs you want to train upon ')
    p.add_argument('--checkpoint_name', type= str, default="checkpoint.path", help= 'change the name of the checkpoint path to load but add the extention *path* to it ')
    p.add_argument('--topk', type= int, default=5, help= 'insert the number of probs you want to visualize')

    param=p.parse_args()

    sys.stdout.write(str(model_predict(param)))

def model_predict(param) :
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = param.arc
    with open('cat_to_name.json', 'r') as f:
           cat_to_name = json.load(f)
    def modelbuild(firstinputlayer=param.firstinputlayer, outputlayer=param.outputlayer, lossf=param.lossf, learningrate=param.learningrate, nepoch =param.nepoch) :
        # Freeze parameters so we don't backprop through them
        for param in model.parameters():
            param.requires_grad = False

        from collections import OrderedDict
        classifier = nn.Sequential(OrderedDict([
                                  ('fc1', nn.Linear(firstinputlayer, 512)),
                                  ('relu', nn.ReLU()),
                                  ('fc2', nn.Linear(512, outputlayer)),
                                  ('output', nn.LogSoftmax(dim=1))
                                  ]))

        model.classifier = classifier
        model.to(device)
        arch=model
        return model
        
    def process_image(image):
        image= Image.open(image)
        testtransform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225]) ])
        image=testtransform(image)
        
        return image

    def predict(image_path=param.image_path , topk=param.topk):
        # to create the model
        model=modelbuild()
        # load the net parameters by name

        model.load_state_dict(torch.load(param.checkpoint_name,map_location=lambda storage, loc: storage))

        # TODO: Implement the code to predict the class from an image file
        model.eval()
        model.to(device)
        image=process_image(image_path)
        image= image.unsqueeze(0)
        image = image.float()
        output=model(image)
        output=F.softmax(output.data ,dim=1)
        k=torch.topk(output,topk)
        prob= np.array(k[0][0])
        name = [cat_to_name[str(index + 1)] for index in np.array(k[1][0])]
        #plt.barh(y=name, width=prob)
        print('The class of the flower is {',name[0],'} ... with probabilty of (', prob[0],').')
    predict()


if __name__ =='__main__':
    main()

