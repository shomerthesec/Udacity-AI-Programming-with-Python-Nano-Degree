import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import json
import argparse
import sys
from collections import OrderedDict

def main():
    parser = argparse.ArgumentParser(description='Image Classifier')
    parser.add_argument('--directory', type= str, default="flowers", help= 'insert the directory location ')
    parser.add_argument('--image_path', type = str, default = 'flowers/test/56/image_02779.jpg', help = 'Insert image path')
    parser.add_argument('--json', type = str, default = 'flower_to_name.json', help = 'cat_to_name json file')
    parser.add_argument('--arch', type = str, default='vgg16' ,help=' input vgg16 or input densenet121 depends on the architecture ' )
    parser.add_argument('--first_layer', type= int, default= 25088, help= ' specify no of input layers input 25088 for vgg16 or 1024 for densenet121')
    parser.add_argument('--final_layer', type= int, default= 102, help= 'specify no of final layers')
    parser.add_argument('--checkpoint', type = str, default = 'checkpoint_from_train.path', help = 'insert the checkpoint name')
    parser.add_argument('--topk', type = int, default = 3, help = 'topk and probabilities')
    parser.add_argument('--lr', type= float, default=0.01, help='Choose a learning rate' )
    parser.add_argument('--criterion',default=nn.NLLLoss(), help='Choose a loss function' )
    parser.add_argument('--epochs', type= int, default= 1, help= 'Insert number of epochs for model ')
    parser.add_argument('--print_every', type= int, default=40, help= 'How often to print model statues')
    parser.add_argument ('--device', type = str, default='cuda',help = "Type 'cuda' to process via GPU , 'CPU' to process via CPU")
    argument_input=parser.parse_args()
    sys.stdout.write(str(train_module(argument_input)))    
    
def train_module(argument_input):
    # using model architecture depending on architecture input by user
    if argument_input.arch == 'vgg16' : 
        model_arch = models.vgg16(pretrained=True)
    elif argument_input.arch == 'densenet121' :
        model_arch = models.densenet121(pretrained=True)
    
    # Loading the dataset via function
    def img_load(directory=argument_input.directory):
        data_dir = str(directory)
        train_dir = data_dir + '/train'
        valid_dir = data_dir + '/valid'
        test_dir = data_dir + '/test'
        
        # TODO: Define your transforms for the training, validation, and testing sets

        train_transforms = transforms.Compose([transforms.RandomRotation(15),
                                              transforms.RandomResizedCrop(224),
                                             transforms.RandomVerticalFlip(),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.485, 0.456, 0.406),
                                             (0.229, 0.224, 0.225))])

        test_transforms = transforms.Compose([ transforms.RandomResizedCrop(224),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize([0.485, 0.456, 0.406], 
                                                                        [0.229, 0.224, 0.225])])

        valid_transforms = transforms.Compose([transforms.Resize(256),
                                                        transforms.CenterCrop(224),
                                                        transforms.ToTensor(),
                                                        transforms.Normalize([0.485, 0.456, 0.406], 
                                                                             [0.229, 0.224, 0.225])])

    
        # TODO: Load the datasets with ImageFolder
        train_datasets = datasets.ImageFolder(root=train_dir, transform=train_transforms)
        valid_datasets = datasets.ImageFolder(root=valid_dir, transform=valid_transforms)
        test_datasets = datasets.ImageFolder(root=test_dir, transform=test_transforms)
    
        # TODO: Using the image datasets and the trainforms, define the dataloaders
        train_loader = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_datasets, batch_size=32, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_datasets, batch_size=64,shuffle=True)
        
        return train_loader,valid_loader,test_loader,train_datasets,valid_datasets,test_datasets
    
    # Taking returned values into variables to use it
    def define_model(arch=model_arch ,first_layer=argument_input.first_layer,final_layer=argument_input.final_layer):
        # TODO: Build and train your network
        # loading the VGG16 pretrained networ
        model=arch
        # Freeze the parameters in features section in VGG16 
        # making it static and ONLY use it as a feature detector
        # preventing back propogation throught them
        for param in model.parameters():
            param.requires_grad = False
        
        # Defining our new classifier
        # input layer will take 25088 feature cause it has to match that of the VGG16
        # 1000 is the number of neurons in the next layer of my choice
        # 'drop' reduces the chance of Overfitting 
        # softmax to use the outputs as probabilites between 0 and 1
        classifier = nn.Sequential(OrderedDict([
                                  ('fc1', nn.Linear(first_layer, 512)),
                                  ('relu', nn.ReLU()),
                                  ('drop', nn.Dropout(p=0.5)),
                                  ('fc2', nn.Linear(512,final_layer)),
                                  ('output', nn.LogSoftmax(dim=1))
                                  ]))
        model.classifier = classifier
        
        model.to(device)
        return model
    
    
    # Training the new classifier
    def train_class(model,lr=argument_input.lr,criterion = argument_input.criterion ,epochs = argument_input.epochs,
                    print_every = argument_input.print_every):
        # defining Loss function
        # using the negative log liklihood loss as criterion
        # I'll use the gradient decent optimizer SGD which will update wieghts and parameters
        optimizer = optim.SGD(model.classifier.parameters() , lr=lr, momentum=0.9)
        for e in range(epochs):
            running_loss = 0
            steps = 0
            for ii, (inputs, labels) in enumerate(train_loader):
                steps += 1
                inputs,labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                output = model.forward(inputs)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
    
                running_loss += loss.item()
    
                if steps % print_every == 0:
                    model.eval()
                    valid_loss = 0
                    accuracy=0
    
    
                    for ii, (inputs2,labels2) in enumerate(valid_loader):
                        optimizer.zero_grad()
    
                        inputs2, labels2 = inputs2.to(device) , labels2.to(device)
                        model.to(device)
                        with torch.no_grad():    
                            output = model.forward(inputs2)
                            valid_loss = criterion(output,labels2)
                            ps = torch.exp(output).data
                            equality = (labels2.data == ps.max(1)[1])
                            accuracy += equality.type_as(torch.FloatTensor()).mean()
    
                    valid_loss = valid_loss / len(valid_loader)
                    accuracy = accuracy /len(valid_loader)
    
    
    
                    print("Epoch: {}/{}... ".format(e+1, epochs),
                          "Loss: {:.4f}... ".format(running_loss/print_every),
                          "Validation Lost {:.4f}... ".format(valid_loss),
                           "Accuracy: {:.4f}... ".format(accuracy))
    
    
                    running_loss = 0
        return
    
    # GPU usage depend on gpu input by user
    if argument_input.device == 'cuda':
        device = torch.device("cuda" )
    elif argument_input.device =='CPU':
        device = torch.device("cpu")  
        
    train_loader,valid_loader,test_loader,train_datasets,valid_datasets,test_datasets=img_load()
    model= define_model()
    train_class(model)
    
    # saving model
    # Saving model checkpoint
    torch.save(model.state_dict(), argument_input.checkpoint)
    print('Finished training !')


if __name__ =='__main__':

    main()
