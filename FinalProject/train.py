
import matplotlib.pyplot as plt
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

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--directory', type= str, default="flowers", help= 'insert the directory location ')
    p.add_argument('--arc', default= models.densenet121(pretrained=True), help='if you want to change the model insert the model command like *models.densenet121(pretrained=True)*' )
    p.add_argument('--firstinputlayer', type= int, default=1024, help='insert the number of output layers from your model before freezing it' )
    p.add_argument('--outputlayer', type= int, default=102, help='insert the amount of class you need to train on' )
    p.add_argument('--lossf', default=nn.NLLLoss() , help='choose your loss function by using the command like *nn.NLLLoss()*' )
    p.add_argument('--learningrate', type= float, default=0.001, help='choose your learning rate' )
    p.add_argument('--nepoch', type= int, default=35, help= 'insert the number of epochs you want to train upon ')
    p.add_argument('--print_every', type= int, default=10, help= 'print the training statistics every...? ')
    p.add_argument('--checkpoint_name', type= str, default="trained_model.path", help= 'change the name of the checkpoint path as you like but add the extention *path* to it ')


    param=p.parse_args()

    sys.stdout.write(str(model_creation(param)))



def model_creation(param) :
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = param.arc
    def imagesuploader (directory=param.directory) :
            data_dir=str(directory)
            batchsize= 64
            transforms1 = transforms.Compose([transforms.Resize(250),
                                             transforms.CenterCrop(224),
                                             transforms.ToTensor()])                          # TODO: compose transforms here
            dataset = datasets.ImageFolder(data_dir, transform=transforms1) # TODO: create the ImageFolder
            dataloader = dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchsize, shuffle=True) # TODO: use the ImageFolder dataset to create the DataLoader
            # Define transforms for the training data and testing data
            #normalization for gray scale is done with ([number],[number]) cause it's 1D

            train_transforms = transforms.Compose([transforms.RandomRotation(80),
                                                   transforms.RandomResizedCrop(224),
                                                   transforms.RandomHorizontalFlip(),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize([0.485, 0.456, 0.406], 
                                                                      [0.229, 0.224, 0.225])])

            test_transforms = transforms.Compose([ transforms.RandomResizedCrop(224),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize([0.485, 0.456, 0.406], 
                                                                        [0.229, 0.224, 0.225])])

            validation_transforms = transforms.Compose([transforms.Resize(256),
                                                        transforms.CenterCrop(224),
                                                        transforms.ToTensor(),
                                                        transforms.Normalize([0.485, 0.456, 0.406], 
                                                                             [0.229, 0.224, 0.225])])

            train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms, )
            test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)
            validation_data = datasets.ImageFolder(data_dir+'/valid', transform=validation_transforms)


            trainloader = torch.utils.data.DataLoader(train_data, batch_size=batchsize,shuffle=True)
            testloader = torch.utils.data.DataLoader(test_data, batch_size=batchsize,shuffle=True)
            vloader = torch.utils.data.DataLoader(validation_data, batch_size =32,shuffle = True)
        
            return trainloader, testloader, vloader, train_data, test_data, validation_data

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

    def train(model , lossf=param.lossf , learningrate=param.learningrate , nepoch = param.nepoch, print_every= param.print_every ):
        epochs = nepoch
        # cross entropy loss combines softmax and nn.NLLLoss() in one single class.
        criterion = lossf

        ## TODO: specify optimizer 
        # stochastic gradient descent with a small learning rate and some momentum
        learning_rate= learningrate
        optimizer = optim.SGD(model.classifier.parameters() , lr=learning_rate, momentum=0.9)
        for n_epochs in range(epochs):
            running_loss = 0
            steps = 0
            for ii, (inputs, labels) in enumerate(trainloader):
                steps += 1

                inputs,labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                # Forward and backward passes
                outputs = model.forward(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if steps % print_every == 0:
                    model.eval()
                    vlost = 0
                    accuracy=0


                    for ii, (inputs2,labels2) in enumerate(vloader):
                        optimizer.zero_grad()

                        inputs2, labels2 = inputs2.to(device) , labels2.to(device)
                        model.to(device)
                        with torch.no_grad():    
                            outputs = model.forward(inputs2)
                            vlost = criterion(outputs,labels2)
                            ps = torch.exp(outputs).data
                            equality = (labels2.data == ps.max(1)[1])
                            accuracy += equality.type_as(torch.FloatTensor()).mean()

                    vlost = vlost / len(vloader)
                    accuracy = accuracy /len(vloader)



                    print("Epoch: {}/{}... ".format(n_epochs+1, epochs),
                          "Loss: {:.4f}... ".format(running_loss/print_every),
                          "Validation Lost {:.4f}... ".format(vlost),
                           "Accuracy: {:.4f}... ".format(accuracy))


                    running_loss = 0
        return 

    trainloader, testloader, vloader, train_data, test_data, validation_data = imagesuploader()

    model=modelbuild()

    train(model)

    print('training finished... \n')

    #to save the model
    torch.save(model.state_dict(),param.checkpoint_name)
    print('model saved!!..')
    return


if __name__ =='__main__':
    main()

