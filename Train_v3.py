# Import all necessary libraries
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from collections import OrderedDict

import argparse
import sys

import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)


# Define script input

def train_model(data_dir,save_dir, arch='vgg11', lr=0.002, hu=500, epochs=3, gpu=False):
    #Summarize parameters
    print(f"Training {arch} model with {hu} hidden units in the first classifier layer and learning rate {lr} for {epochs} epochs. Save to {save_dir}. Use GPU: {gpu}")
    
    #Set up data
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Define transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
    print('finish data transform')
    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
    test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)
    valid_data = datasets.ImageFolder(data_dir + '/valid', transform=valid_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
    print('finish data loading')
    # Initialize model with input of arch
    if arch == 'vgg11':
        model = models.vgg11(pretrained=True)
    elif arch == 'vgg13':
        model = models.vgg13(pretrained=True)
    else:
        print('Please choose between vgg11 or vgg13 for training.')
        sys.exit(1)
    
    for param in model.parameters():
        param.requires_grad = False
    print('finish model selection')
    # Create classifier with input of hidden units
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, hu)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(hu, len(cat_to_name))),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    model.classifier = classifier
    print('finish classifier definition')
    # Set up optimization with input of learning rate
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr) 
    
    # Set up device for training with input of infra
    device = torch.device("cuda" if torch.cuda.is_available() and gpu else "cpu")
    model.to(device)
    print('finish model device setup', device)
    # Train model with input of epochs
    epochs = epochs
    steps = 0
    running_loss = 0
    print_every = 30
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1

            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

            if steps % print_every == 0:
                valid_loss = 0
                running_accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                    
                        valid_loss += batch_loss.item()
                    
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        running_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                accuracy = running_accuracy/len(validloader)
                print("Epoch: ",epoch+1, "; Current step: ", steps, "; Current accuracy: ", accuracy, '; Training loss: ', running_loss/print_every,'; Validation loss: ',valid_loss/len(validloader))
                running_loss = 0
                model.train()
    
    # Save model with input of save_dir
    model.class_to_idx=train_data.class_to_idx
    checkpoint = {'input_size': 224*224,
              'output_size': 102,
              'state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
              'class_to_idx': model.class_to_idx,
              'epochs': epochs,
              'arch':arch,
              'hidden_units':hu}
    torch.save(checkpoint, save_dir)
    print('model saved')
    
def main(args):
    train_model(args.data_directory, args.save_dir, args.arch, args.learning_rate, args.hidden_units, args.epochs, args.gpu)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a new network on a given dataset and save the model as a checkpoint.')
    
    parser.add_argument('data_directory', type=str, help='Directory containing the dataset')
    
    parser.add_argument('--save_dir', type=str, default='checkpoint.pth', help='Directory to save checkpoints')
    parser.add_argument('--arch', type=str, default='vgg11', help='Model architecture')
    parser.add_argument('--learning_rate', type=float, default=0.002, help='Learning rate for training')
    parser.add_argument('--hidden_units', type=int, default=500, help='Number of hidden units for classifier input')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs to train')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training')
    
    args = parser.parse_args()
    main(args)