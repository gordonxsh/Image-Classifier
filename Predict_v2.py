# Import all necessary libraries
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
from torch.cuda import device_of
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from collections import OrderedDict

import argparse
import sys

import json

from PIL import Image
import numpy as np


# load checkpoint from input of filepath
def load_checkpoint(filepath, gpu):
    device = torch.device("cuda" if torch.cuda.is_available() and gpu else "cpu")
    checkpoint = torch.load(filepath, map_location=device)
    if checkpoint['arch'] == 'vgg11':
        model = models.vgg11(weights=None)
    elif checkpoint['arch'] == 'vgg13':
        model = models.vgg13(weights=None)
    else:
        print('The selected model is not supported')
        sys.exit(1)
        
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, checkpoint['hidden_units'])),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(checkpoint['hidden_units'], 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    model.classifier = classifier
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.002)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.to(device)
    return model



#Process image

def process_image(image_path):

    with Image.open(image_path) as img:
      #Resize
        width, height = img.size
        if width<height:
          new_size = (256, int(256*height/width))
        else:
          new_size = (int(256*width/height),256)
        
        img.thumbnail(new_size)

      #crop
        width, height = img.size
        left = (width - 224) / 2
        top = (height - 224) / 2
        right = (width + 224) / 2
        bottom = (height + 224) / 2
        img_cropped = img.crop((left, top, right, bottom))

      #color channel adjustment
        np_image = np.array(img_cropped)
        means = np.array([0.485, 0.456, 0.406])
        stds = np.array([0.229, 0.224, 0.225])
        np_image = (np_image / 255 - means) / stds

      #transpose
        np_image = np_image.transpose((2, 0, 1))

      #create tensor
        tensor_image = torch.tensor(np_image, dtype=torch.float32)

    return tensor_image



# define predict function

def predict(image_path, model, topk=3, gpu=False):
    device = torch.device("cuda" if torch.cuda.is_available() and gpu else "cpu")
    model.to(device)
    model.eval()
    inputs = process_image(image_path).unsqueeze(0).to(device)
    with torch.no_grad():
        logps = model.forward(inputs)
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(topk, dim=1)
        top_p = top_p[0].tolist()
        top_class=top_class[0].tolist()

        idx_to_class = {v: k for k, v in model.class_to_idx.items()}
        top_labels = [idx_to_class[class_idx] for class_idx in top_class]

    return top_labels, top_p



# define arguments

def main():
    parser = argparse.ArgumentParser(description='Predict flower name from an image.')
    parser.add_argument('image_path', type=str, help='Image path')
    parser.add_argument('checkpoint_path', type=str, help='Model checkpoint file path')
    parser.add_argument('--top_k', type=int, default=3, help='Return top K predictions')
    parser.add_argument('--category_names_path', type=str, help='File path to category to name json file')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')

    args = parser.parse_args()
    
    model = load_checkpoint(args.checkpoint_path, args.gpu)
    labels, probabilities = predict(args.image_path, model, args.top_k, args.gpu)

    if args.category_names_path:
        with open(args.category_names_path) as f:
            cat_to_name = json.load(f)
        labels = [cat_to_name[str(label)] for label in labels]

    print('Predicted labels:', labels)
    print('Label probabilities:', probabilities)

if __name__ == '__main__':
    main()

