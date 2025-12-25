
import torch
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from torch.utils import data
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F

corp_size=256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MEAN_STD = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

import warnings
warnings.filterwarnings("ignore")

# Load the model as in your existing code.
from timmML2.models.factory import create_model
model = create_model('efficientnet_lite2')
PATH_model="teacher.pt"

# model.load_state_dict(torch.load(PATH_model))
model.load_state_dict(torch.load(PATH_model, map_location=device))
model.eval()
# /s
# Define image transformations.
img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN_STD[0],MEAN_STD[1])
])

# Open your image.
filename = './07_08_2023_16_38_48.jpg'  # Update this to the path of your image.
img = Image.open(filename).convert('RGB')

c_size=256#240 for lite1, 256 for lite2
c=0

# Resize image as per existing code.
if img.size[0]<c_size*2 and c==0:
    img=img.resize((c_size*2,int(c_size*2*img.size[1]/img.size[0])))
    c=1
if img.size[1]<c_size*2 and c==0:
    img=img.resize((int(c_size*2*img.size[0]/img.size[1]),c_size*2))
    c=1

kk=14
if img.size[0]>c_size*kk and c==0:
    img=img.resize((c_size*kk,int(c_size*kk*img.size[1]/img.size[0])))
    c=1
if img.size[1]>c_size*kk and c==0:
    img=img.resize((int(c_size*kk*img.size[0]/img.size[1]),c_size*kk))

# Perform image transformations.
input_tensor = img_transform(img)
input_batch = input_tensor.unsqueeze(0) 
input_batch = input_batch.to('cuda')
model.to('cuda')

# Generate and print output.
with torch.no_grad():
    output,features = model(input_batch)

# pred_count= torch.sum(output).item()
pred_count = round(torch.sum(output).item())

print(f"Predicted count: {pred_count}")
