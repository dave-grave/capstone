# NOTE: make sure to pip install torch, torchvision, pygame, numpy, matplotlib, ipython

import torch
import pygame

print(torch.__version__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
