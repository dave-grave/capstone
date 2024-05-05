import torch
import random
import numpy as np
import pygame
from pathlib import Path
from collections import deque
from googlesnake.gs_GAME import Snake, Direction
from model import Linear_QNet, QTrainer