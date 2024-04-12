"""
plan for implementation of reader.py below:
1) implement computer vision agent
2) create functions for agent to read snake movements
3) turn snake movements into data
4) connect with neural network

NORMAL SPEED: 135ms per tile
"""

import time
import pyautogui

# control mouse movements, clicks, drags, scroll
# (0,0) is at top left. x increases going right. y increases going down.
print(pyautogui.size())

# get current cursor position
x, y = pyautogui.position()

# check if the cursor is on screen
print(pyautogui.onScreen(x, y))