import pyautogui
import time
from enum import Enum

"""
plan for implementation of reader.py below:
1) create functions for agent to perform snake movements
2) turn snake movements into data
3) connect with neural network

NORMAL SPEED: 135ms per tile
FAST SPEED: 89.1ms per tile
"""

SPEED = 0.0891
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


class Snake():
    def __init__(self):
        self.direction = Direction.RIGHT
        pyautogui.hotkey('win', 'h') 

    def reset():
        time.sleep(1)
        pyautogui.click(876, 716)

    def _move():
        pass 
        """clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx] # no change in direction
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] # right turn 
        else: # [0, 0, 1]
            next_idx = (idx - 1) % 4 
            new_dir = clock_wise[next_idx] # left turn

        self.direction = new_dir 

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
            
        self.head = Point(x, y)"""


if __name__ == "__main__":
    game = Snake()
    for i in range(10):
        Snake.reset()
        pyautogui.press('d')
    
# (0,0) is at top left. x increases going right. y increases going down.
# print resolution
# print(pyautogui.size())

# move the mouse
# pyautogui.moveTo(100, 100, 10)

# print current position of mouse
# print(pyautogui.position())

# pyautogui.rightClick(200, 200)
