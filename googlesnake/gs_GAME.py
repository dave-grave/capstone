import pyautogui
import time
import numpy as np
from enum import Enum
from collections import namedtuple

"""
plan for implementation of reader.py below:
1) create computer vision functions for agent 
2) turn snake movements into data
3) connect with neural network

SLOW SPEED: idk ms per tile
NORMAL SPEED: 135ms per tile
FAST SPEED: 89.1ms per tile
"""

SPEED = 0.0891
Point = namedtuple('Point', ['x', 'y'])


class Direction(Enum):
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    UP = 4


class Game():
    def __init__(self):
        # ubuntu shortcut to minimize tab
        # pyautogui.hotkey('win', 'h') 
        
        self.reset()

    def reset(self):
        self.head = Point(2, 4) # (0,0) is at top left. x increases going right. y increases going down.
        self.direction = Direction.RIGHT
        self.snake = [self.head, 
                      Point(self.head.x-1, self.head.y),
                      Point(self.head.x-2, self.head.y)]
        
        self.score = 0
        # TODO: use computer vision with pyautogui to locate where apple is and then convert it into Point
        self.frame_iteration = 0 

        time.sleep(1)
        pyautogui.click(876, 716)

    def play_step(self, action):
        # 1. increment frames
        self.frame_iteration += 1  

        # 2. move 
        self._move(action)
        self.snake.insert(0, self.head)

        # 3. check if game over
        reward = 0   
        game_over = False

        if self.is_collision():
            game_over = True
            reward = -10   
            return reward, game_over, self.score
        
        # TODO: place new food or just move

        # 5. return game over and score
        return reward, game_over, self.score

    def is_collision(self, pt=None):
            if pt is None:
                pt = self.head

            # hits boundary
            if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
                return True
            
            # hits itself
            if pt in self.snake[1:]:
                return True
            
            return False
    
    def _move(self, action): 
        # action = [STRAIGHT, RIGHT, LEFT]
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
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
            x += 1
        elif self.direction == Direction.LEFT:
            x -= 1
        elif self.direction == Direction.DOWN:
            y += 1
        elif self.direction == Direction.UP:
            y -= 1
            
        self.head = Point(x, y)


if __name__ == "__main__":
    game = Game()
    for i in range(10):
        Game.reset(game)
        pyautogui.press('d')

