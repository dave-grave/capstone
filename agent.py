import torch
import random
import numpy as np
import pygame
from pathlib import Path
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
# from plot import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001
f = open("info.txt")  # use to read epoch number

MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)
MODEL_NAME = "model.pth" 
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME


class Agent: 

    def __init__(self):
        self.epoch = int(f.read()) 
        self.epsilon = 0 # randomness parameter
        self.gamma = 0.9 # discount rate (must be < 1)
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(11, 256, 3) # input_size must be 11 and output_size must be 3
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def save(self):
        """MODEL_PATH = Path("models")
        MODEL_PATH.mkdir(parents=True, exist_ok=True)
        MODEL_NAME = "model.pth" 
        MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

        print(f"Saving model to: {MODEL_SAVE_PATH}")"""

        checkpoint = {
            "model_state": self.model.state_dict(),
            "optim_state": self.trainer.optimizer.state_dict()
        }

        torch.save(checkpoint, 'checkpoint.pth')

    def load(self):
        loaded_checkpoint = torch.load('checkpoint.pth')

        print(self.model, self.trainer.optimizer)

        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=0, gamma=self.gamma)

        self.model.load_state_dict(loaded_checkpoint["model_state"])
        self.trainer.optimizer.load_state_dict(loaded_checkpoint["optim_state"])

        print(self.model, self.trainer.optimizer)

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [

            # danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)), 

            # danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)), 

            # danger left 
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)), 

            # move direction
            dir_l,
            dir_r, 
            dir_u, 
            dir_d,

            # food location
            game.food.x < game.head.x, # food left
            game.food.x > game.head.x, # food right
            game.food.y < game.head.y, # food up
            game.food.y > game.head.y # food down
        ]
    
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft() if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE: 
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else: 
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)
    
    def get_action(self, state):
        # random moves: tradeoff exploration w/ exploitation in Deep Learninh
        self.epsilon = 80 - self.epoch # this number can be adjusted 
        final_move = [0,0,0]

        # if epsilon is small enough (early in training), make a random move
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
            
        else: 
            state0 = torch.tensor(state, dtype=torch.float32)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train(): 
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0 
    agent = Agent()
    game = SnakeGameAI()

    agent.load()

    # training loop
    while True:

        # get old state
        state_old = agent.get_state(game)

        # get move 
        final_move = agent.get_action(state_old)

        # perform move, get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember 
        agent.remember(state_old, final_move, reward, state_new, done)

        # check if game over
        if done:
            # get global number of plays
            with open("info.txt", "w") as f:
                f.write(str(agent.epoch))    
            agent.epoch += 1

            # train long memory, plot results
            game.reset()
            agent.train_long_memory()

            if score > record:
                record = score
                agent.save()

            print(f"Game: {agent.epoch}\nScore: {score}\nRecord: {record}")

            """plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)"""

            # TODO: fix plotting on plot.py


if __name__ == '__main__':
    train()