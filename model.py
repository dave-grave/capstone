import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import numpy as np
from pathlib import Path

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
    
    def save(self):
        MODEL_PATH = Path("models")
        MODEL_PATH.mkdir(parents=True, exist_ok=True)

        MODEL_NAME = "model.pth" 

        MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

        print(f"Saving model to: {MODEL_SAVE_PATH}")
        torch.save(obj=self.state_dict(), f=MODEL_SAVE_PATH)

        # TODO: craete importing of model's state_dict as a checkpoint so we don't have to restart every time

    def load(self):
        loaded_model = Linear_QNet(11, 256, 3)

        MODEL_PATH = Path("models")
        MODEL_PATH.mkdir(parents=True, exist_ok=True)
        
        MODEL_NAME = "model.pth" 

        MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

        loaded_model.load_state_dict(torch.load(f=MODEL_SAVE_PATH))

        return loaded_model


class QTrainer: 
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(np.stack(state), dtype=torch.float32)
        next_state = torch.tensor(np.stack(next_state), dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.float64)
        reward = torch.tensor(reward, dtype=torch.float32)

        if len(state.shape) == 1:
            # make dimension into (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, ) # make a tuple w/ only one value

        # 1: predicted Q values with current state
        pred = self.model(state)

        target = pred.clone()
        for i in range(len(done)):
            Q_new = reward[i]
            if not done[i]:
                Q_new = reward[i] + self.gamma * torch.max(self.model(next_state[i]))

            target[i][torch.argmax(action[i]).item()] = Q_new

        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()
    