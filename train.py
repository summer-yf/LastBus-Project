from random import random, randint
from statistics import mode

import torch.optim as optim
import numpy as np
import torch
import torch.nn as nn
from src.flappy_bird import FlappyBird
from src.utils import *
from src.a2c import *

# from stable_baselines.common.policies import MlpPolicy
# from stable_baselines.common import make_vec_env
# from stable_baselines import A2C

LEARNING_RATE = 0.0001
MAX_ITER = 2000000
MAX_EXPERIENCE = 50000
DISCOUNT_FACTOR = 0.8
BATCH_SIZE = 10000

iter = 0
criterion = nn.MSELoss()
game = FlappyBird()

# INSERT YOUR MODEL HERE
model = ActorCriticModel()


optimizer = optim.Adam(model.parameters(), LEARNING_RATE)


def train():

    action = torch.zeros([2], dtype=torch.float32)
    action[0] = 1
    state_image, reward, terminal = game.next_frame(action)
    state = pre_processing(state_image)
    # target_net.load_state_dict(policy_net.state_dict()) # This will copy the weights of the policy network to the target network
    iter = 0
    # optimizer = torch.optim.Adam(policy_net.parameters(), lr=LEARNING_RATE) # The optimizer will update ONLY the parameters of the policy network
    while iter < MAX_ITER:
        iter = iter + 1
        # Exploration or exploitation
        state_tensor = torch.tensor(state, dtype=torch.float32)[None, :, :]
        state_tensor = torch.cat((state_tensor, state_tensor, state_tensor, state_tensor)).unsqueeze(0)
        # Epsilon-Greedy implementation
        # take random action if random float is less than epsilon
        # otherwise select action with highest Q-score for the given state
        # Q: what should this epsilon value be
        epsilon = 1e-4 + ((MAX_ITER - iter) * (0.1 - 1e-4) / MAX_ITER)
        random_action = np.random.uniform(0, 1)

        action = torch.zeros([2], dtype=torch.float32)
        output = model(state_tensor)[0]

        if random_action <= epsilon:
            action_index = torch.randint(2, torch.Size([]), dtype=torch.int)
        else:
            action_index = torch.argmax(output)
        action[action_index] = 1

        next_state_image, reward, terminal = game.next_frame(action)
        next_state = pre_processing(next_state_image)

        print("Iteration: {}/{}, Action: {}, Reward: {}, Q-value: {}".format(
            iter + 1,
            MAX_ITER,
            action,
            #careful of the number in []
            reward, torch.max(model(state_tensor)[1])))

        state = next_state

if __name__ == "__main__":
    train()
