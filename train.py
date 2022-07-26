from random import random, randint
from statistics import mode

import torch.optim as optim
import numpy as np
import torch
import torch.nn as nn
from src.replayMemory import ReplayMemory
from src.a2c import *
from src.flappy_bird import FlappyBird
from src.utils import *

# from stable_baselines.common.policies import MlpPolicy
# from stable_baselines.common import make_vec_env
# from stable_baselines import A2C

LEARNING_RATE = 0.0001
MAX_ITER = 2000000
MAX_EXPERIENCE = 50000
DISCOUNT_FACTOR = 0.8
BATCH_SIZE = 10000

iter = 0
# Constructing memory class
memory = ReplayMemory(MAX_EXPERIENCE)
criterion = nn.MSELoss()
game = FlappyBird()
model = ActorNetwork3Layer()
optimizer = optim.Adam(model.parameters(), LEARNING_RATE)


def train():
    # model = A2C(MlpPolicy, env, verbose=1)
    # model.learn(total_timesteps=25000)
    # model.save("a2c_flappy_bird")

    # del model # remove to demonstrate saving and loading

    # model = A2C.load("a2c_flappy_bird")

    # obs = env.reset()
    # while iter < MAX_ITER:
    #     action, _states = model.predict(obs)
    #     obs, rewards, dones, info = env.step(action)
    #     env.render()
    
    # Constructing model (neural network)
    # model = DQN(env.observation_space.shape[0], env.action_space.n)
    # model = CriticNetwork3Layer()
    
    # policy_net = CriticNetwork3Layer()
    # target_net = CriticNetwork3Layer()
    
    action = torch.zeros([2], dtype=torch.float32)
    action[0] = 1
    state_image, reward, terminal = game.next_frame(action)
    state = pre_processing(state_image)
    # target_net.load_state_dict(policy_net.state_dict()) # This will copy the weights of the policy network to the target network

    # optimizer = torch.optim.Adam(policy_net.parameters(), lr=LEARNING_RATE) # The optimizer will update ONLY the parameters of the policy network
    while iter < MAX_ITER:
        iter = iter + 1
        # Exploration or exploitation
        state_tensor = torch.tensor(state, dtype=torch.float32)[None, :, :]
        state_tensor = torch.cat((state_tensor, state_tensor, state_tensor, state_tensor)).unsqueeze(0)
        # Epsilon-Greedy implementation
        epsilon = 1e-4 + (
                (MAX_ITER - iter) * (0.1 - 1e-4) / MAX_ITER)
        u = random()
        random_action = u <= epsilon

        action = torch.zeros([2], dtype=torch.float32)
        output = model(state_tensor)[0]
        if random_action:
            print("Performed random action!")
        
        action_index = [torch.randint(2, torch.Size([]), dtype=torch.int)
                        if random_action
                        else torch.argmax(output)][0]

        action[action_index] = 1

        next_state_image, reward, terminal = game.next_frame(action)
        next_state = pre_processing(next_state_image)
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32)[None, :, :]
        next_state_tensor = torch.cat((next_state_tensor, next_state_tensor, next_state_tensor, next_state_tensor)).unsqueeze(0)

        # Experience is array of gameplay, with each gameplay = (state, action, next_state, reward, terminal)

        action = action.unsqueeze(0)
        reward = torch.from_numpy(np.array([reward], dtype=np.float32)).unsqueeze(0)

        memory.push(state_tensor, action, next_state_tensor, reward, terminal)

        if iter % 1000 == 0:
            batch = memory.sample(BATCH_SIZE)
            # unpack minibatch
            state_batch = torch.cat(tuple(d[0] for d in batch))
            action_batch = torch.cat(tuple(d[1] for d in batch))
            reward_batch = torch.cat(tuple(d[3] for d in batch))
            state_1_batch = torch.cat(tuple(d[2] for d in batch))

            if torch.cuda.is_available():  # put on GPU if CUDA is available
                state_batch = state_batch.cuda()
                action_batch = action_batch.cuda()
                reward_batch = reward_batch.cuda()
                state_1_batch = state_1_batch.cuda()

            # get output for the next state
            output_1_batch = model(state_1_batch)

            # set y_j to r_j for terminal state, otherwise to r_j + gamma*max(Q)
            y_batch = torch.cat(tuple(reward_batch[i] if batch[i][4]
                                    else reward_batch[i] + DISCOUNT_FACTOR * torch.max(output_1_batch[i])
                                    for i in range(len(batch))))

            # extract Q-value
            q_value = torch.sum(model(state_batch) * action_batch, dim=1)

            # PyTorch accumulates gradients by default, so they need to be reset in each pass
            optimizer.zero_grad()

            # returns a new Tensor, detached from the current graph, the result will never require gradient
            y_batch = y_batch.detach()

            # calculate loss
            loss = criterion(q_value, y_batch)

            # do backward pass
            loss.backward()
            optimizer.step()

            # set state to be state_1
            state = next_state

        print("Iteration: {}/{}, Action: {}, Reward: {}, Q-value: {}".format(
            iter + 1,
            MAX_ITER,
            action,
            reward, torch.max(model(state_tensor))))

        state = next_state

def calc_actor_critic_loss():
    actor_loss = []
    critic_loss = []
    rewards = []

    eps = np.finfo(np.float32).eps.item()

    # Get Action Experience
    actions = []

    for r in model.rewards[::-1]:
        # calculate the discounted value
        R = r + DISCOUNT_FACTOR * R
        rewards.insert(0, R)

    rewards = torch.tensor(rewards)
    rewards = (rewards - rewards.mean())/(rewards.std() + eps)

    for (log_prob, value), R in zip(actions, rewards):
        advantage = R - value.item()

        # calculate actor (policy) loss 
        actor_loss.append(-log_prob * advantage)

        # calculate critic (value) loss using L1 smooth loss
        critic_loss.append(F.smooth_l1_loss(value, torch.tensor([R])))

    # reset gradients
    optimizer.zero_grad()

    # sum up all the values of policy_losses and value_losses
    loss = torch.stack(actor_loss).sum() + torch.stack(critic_loss).sum()

    # perform backprop
    loss.backward()
    optimizer.step()
    
if __name__ == "__main__":
    train()
