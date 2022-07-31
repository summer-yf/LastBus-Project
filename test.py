import os
import shutil

import torch.optim as optim
import torch
import torch.nn as nn
from src.memory import *
from src.a2c import *
from src.utils import *
from src.flappy_bird import *
import random
from matplotlib import pyplot as plt

#test will test 100000 iter and save the graph

def test():
    save_path = "trained_models"
    ##modify the model you want to test, keep "_"
    model_number = "_100000"
    model = torch.load("{}/flappy_bird{}".format(save_path, model_number))
    model.eval()

    action = torch.zeros(2, dtype=torch.float32)
    state_image, reward, terminal = game.next_frame(action[1])
    state = pre_processing(state_image)
    state = torch.cat((state, state, state, state)).unsqueeze(0)
    alive_time = 0
    iter = 0
    graph_path = "graph"
    alive_stat = []
    while iter < MAX_ITER:
        q_value = model(state)[0]
        action = torch.argmax(q_value, 0)
        next_state_image, reward, terminal = game.next_frame(action)
        next_state_image = pre_processing(next_state_image)
        next_state = torch.cat((state.squeeze(0)[1:, :, :], next_state_image)).unsqueeze(0)
        state = next_state

        alive_time += 1
        if terminal:
            alive_stat.append(alive_time)
            plot_duration(alive_stat)

            alive_time = 0
        if(iter+1) % 100000 == 0:
            plt.savefig('{}/test{}.jpg'.format(graph_path, model_number))
        iter += 1



def plot_duration(duration):
    """Plot durations of episodes and average over last 100 episodes"""
    plt.figure(1)
    plt.clf()
    durations_t = torch.tensor(duration, dtype=torch.float)
    plt.title('Testing...')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)



if __name__ == "__main__":
    MAX_ITER = 100000
    iter = 0
    game = FlappyBird()

    plt.ion()

    test()
    plt.ioff()
    plt.show()
