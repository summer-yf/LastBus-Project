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
import time as t

#train_model will train model 2 million times and save model and graph every 100k times

def train_model():

    action = torch.zeros(2, dtype=torch.float32)
    iter = 0
    state_image, reward, terminal = game.next_frame(action[1])
    state = pre_processing(state_image)
    state = torch.cat((state, state, state, state)).unsqueeze(0)
    alive_stat = []
    alive_time = 0
    log_path = "tensorboard"
    save_path = "trained_models"
    graph_path = "graph"
    if os.path.isdir(log_path):
        shutil.rmtree(log_path)
    os.makedirs(log_path)
    while iter < MAX_ITER:
        iter = iter + 1
        q_value = model(state)[0]

        action = torch.zeros(2, dtype=torch.float32)
        if torch.cuda.is_available():
            action = action.cuda()

        # Epsilon-Greedy implementation
        epsilon = INITIAL_EPSILON * (1 - iter / MAX_ITER)
        u = random.random()
        random_action = False if epsilon < u else True
        index = torch.argmax(q_value).item()

        if random_action:
            index = randint(0, 1)
        action[index] = 1

        next_state_image, reward, terminal = game.next_frame(action[1])
        next_state_image = pre_processing(next_state_image)
        next_state = torch.cat((state.squeeze(0)[1:, :, :], next_state_image)).unsqueeze(0)

        reward = torch.from_numpy(np.array([reward], dtype=np.float32)).unsqueeze(0)

        action = action.unsqueeze(0)

        # save replay
        memory.push(state, action, next_state, reward, terminal)

        state = next_state
        if(iter+1) % 4 == 0:
            target_model.load_state_dict(model.state_dict())

        if len(memory.memory) >= BATCH_SIZE:
            update_model()

        alive_time += 1
        if(iter+1) % 100000 == 0:
            torch.save(model, "{}/flappy_bird_{}".format(save_path, iter+1))

        if terminal:
            alive_stat.append(alive_time)
            plot_duration(alive_stat)

            alive_time = 0
        if(iter+1) % 100000 == 0:
            plt.savefig('{}/train_{}.jpg'.format(graph_path, iter+1))
    torch.save(model,"{}/flappy_bird".format(save_path))



def update_model():

    batch = memory.sample(BATCH_SIZE)
    # unpack minibatch
    state_batch = torch.cat(tuple(d[0] for d in batch))
    action_batch = torch.cat(tuple(d[1] for d in batch))
    next_state_batch = torch.cat(tuple(d[2] for d in batch))
    reward_batch = torch.cat(tuple(d[3] for d in batch))
    terminal_batch =[d[4] for d in batch]

    if torch.cuda.is_available():
        state_batch = state_batch.cuda()
        action_batch = action_batch.cuda()
        reward_batch = reward_batch.cuda()
        next_state_batch = next_state_batch.cuda()
        terminal_batch = terminal_batch.cuda()

    next_action_batch = target_model(next_state_batch)
    # if dead, rj, otherwise r_j + gamma*max(Q_t+1)
    y_batch = torch.cat(tuple(reward_batch[i] if batch[i][4]
                                  else reward_batch[i] + DISCOUNT_FACTOR * torch.max(next_action_batch[i])
                                  for i in range(len(batch))))
    # Extract Q-value (this part i don't understand)

    q_value = torch.sum(model(state_batch) * action_batch, dim=1)

    optimizer.zero_grad()

    # Returns a new Tensor, detached from the current graph, the result will never require gradient
    y_batch = y_batch.detach()

    # Calculate loss
    loss = criterion(q_value, y_batch)

    # Do backward pass
    loss.backward()
    optimizer.step()



def plot_duration(duration):
    """Plot durations of episodes and average over last 100 episodes"""
    plt.figure(1)
    plt.clf()
    durations_t = torch.tensor(duration, dtype=torch.float)
    plt.title('Training...')
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


    LEARNING_RATE = 1e-5
    MAX_ITER = 2000000
    MAX_EXPERIENCE = 40
    DISCOUNT_FACTOR = 0.99
    BATCH_SIZE = 30
    INITIAL_EPSILON = 0.2

    iter = 0
    start_time = t.time()
    # Constructing memory class
    memory = Memory(MAX_EXPERIENCE)

    criterion = nn.MSELoss()
    game = FlappyBird()
    model = ThreeLayerConvModel()
    target_model = ThreeLayerConvModel()
    optimizer = optim.Adam(model.parameters(), LEARNING_RATE)
    graph_path = "graph"
    plt.ion()
    train_model()
    print(t.time - start_time)
    plt.ioff()
    #plt.savefig("{}/train_{}".format(graph_path, iter+1))
    plt.show()
