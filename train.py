import torch.optim as optim
import torch
import torch.nn as nn
from src.memory import *
from src.a2c import *
from src.utils import *
from src.flappy_bird import *
import random
from matplotlib import pyplot as plt

def train_model():
    
    action = torch.zeros(2, dtype=torch.float32)
    iter = 0
    state_image, reward, terminal = game.next_frame(action[1])
    state = pre_processing(state_image)
    state = torch.cat((state, state, state, state)).unsqueeze(0)
    alive_stat = []
    alive_time = 0
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
            print("Performed random action!")
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
        
        if iter % 10 == 0:
            update_model()

        print("Iteration: {}/{}, Action: {}, Reward: {}, Q-value: {}".format(
            iter + 1,
            MAX_ITER,
            int(action[0][1]),
            reward, torch.max(q_value)))

        alive_time += 1
        
        if terminal:
            alive_stat.append(alive_time)
            plot_duration(alive_stat)
            alive_time = 0

def update_model():
    
    batch = memory.sample(BATCH_SIZE)
    # unpack minibatch
    state_batch = torch.cat(tuple(d[0] for d in batch))
    action_batch = torch.cat(tuple(d[1] for d in batch))
    next_state_batch = torch.cat(tuple(d[2] for d in batch))
    reward_batch = torch.cat(tuple(d[3] for d in batch))
    terminal_batch = [d[4] for d in batch]
    
    if torch.cuda.is_available(): 
        state_batch = state_batch.cuda()
        action_batch = action_batch.cuda()
        reward_batch = reward_batch.cuda()
        next_state_batch = next_state_batch.cuda()
        terminal_batch = terminal_batch.cuda()

    next_action_batch = model(next_state_batch)
    
    # (1) reward if terminal, reward + discount factor * max next Q if not terminal
    y_batch = torch.cat(tuple(reward_batch[i] if batch[i][4]
                                  else reward_batch[i] + DISCOUNT_FACTOR * torch.max(next_action_batch[i])
                                  for i in range(len(batch))))   
    
    # (2) current q value
    q_value = torch.sum(model(state_batch) * action_batch, dim=1)
    
    optimizer.zero_grad()

    # Free up y_batch to prevent from code error when running loss.backward
    y_batch = y_batch.detach()

    # Calculate loss ((1) vs (2))
    loss = criterion(q_value, y_batch)

    # Update Model
    loss.backward()
    optimizer.step()
    


def plot_duration(duration):
    plt.figure(1)
    plt.clf()
    
    plt.xlabel('Trial')
    plt.ylabel('Frame Survived')
    plt.plot(duration)
    

    plt.pause(0.001)

if __name__ == "__main__":
    
    
    LEARNING_RATE = 1e-5
    MAX_ITER = 5000000
    MAX_EXPERIENCE = 40
    DISCOUNT_FACTOR = 0.99
    BATCH_SIZE = 30
    INITIAL_EPSILON = 0.2
    
    iter = 0
    
    # Constructing memory class
    memory = Memory(MAX_EXPERIENCE)
    
    criterion = nn.MSELoss()
    game = FlappyBird()
    model = ThreeLayerConvModel()
    optimizer = optim.Adam(model.parameters(), LEARNING_RATE)
    
    plt.ion()
    train_model()
    plt.ioff()
    plt.show()
