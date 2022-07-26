import torch.nn as nn
import torch.nn.functional as F
import torch

class ActorNetwork3Layer(nn.Module):
    def __init__(self):
        super(ActorNetwork3Layer, self).__init__()
        # input image is 90 x 90
        self.conv1 = nn.Conv2d(4, 5, 4)
        self.pool1 = nn.MaxPool2d(3)
        self.conv2 = nn.Conv2d(5, 10, 6, 4)
        self.fc1 = nn.Linear(360, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, input):
        output = self.pool1(F.relu(self.conv1(input)))
        output = F.relu(self.conv2(output))
        output = output.view(-1, 360)
        output = F.relu(self.fc1(output))
        output = F.relu(self.fc2(output))
        output = F.relu(self.fc3(output))

        return output

class CriticNetwork3Layer(nn.Module):
    def __init__(self, state_space_dim, action_space_dim):
        super(CriticNetwork3Layer, self).__init__()

        self.fc1 = nn.Linear(state_space_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_space_dim)

        self.loss = nn.MSELoss()
        
    def forward(self, input):
        output = F.relu(self.fc1(input))
        output = F.relu(self.fc2(output))
        output = F.relu(self.fc3(output))

        return output

class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()

        self.layer1 = nn.Sequential(nn.Conv2d(1,32,8), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Conv2d(32,64,3), nn.ReLU())
        self.out = nn.Linear(5625,2)
        self._create_weights()

    def _create_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.01, 0.01)
                nn.init.constant_(m.bias, 0)

    
    def forward(self, input):
        in_to_layer1 = self.layer1(input)
        layer1_to_layer2 = self.layer2(in_to_layer1)
        output = layer1_to_layer2.view(layer1_to_layer2.size(0), -1)
        output = self.out(output)

        return output

class ActorCriticModel():
    def __init__(self):
        super(ActorCriticModel, self).__init__()
        self.layer1 = nn.Linear(4, 128)
        self.actor = nn.Linear(128, 2)
        self.critic = nn.Linear(128, 1)
    
    def forward(self, input):
        output = F.relu(self.layer1(input))
        actor_pr = F.softmax(self.actor(output), dim=-1)
        state_value = self.critic(output)

        return actor_pr, state_value