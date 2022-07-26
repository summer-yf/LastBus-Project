import torch.nn as nn
import torch.nn.functional as F
import torch

# JUST FOR REFERENCING 
# class ActorNetwork3Layer(nn.Module):
#     def __init__(self):
#         super(ActorNetwork3Layer, self).__init__()
#         # input image is 90 x 90
#         self.conv1 = nn.Conv2d(4, 5, 4)
#         self.pool1 = nn.MaxPool2d(3)
#         self.conv2 = nn.Conv2d(5, 10, 6, 4)
#         self.fc1 = nn.Linear(360, 64)
#         self.fc2 = nn.Linear(64, 64)
#         self.fc3 = nn.Linear(64, 2)

#     def forward(self, input):
#         output = self.pool1(F.relu(self.conv1(input)))
#         output = F.relu(self.conv2(output))
#         output = output.view(-1, 360)
#         output = F.relu(self.fc1(output))
#         output = F.relu(self.fc2(output))
#         output = F.relu(self.fc3(output))

#         return output

class ActorCriticModel():
    def __init__(self):
        super(ActorCriticModel, self).__init__()
        self.layer1 = nn.Linear(4, 128)

        # Actor
        self.actor = nn.Linear(128, 2)

        # Critic
        self.critic = nn.Linear(128, 1)
    
    def forward(self, input):
        output = F.relu(self.layer1(input))
        actor_pr = F.softmax(self.actor(output), dim=-1)
        critic_value = self.critic(output)

        return actor_pr, critic_value