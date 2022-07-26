import torch.nn as nn
import torch.nn.functional as F
import torch
#https://github.com/lambders/drl-experiments
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

class ActorCriticModel(nn.Module):

    def __init__(self):

        super(ActorCriticModel, self).__init__()

        #im not sure how many layers we want
        self.conv1 = nn.Conv2d(4, 16, 8, 4)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, 4, 2)
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(2592, 256)
        self.relu3 = nn.ReLU()

        self.actor = nn.Linear(256, 2)
        self.critic = nn.Linear(256, 1)

        self.softmax = nn.Softmax()
        self.logsoftmax = nn.LogSoftmax()
        self.create_weightss()


    def create_weightss(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.01, 0.01)
                nn.init.constant_(m.bias, 0)


    def forward(self, input):

        output = self.relu1(self.conv1(input))
        output = self.relu2(self.conv2(output))
        output = output.view(output.size()[0], -1)
        output = self.relu3(self.fc3(output))

        act_value = self.actor(output)
        crit_value = self.critic(output)
        # Careful which one is first
        return act_value, crit_value


"""
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
"""
