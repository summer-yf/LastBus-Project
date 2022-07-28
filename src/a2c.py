import torch.nn as nn
import torch.nn.functional as F
import torch

class ThreeLayerConvModel(nn.Module):
    def __init__(self):
        super(ThreeLayerConvModel, self).__init__()
        # input image is 90 x 90
        self.conv1 = nn.Conv2d(4, 32, 8, 4)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        self.fc1 = nn.Linear(3136, 512)
        # Output 2 values: fly up and do nothing
        self.fc2 = nn.Linear(512, 2)
        self.create_weightss()
        self.relu = nn.ReLU(inplace=True)

    def create_weightss(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.01, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, input):
        x = self.relu(self.conv1(input))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        # Flatten output to feed into fully connected layers
        x = x.view(x.size()[0], -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x
    
class QNetwork(nn.Module):

    def __init__(self):
        super(QNetwork, self).__init__()
        
        # Discount factor
        self.gamma = 0.99
        # Epsilon values for ϵ greedy exploration
        self.initial_epsilon = 0.1
        self.final_epsilon = 0.0001
        self.replay_memory_size = 10000
        self.num_iterations = 2000000
        self.minibatch_size = 32
        self.episode_durations = []
        
        # Use gpu if it is availiable
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Use same network architecture as DeepMind
        # Input is 4 frames stacked to infer velocity
        self.conv1 = nn.Conv2d(4, 32, 8, 4)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)

        self.fc1 = nn.Linear(3136, 512)
        # Output 2 values: fly up and do nothing
        self.fc2 = nn.Linear(512, 2)
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        # Flatten output to feed into fully connected layers
        x = x.view(x.size()[0], -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x