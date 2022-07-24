from random import sample

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        if len(self.memory) == self.capacity:
            del self.memory[0]
        self.memory.append(transition)
        return

    def sample(self, batch_size):
        sample(self.memory, min(len(self.memory), batch_size))
        return

    def __len__(self):
        return len(self.memory)