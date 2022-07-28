from random import sample

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, state, action, next_state, reward, terminal):
        if len(self.memory) == self.capacity:
            del self.memory[0]
        self.memory.append([state, action, next_state, reward, terminal])
        return

    def sample(self, batch_size):
        batch = sample(self.memory, min(len(self.memory), batch_size))
        return batch