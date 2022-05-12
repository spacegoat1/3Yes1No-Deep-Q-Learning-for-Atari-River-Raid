import numpy as np
import tensorflow as tf


class Memory:
    def __init__(self, length):
        # self.data = [None] * (length + 1)
        self.state_mem = [None] * (length + 1)
        self.state_next_mem = [None] * (length + 1)
        self.action_mem = [None] * (length + 1)
        self.reward_mem = [None] * (length + 1)
        self.done_mem = [None] * (length + 1)
        self.start = 0
        self.end = 0
        
    def append(self, experience):
        state, state_next, action, reward, done = experience
        self.state_mem[self.end] = state
        self.state_next_mem[self.end] = state_next
        self.action_mem[self.end] = action
        self.reward_mem[self.end] = reward
        self.done_mem[self.end] = done
        self.end = (self.end + 1) % len(self.state_mem)
        if self.end == self.start:
            self.start = (self.start + 1) % len(self.state_mem)
        
    def sample_memory(self, size):
        indices = np.random.choice(range(len(self)), size=size)
        state_sample = np.array([self.state_mem[i] for i in indices])
        state_next_sample = np.array([self.state_next_mem[i] for i in indices])
        rewards_sample = [self.reward_mem[i] for i in indices]
        action_sample = [self.action_mem[i] for i in indices]
        done_sample = tf.convert_to_tensor([float(self.done_mem[i]) for i in indices])
        return state_sample, state_next_sample, action_sample, rewards_sample, done_sample

    def __len__(self):
        if self.end < self.start:
            return self.end + len(self.state_mem) - self.start
        else:
            return self.end - self.start
        
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

