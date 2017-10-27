from collections import deque
import random
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt

env = gym.make('CartPole-v0')
np.random.seed(0)

N_STATES = env.observation_space.shape[0]
N_ACTIONS = env.action_space.n
LR = 0.005
LR_DECAY = 0.01
MAX_EPISODES = 1000
MEM_SIZE = 10000
EPSILON = 1.0
MIN_EPSILON = 0.05
EPSILON_DECAY = 0.995
BATCH_SIZE = 64
GAMMA = 0.9
ITER_UPDATE_TARGET = 300

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor
    

class Network(nn.Module):
    def __init__(self, input_size, output_size):
        super(Network, self).__init__()
        self.layer1 = nn.Linear(input_size, 24)
        self.layer2 = nn.Linear(24, 48)
        self.output = nn.Linear(48, output_size)
    
    def forward(self, x):
        x = F.tanh(self.layer1(x))
        x = F.tanh(self.layer2(x))
        x = self.output(x)
        return x

class Memory:
    def __init__(self, maxlen=10000):
        self.memory = deque(maxlen=maxlen)

    def append(self, s, r, a, s_, done):
        self.memory.append((s, r, a, s_, done))

    def sample(self, batch_size=32):
        batch_size = min(batch_size, len(self.memory))
        sampled_experience = random.sample(self.memory, batch_size)
        sampled_s = []
        sampled_r = []
        sampled_a = []
        sampled_s_ = []
        sampled_done = []
        for (s, r, a, s_, done) in sampled_experience:
            sampled_s.append(s)
            sampled_r.append(r)
            sampled_a.append(a)
            sampled_s_.append(s_)
            sampled_done.append(done)
        return sampled_s, sampled_r, sampled_a, sampled_s_, sampled_done

    def isFull(self):
        return len(self.state) == self.state.maxlen


class DQN:
    def __init__(self, input_size=N_STATES, output_size=N_ACTIONS, lr=0.01, lr_decay=0.01, gamma=0.9,
        epsilon=1.0, min_epsilon=0.1, epsilon_decay=0.99, mem_size=10000, batch_size=32):
        self.evaluate_net = Network(input_size, output_size)
        self.target_net = Network(input_size, output_size)
        self.optimizer = torch.optim.Adam(self.evaluate_net.parameters(), lr=lr, weight_decay=lr_decay)
        self.loss_function = nn.MSELoss()
        
        self.memory = Memory(maxlen=mem_size) 
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.gamma = gamma

        self.iter_counter = 0

    def select_action(self, s):
        if self.epsilon > np.random.uniform():
            # epsilon
            action = np.random.choice([0, 1]) 
        else:
            # greedy
            state = self._transform_state(s)
            Q = self.evaluate_net(state)
            action = torch.max(Q, 1)[1].data[0]

        return int(action)

    def store_transition(self, s, action, reward, s_, done):
        self.memory.append(s, action, reward, s_, done)

    def learn(self):
        self.iter_counter += 1 
        
        sampled_s, sampled_a, sampled_r, sampled_s_, sampled_done = self.memory.sample(batch_size=self.batch_size) 
        batch_s = Variable(torch.FloatTensor(sampled_s))
        batch_a = Variable(torch.LongTensor(sampled_a)).unsqueeze(1)
        batch_q_eval = self.evaluate_net(batch_s).gather(1, batch_a)

        target_list = []
        for s_, r, done in zip(sampled_s_, sampled_r, sampled_done):
            target = r
            if not done:
                target += self.gamma*self.target_net(self._transform_state(s_)).max().data[0]
            target_list.append(target)
        batch_q_target = Variable(torch.FloatTensor(target_list).unsqueeze(1))

        self.optimizer.zero_grad()
        loss = self.loss_function(batch_q_eval, batch_q_target)
        loss.backward()
        self.optimizer.step()

        if self.iter_counter % ITER_UPDATE_TARGET == 0:
            self.target_net.load_state_dict(self.evaluate_net.state_dict())
        
    def _transform_state(self, s):
        state_transformed = Variable(torch.FloatTensor(s).unsqueeze(0))
        return state_transformed

    def decay_epsilon(self):
        new_epsilon = self.epsilon * self.epsilon_decay
        if new_epsilon < self.min_epsilon:
            self.epsilon = self.min_epsilon
        else:
            self.epsilon = new_epsilon


agent = DQN(
    input_size=N_STATES,
    output_size=2,
    lr=LR,
    lr_decay=LR_DECAY,
    epsilon=EPSILON,
    min_epsilon=MIN_EPSILON,
    epsilon_decay=EPSILON_DECAY,
    gamma=GAMMA,
    mem_size=MEM_SIZE,
    batch_size=BATCH_SIZE,
)

return_list = []
for i_episode in range(MAX_EPISODES):
    s = env.reset()
    R = 0
    done = False
    while not done: 
        # env.render()
        action = agent.select_action(s)
        s_, reward, done, info = env.step(action)
        agent.store_transition(s, action, reward, s_, done)
        s = s_
        agent.learn()
        
        R += reward
        if done:
            return_list.append(R)
            print("Episode:", i_episode, "Epsilon:", agent.epsilon, "R", R)

    agent.decay_epsilon()

plt.plot(return_list)
plt.show()