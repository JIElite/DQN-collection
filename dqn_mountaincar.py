# 這是自己 implement 的演算法，並且最後有成功讓 Mountain Car 學起來

from collections import deque
import random
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt

env = gym.make('MountainCar-v0')
np.random.seed(0)

N_STATES = env.observation_space.shape[0]
N_ACTIONS = env.action_space.n - 1
LR = 0.0025
LR_DECAY = 0.01
MAX_EPISODES = 10000
MEM_SIZE = 100000
EPSILON = 1.0
MIN_EPSILON = 0.01
EPSILON_DECAY = 0.95
BATCH_SIZE = 64
GAMMA = 0.999
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
        return len(self.memory) == self.memory.maxlen


class DQN:
    def __init__(self, input_size=N_STATES, output_size=N_ACTIONS, lr=0.01, lr_decay=0.01, gamma=0.9,
        epsilon=1.0, min_epsilon=0.1, epsilon_decay=0.99, mem_size=10000, batch_size=32):
        self.evaluate_net = Network(input_size, output_size).cuda()
        self.target_net = Network(input_size, output_size).cuda()
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
        batch_s = Variable(FloatTensor(sampled_s))
        batch_s_ = Variable(FloatTensor(sampled_s_))
        batch_a = Variable(LongTensor(sampled_a)).unsqueeze(1)
        batch_r = Variable(FloatTensor(sampled_r)).unsqueeze(1)
        sampled_done_trans = [not done for done in sampled_done]
        batch_done = Variable(FloatTensor(sampled_done_trans)).unsqueeze(1)

        batch_q_eval = self.evaluate_net(batch_s).gather(1, batch_a)
        batch_q_next = self.target_net(batch_s_).detach()
        batch_q_target = batch_r + batch_done*GAMMA*batch_q_next.max(1)[0].unsqueeze(1)

        self.optimizer.zero_grad()
        loss = self.loss_function(batch_q_eval, batch_q_target)
        loss.backward()
        self.optimizer.step()

        if self.iter_counter % ITER_UPDATE_TARGET == 0:
            self.target_net.load_state_dict(self.evaluate_net.state_dict())
        
    def _transform_state(self, s):
        state_transformed = Variable(FloatTensor(s).unsqueeze(0))
        return state_transformed

    def decay_epsilon(self):
        new_epsilon = self.epsilon * self.epsilon_decay
        if new_epsilon < self.min_epsilon:
            self.epsilon = self.min_epsilon
        else:
            self.epsilon = new_epsilon

def transform_reward(env, s_):
    x, x_dot, theta, theta_dot = s_
    r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
    r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
    r = r1 + r2
    return r


agent = DQN(
    input_size=N_STATES,
    output_size=N_ACTIONS,
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
    step = 0
    while not done: 
        env.render()
        # print(s)
        step += 1
        action = agent.select_action(s)
        transformed_action = 2 if action == 1 else 0
        s_, reward, done, info = env.step(transformed_action)
        # reward = s_[1]
        agent.store_transition(s, action, reward, s_, done)
        s = s_
        agent.learn()
        R += reward
        if done:
            return_list.append(R)
            print("Episode:", i_episode, "R:", R)

    agent.decay_epsilon()

plt.plot(return_list)
plt.show()