# Inspired by https://keon.io/deep-q-learning/

import random
import gym
import math
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib.pylab as plt


class DQNCartPoleSolver():
    def __init__(self, n_episodes=1000, n_win_ticks=195, max_env_steps=None, gamma=1.0, epsilon=1.0, epsilon_min=0.01, epsilon_log_decay=0.995, alpha=0.01, alpha_decay=0.01, batch_size=64, monitor=False, quiet=False):
        self.memory = deque(maxlen=100000)
        self.env = gym.make('CartPole-v0')
        self.env = self.env.unwrapped
        
        if monitor: self.env = gym.wrappers.Monitor(self.env, '../data/cartpole-1', force=True)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_log_decay
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.n_episodes = n_episodes
        self.n_win_ticks = n_win_ticks
        self.batch_size = batch_size
        self.quiet = quiet
        if max_env_steps is not None: self.env._max_episode_steps = max_env_steps

        # Init model
        self.model = Sequential()
        self.model.add(Dense(24, input_dim=4, activation='tanh'))
        self.model.add(Dense(48, activation='tanh'))
        self.model.add(Dense(2, activation='linear'))
        self.model.compile(loss='mse', optimizer=Adam(lr=self.alpha, decay=self.alpha_decay))
        self.episode_return = []

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state, epsilon):
        return self.env.action_space.sample() if (np.random.random() <= epsilon) else np.argmax(self.model.predict(state))

    def get_epsilon(self, t):
        return max(self.epsilon_min, min(self.epsilon, 1.0 - math.log10((t + 1) * self.epsilon_decay)))

    def preprocess_state(self, state):
        return np.reshape(state, [1, 4])

    def replay(self, batch_size):
        x_batch, y_batch = [], []
        minibatch = random.sample(
            self.memory, min(len(self.memory), batch_size))
        for state, action, reward, next_state, done in minibatch:
            y_target = self.model.predict(state)
            y_target[0][action] = reward if done else reward + self.gamma * np.max(self.model.predict(next_state)[0])
            x_batch.append(state[0])
            y_batch.append(y_target[0])
        
        self.model.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def get_returns(self):
        return self.episode_return

    def run(self):
        scores = deque(maxlen=100)

        for e in range(self.n_episodes):
            state = self.preprocess_state(self.env.reset())
            done = False
            i = 0
            R = 0
            while not done:
                # self.env.render()
                epsilon = self.get_epsilon(e)
                action = self.choose_action(state, epsilon)
                next_state, reward, done, _ = self.env.step(action)
                s_ = next_state
                next_state = self.preprocess_state(next_state)

                x, x_dot, theta, theta_dot = s_
                r1 = (self.env.x_threshold - abs(x)) / self.env.x_threshold - 0.8
                r2 = (self.env.theta_threshold_radians - abs(theta)) / self.env.theta_threshold_radians - 0.5
                reward = r1 + r2
                
                self.remember(state, action, reward, next_state, done)
                state = next_state
                i += 1
                R += reward
            print("Episode:", e,  "Return", R)
            self.replay(self.batch_size)
            # scores.append(i)
            # mean_score = np.mean(scores)
            # if mean_score >= self.n_win_ticks and e >= 100:
            #     if not self.quiet: print('Ran {} episodes. Solved after {} trials ✔'.format(e, e - 100))
            #     return e - 100
            # if e % 100 == 0 and not self.quiet:
            #     print('[Episode {}] - Mean survival time over last 100 episodes was {} ticks.'.format(e, mean_score))

            self.episode_return.append(R)
        
        # if not self.quiet: print('Did not solve after {} episodes 😞'.format(e))
        # return e

if __name__ == '__main__':
    agent = DQNCartPoleSolver()
    agent.run()

    episodes = len(agent.get_returns())
    plt.plot(range(episodes), agent.get_returns(), 'b')
    plt.show()