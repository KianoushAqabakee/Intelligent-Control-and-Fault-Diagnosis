
import gymnasium as gym
import scipy.io as sc
import numpy as np
from gymnasium import spaces
import matplotlib.pyplot as plt
from IPython import display
import pandas as pd

plt.style.use('fivethirtyeight')
plt.rc('font', size=10)

batch_size = 1#500

class Dataset_A_env(gym.Env):
    def __init__(self, max_steps=500, make_data = False):
        super(Dataset_A_env, self).__init__()
        self.observation_shape = (784,)
        self.observation_space = spaces.Box(low=np.zeros(self.observation_shape), high=np.ones(self.observation_shape), dtype=np.float64)
        self.action_space = spaces.Discrete(10)
        self.canvas = np.ones(self.observation_shape) * 1
        self.make_data = make_data
        # Define elements present inside the environment
        self.elements = []
        
        # Initialize your environment here
        self.max_steps = max_steps
        self.observations = []
            
        self.history = {'reward_sum':[]}
        self.reward_sum = -410
        plt.figure(1)
        plt.cla()
        

        self.reset()

    def reset(self, seed = 0):
        # Reset the environment and return the initial observation

        if self.make_data == True:
            data2 = sc.loadmat('data2.mat')['data2']
            DE = data2['DE'][0][0]
            FE = data2['FE'][0][0]
            label = data2['label'][0][0]

            self.X = np.double(DE.reshape([-1, 784]))
            self.Y = np.int64(label.reshape([-1, 784]).mean(1))
            
            MAX = self.X.max()
            MIN = self.X.min()
            self.X = (self.X - MIN) / (MAX - MIN)
            
            r_indexes = np.arange(self.X.shape[0])
            np.random.shuffle(r_indexes)
            self.X = self.X[r_indexes]
            self.Y = self.Y[r_indexes]
            train_size = int(self.X.shape[0]*.8)
            #test_size = self.X.shape[0]-train_size
            train_idx = np.random.randint(self.X.shape[0], size = train_size)
            train_data = {'x':self.X[train_idx],'y':self.Y[train_idx]}
            test_data = {'x':np.delete(self.X, train_idx, 0),'y':np.delete(self.Y, train_idx)}
            np.save('TrainData.npy', train_data, allow_pickle=True)
            np.save('TestData.npy', test_data, allow_pickle=True)
        else:
            temp = np.load('TrainData.npy', allow_pickle=True)
            self.X = temp.reshape([-1])[0]['x']
            self.Y = temp.reshape([-1])[0]['y']
        
        #r_indexes = np.arange(self.X.shape[0])
        #np.random.shuffle(r_indexes)

        self.ind = np.random.randint(self.X.shape[0], size = batch_size)
        self.state = self.X[self.ind]#.reshape([1500,1,1,-1])#[self.ind]
        return self.observation()[0], {}

    def observation(self):
        #return np.array([self.state[o] for o in self.observations])
        return (self.state)#.reshape([1,1,-1]))
    
    def step(self, action):
        # Take an action in the environment and return the next observation, reward, done flag, and additional information
        if action == self.Y[self.ind][0]:
            reward = 1
        else:
            reward = -1
        #reward = (action == self.Y[self.ind]).sum() - (action != self.Y[self.ind]).sum()
        self.reward_sum += reward
        
        self.ind = np.random.randint(self.X.shape[0], size = batch_size)
        self.state = self.X[self.ind]#.reshape([1500,1,1,-1])#[self.ind]
        
        #done = False
        #if self.Return >= self.max_steps:
        #    done = True
        done = True

        self.history['reward_sum'].append(self.reward_sum)
        return self.observation()[0], reward, done, bool(15) , {}
        #return self.state, self.Return, done, {}
    
    def render(self, mode='human'):
        #plt.figure(1)
        window=100
        display.clear_output(wait=True)
        #plt.figure(figsize=(8,4))
        reward_sum = self.history['reward_sum']
        rolling_mean = pd.Series(reward_sum).rolling(window).mean()
        std = pd.Series(reward_sum).rolling(window).std()

        plt.plot(reward_sum)
        plt.plot(rolling_mean)
        plt.fill_between(range(len(reward_sum)),rolling_mean-std, rolling_mean+std, color='violet', alpha=0.4)

        plt.grid('on')
        plt.xlabel('Episode')
        plt.ylabel('Score')
        plt.legend(['Score', 'Rolling Mean'])
        
        plt.pause(0.001)
        # Optional: Implement a method to visualize the environment
        pass
    
    def close(self):
        pass




'''
import gym
import scipy.io as sc
import numpy as np
from gym import spaces


class Dataset_A_env(gym.Env):
    def __init__(self, max_steps=1000):
        super(Dataset_A_env, self).__init__()
        self.observation_shape = (784)
        self.observation_space = spaces.Box(low = np.zeros(self.observation_shape), high = np.ones(self.observation_shape),dtype = np.float16)
        self.action_space = spaces.Discrete(10,)
        self.canvas = np.ones(self.observation_shape) * 1
        # Define elements present inside the environment
        self.elements = []
        
        # Initialize your environment here
        self.max_steps = max_steps
        self.observations = [1, 2, 3]
        
        data2 = sc.loadmat('data2.mat')['data2']
        DE = data2['DE'][0][0]
        FE = data2['FE'][0][0]
        label = data2['label'][0][0]

        self.X = DE.reshape([-1, 784])
        self.Y = np.int64(label.reshape([-1, 784]).mean(1))
        #self.action_space.n = 10

        MAX = self.X.max()
        MIN = self.X.min()
        self.x = (self.x-MIN)/(MAX-MIN)

        self.reset()

    def observation(self):
        return np.array([self.state[o] for o in self.observations])
        
    def reset(self):
        # Reset the environment and return the initial observation

        self.Return = -410
        r_indexes = np.arange(self.X.shape[0])
        np.random.shuffle(r_indexes)

        self.X = self.X[r_indexes]
        self.Y = self.Y[r_indexes]

        self.ind = np.random.randint(self.X.shape[0])
        self.state = self.X[self.ind]
    def step(self, action):
        # Take an action in the environment and return the next observation, reward, done flag, and additional information
        if action == self.Y[self.ind]:
            self.Return += 1
        else:
            self.Return -= 1

        self.ind = np.random.randint(self.X.shape[0])
        self.state = self.X[self.ind]
    def render(self, mode='human'):
        # Optional: Implement a method to visualize the environment
        a=1
    def close(self):
        pass

'''
