# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 10:01:12 2019

@author: DEEPSHIKHA
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 08:58:03 2019

@author: DEEPSHIKHA
"""

import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable


class Neural_Network(nn.Module):
    
    def __init__(self, input_size, number_of_action):
        super(Neural_Network, self).__init__()
        self.input_size = input_size
        self.number_of_action = number_of_action
        self.fullconnection1 = nn.Linear(input_size, 25)#full connection between input and hidden layer
        self.fullconnection2 = nn.Linear(25, number_of_action)#full connection between hidden layer and output layer
    
    def forward(self, state):#forward the process and return the output of the network
        hid_neuron = F.relu(self.fullconnection1(state))#activating hidden neuron
        q_values = self.fullconnection2(hid_neuron)#getting the q values
        return q_values


class Replay_Memory(object):
    
    def __init__(self, capacity):#number of action that to be saved in the memory
        self.capacity = capacity
        self.memory = []
    
    def push(self, event):#appending the event into the memory
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]#deleting the oldest transition
    
    def sample(self, batch_size):
        samples = zip(*random.sample(self.memory, batch_size))#choosing the batch size from memory to train
                                                              #zip with * is used for reshaping 
                                                              #list ((1,2,3),(4,5,6))state,action,reward
                                                              #after zip* it will become ((1,4),(2,5),(3,6))
        #returning the variable in torch tensor variable
        return map(lambda x: Variable(torch.cat(x, 0)), samples)#cat for best allignment in one dimension


class Dqn():
    
    def __init__(self, input_size, nb_action, gamma):#gamma parameter - delay coefficient
        self.gamma = gamma
        self.reward_window = []
        self.model = Neural_Network(input_size, nb_action)
        self.memory = Replay_Memory(100000)
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001)#stocastistic gradient descent
                                                                        #it will connect the adam optimizer with model
        self.last_state = torch.Tensor(input_size).unsqueeze(0)#one dimension
        self.last_action = 0
        self.last_reward = 0
    
    def select_action(self, state):
        probs = F.softmax(self.model(Variable(state, volatile = True))*100) # T=100 Temperature parameter
        #now state is the tensor so we have to convert it into the variable
        #volatile for removing gradient descent
        #probability is very small so convert it into the higher values
        action = probs.multinomial(num_samples = 1)
        return action.data[0,0]
    
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        target = self.gamma*next_outputs + batch_reward
        td_loss = F.smooth_l1_loss(outputs, target)
        self.optimizer.zero_grad()
        td_loss.backward()
        self.optimizer.step()
    
    def update(self, reward, new_signal):
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
        action = self.select_action(new_state)
        if len(self.memory.memory) > 100:
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        return action
    
    def score(self):
        return sum(self.reward_window)/(len(self.reward_window)+1.)
    
    def save(self):
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer' : self.optimizer.state_dict(),
                   }, 'last_brain.pth')
    
    def load(self):
        if os.path.isfile('last_brain.pth'):
            print("=> loading checkpoint... ")
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("done !")
        else:
            print("no checkpoint found...")
