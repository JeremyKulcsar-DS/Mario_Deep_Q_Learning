# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 14:32:07 2020

@author: Lucas & Jeremy
"""

import random
import time
import retro
import tensorflow as tf
import threading
import cv2
import numpy as np
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D, Conv3D, MaxPooling3D
from timeit import default_timer as timer


env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

class Agent:
    def __init__(self, environment, optimizer):
        # Initialize atributes
        self._state_size = environment.observation_space.shape
        self._action_size = environment.action_space.n
        self._optimizer = optimizer
        
        # Initialize discount and exploration rate
        self.gamma = 0.9
        self.epsilon = 0.15
        
        # Build networks
        self.q_network = self._build_compile_model()
        self.target_network = self.q_network
        
        # Storing last 4 frames and target
        self.Batch = []
        self.Target = []

    def store(self, q_values, States, action, reward, next_state, terminated):
        target = q_values
        
        if reward < -14:
            target[action] = reward
        else:
            NextStates = np.roll(States, -1)
            NextStates[-1] = next_state
            t = self.target_network.predict(np.array([NextStates,]))
            target[action] = reward + self.gamma * np.max(t[0])

        self.Batch.append(States)
        self.Target.append(target)
    
    def _build_compile_model(self):
        model = Sequential()
        
        model.add(Conv2D(32, (8,8), strides=(4,4), activation='relu',input_shape=(4,100,100), data_format='channels_first'))
        
        model.add(Conv2D(64, (6,6), strides=(2,2), activation='relu', data_format='channels_first'))
        
        model.add(Conv2D(128, (4,4), strides=(1,1), activation='relu', data_format='channels_first'))
        
        model.add(Conv2D(256, (3,3), strides=(1,1), activation='relu', data_format='channels_first'))
         
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self._action_size))
        
        model.compile(loss='mse', optimizer=self._optimizer)
        
        return model
    
    def act(self, States):    
        q_values = self.q_network.predict(np.array([States,]))
        if random.random() < self.epsilon:               #adding noise to optimize agent exploration
            action = random.randint(0,self._action_size-1)
        else:
            action = np.argmax(q_values[0])
        return q_values[0], action
    
    def nojump(self, q_values):
        allowed_actions = [0,1,3,6]
        new_q = q_values[allowed_actions]
        if random.random() < self.epsilon:               #adding noise to optimize agent exploration
            action = random.choice(allowed_actions)
        else:
            action = allowed_actions[np.argmax(new_q)]
        return action
        

    def train(self):
        X_train = np.asarray(self.Batch)
        Y_train = np.asarray(self.Target)
        self.q_network.fit(X_train, Y_train, epochs=1, verbose=1, shuffle=True)

agent = Agent(env, 'adam')

from tensorflow.keras.models import load_model

agent.target_network = agent.q_network

size=(100, 100)

state = env.reset()

state = cv2.cvtColor(cv2.resize(env.reset(), dsize=size), cv2.COLOR_BGR2GRAY)
state = state.astype(float)

States = np.array([state for k in range(4)]) # Last 4 states list
Qvalue = 0

jump_test = np.array([1,2,3]) 
epsilon_change = 3600

x_pos = np.array([env.env.env._x_position, env.env.env._x_position])
clock = np.array([env.env.env._time, env.env.env._time])
is_dead = False

start = timer()
save = timer()

death_count = 0
training_time = 172800

for step in range(1, 1000000000000000):
    env.render()
    q_values, action = agent.act(States)
    
    # action 0 = idle
    # action 1 = walk right
    # action 2 = jump walk right
    # action 3 = run right
    # action 4 = jump run right
    # action 5 = jump
    # action 6 = walk left
    
    if jump_test[1] == jump_test[2] and jump_test[0] > jump_test[1]:
        action = agent.nojump(q_values)
      
    next_state, rew, terminated, info = env.step(action)   #terminated = game over or level finished
    
    x_pos = np.roll(x_pos, -1)
    x_pos[-1] = env.env.env._x_position
    
    clock = np.roll(clock, -1)
    clock[-1] = env.env.env._time
    
    is_dead = rew < -14
    
    reward= (2*(np.heaviside(x_pos[1]-x_pos[0],0) - np.heaviside(x_pos[0]-x_pos[1],0)) + clock[1] - clock[0]) * (not is_dead) - 100*is_dead
    Qvalue += reward
    
    jump_test = np.roll(jump_test, -1)
    jump_test[-1] = info.get('y_pos')
    
    # position
    x_pos = np.roll(x_pos, -1)
    x_pos[-1] = info.get('x_pos')
    
    
    next_state = cv2.cvtColor(cv2.resize(next_state, dsize=size), cv2.COLOR_BGR2GRAY)
    next_state = next_state.astype(float)
    agent.store(q_values, States, action, reward, next_state, terminated)
    
    States = np.roll(States, -1)
    States[-1] = next_state
        
    if reward < -14:                #Mario is dead
        state = env.reset()
        state = cv2.cvtColor(cv2.resize(env.reset(), dsize=size), cv2.COLOR_BGR2GRAY)
        state = state.astype(float)
        States = np.array([state for k in range(4)])
        
        print(Qvalue)
        Qvalue = 0
        jump_test = np.array([1,2,3])
        x_pos = np.array([env.env.env._x_position, env.env.env._x_position])
        clock = np.array([env.env.env._time, env.env.env._time])
        is_dead = False
        
        death_count+=1
    
    if step % 1000 == 0:    #Training network
        print('TRAINING')
        agent.train()
        agent.Batch = []
        agent.Target = []


    if step % 25000 == 0:    #Updating target network
        agent.target_network = agent.q_network
        
    if timer() > epsilon_change and timer() < 43200:
        agent.epsilon = agent.epsilon - 0.01
        epsilon_change = epsilon_change + 3600

        
    if timer() - start > training_time:
        print('Training Finished')
        print(step)
        break
    




