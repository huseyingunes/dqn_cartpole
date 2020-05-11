import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import random

class DQLAgent:
    def __init__(self, env):
        pass

    def build_model(self):
        # ann modeli burada tanımlanacak
        pass

    def remember(self, state, action, reward, next_state, done):
        # hafıza -> bilgileri burada depolayacağız
        pass

    def act(self, state):
        # eylem
        pass

    def replay(self, batch_size): #batch_size ->kaç tane geçmiş veriden faydalanacağımız
        #eğitim
        pass

    def adaptiveEGreedy(self):
        pass

if __name__ == "main":

    env = gym.make("CartPole-v0")
    agent = DQLAgent(env)

    episodes = 100
    for e in range(episodes):
        #initialize environment
        state = env.reset()
        time = 0 # zaman en kadar çok geçerse o kadar balarılı çünkü her zamanda 1 ödül alacak
        while True:

            #act

            #step

            #remember

            #update step

            #replay

            #adjust epsilon

            if done:
                break