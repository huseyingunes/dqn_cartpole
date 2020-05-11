import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import random

class DQLAgent:
    def __init__(self, env):
        self.state_size =

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
    agent = DQLAgent(env) # ortam için ajanı üret ve ilk pozisyonunu al

    batch_size = 16
    episodes = 100
    for e in range(episodes):
        #initialize environment
        state = env.reset()
        state = np.reshape(state, [1, 4])
        time = 0 # zaman en kadar çok geçerse o kadar balarılı çünkü her zamanda 1 ödül alacak
        while True:
            #act
            action = agent.act(state) # eylemi seç

            #step
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, 4])

            #remember
            agent.remember(state, action, reward, next_state, done)

            #update state
            state = next_state

            #replay
            agent.replay(batch_size) # rastgele batch_size ı kullanacak

            #adjust epsilon
            agent.adaptiveEGreedy()

            time += 1

            if done:
                print("Episode: {}, time: {}".format(e, time))
                break