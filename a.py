import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import random

class DQLAgent:
    def __init__(self, env):
        self.state_size = env.observation_space.shape[0] # notlarda yazan observations 0,1,2,3
        self.action_size = env.action_space.n # notlarda yazan eylemler 0,1

        self.gamma = 0.95 # gelecekteki ödüle mi odaklanmalıyım yoksa elimdeki ile mi yetinmeliyim
        self.learning_rate = 0.001 # öğrenme hızını belirler
        # bu 2 değerde dqn denkleminin parametreleridir

        self.epsilon = 1 # başlangıçta her yeri ara demek, her yeri keşfedebilirsin
        self.epsilon_decay = 0.995 # epsilonu azaltacağız sürekli yeni yer aramasın, epsilon ile bu değeri çarpacağız
        self.epsilon_min = 0.01 # buraya kadar düşecek, burdan aşağı düşmeyecek
                                # böylece her zaman yeni yerleri arama ihtimali olacak

        self.memory = deque(maxlen = 1000) # bu liste dolarsa ilk giren atılır son giren eklenir
        #bu hafızamız oluyor

        self.model = self.build_model() # ajanın yapay sinir ağı modeli (resimden bakabilirsin)

    def build_model(self):
        # ann modeli burada tanımlanacak
        model = Sequential() # keras ile ann ekleniyor
        model.add(Dense(48, input_dim = self.state_size, activation="tanh")) # ANN'ye layer ekliyoruz
        # Bu katmanda 48 nöron olacak
        # aktivasyon fonksiyonu da tanh olsun
        model.add(Dense(self.action_size)) # ANN'ye output player ekliyoruz
        # sonuç olarak ANN'nin çıktısı eylem ne yapılacaksa o olacak
        model.compile(loss="mse", optimizer=Adam(lr=self.learning_rate))
            # “Loss” gerçek değerin tahmin değerinden ne kadar farklı olduğunu göstermektedir.
            # MSE (Mean Squared Error) yani Ortalama Standart Hata en çok kullanılan “Loss” metriklerin biridir.
            # Ortalama standart hata gerçek (doğru) ve tahmin değerleri arasındaki farkı gösteren bir metriktir.
                # Bu bağlamda, hesaplanırken tüm gerçek değerler tahmin edilen değerilerden çıkartılarak karesi alınır ve toplanır.

            # optimizer olarak adaptive momentum kullanılıyor
            # https://keras.io/api/optimizers/adam/ - https://arxiv.org/abs/1412.6980
        return model

    def remember(self, state, action, reward, next_state, done):
        # hafıza -> bilgileri burada depolayacağız
        self.memory.append(state, action, reward, next_state, done)

    def act(self, state):
        # eylem
        pass

    def replay(self, batch_size): #batch_size ->kaç tane geçmiş veriden faydalanacağımız
        #eğitim
        pass

    def adaptiveEGreedy(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

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