import gym
import numpy as np
import random
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.optimizers import Adam
from collections import deque

class DQN(object):
    def __init__(self, num_episodes=500, num_steps=200, lr=0.1, min_explore=0.05, discount=0.95, batch_size=32, decay=0.995):
        self.num_episodes = num_episodes
        self.num_steps = num_steps
        self.lr = lr
        self.min_explore = min_explore
        self.discount = discount
        self.batch_size = batch_size
        self.decay = decay
        self.env = gym.make('BreakoutDeterministic-v4')
        self.memory=deque(maxlen=2000)
        self.explore_rate = self.get_explore_rate(0)
        self.model=self.create_model()
        print(self.model.summary())
        self.target_model = self.create_model()
    
    def create_model(self):
        model = Sequential()
        
        state_shape = self.env.observation_space.shape
        input_shape = state_shape[:-1] + (1,)
        
        model.add(Conv2D(32, (8,8), strides=(4,4), activation='relu', input_shape=input_shape))
        model.add(Conv2D(64, (4,4), strides=(2,2), activation='relu'))
        model.add(Conv2D(64, (3,3), strides=(1,1), activation='relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.env.action_space.n, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.lr))
        return model
    
    def _preprocess(self, img):
        processed = img.mean(axis=2)
        processed = processed[..., np.newaxis]
        processed = processed[np.newaxis, ...]
        return processed
    
    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])
    
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
        
    def replay(self):
        if len(self.memory) < self.batch_size():
            return
        samples = random.sample(self.memory, self.batch_size)
        for sample in samples:
            state, action, reward, new_state, done = sample
            target = reward
            if not done:
                target = reward + self.discount * np.amax(self.target_model.predict(new_state)[0])
            target_fn = self.target_model.prdict(state)
            target_fn[0][action] = target
            self.model.fit(state, target_fn, epochs=1, verbose=0)
    
    def get_explore_rate(self, e):
        return max(self.min_explore, np.exp(-e * self.decay))
    
    def choose_action(self, state):
        if np.random.random < self.explore_rate:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.model.predict(state)[0])
    
    def train(self):
        for episode in range(self.num_episodes):
            current_state = self.env.reset()
            current_state = self._preprocess(current_state)
            self.explore_rate = self.get_explore_rate(episode)
            done = False
            while not done:
                action = self.choose_action(current_state)
                new_state, reward, done, _ = self.env.step(action)
                new_state = self._preprocess(new_state)
                self.remember(current_state, action, reward, new_state, done)
                current_state = new_state
            self.replay()
            self.update_target_model()
            print('Episode ({:02} / {})'.format(episode+1, self.episodes))
            if(episode+1%100 == 0):
                self.model.save('model.h5')
        self.model.save('model.h5')
            
    def run(self):
        self.model = load_model('model.h5')
        self.explore_rate = self.min_explore
        current_state = self.env.reset()
        current_state = self._preprocess(current_state)
        done = False
        while not done:
            self.env.render()
            action = self.choose_action(current_state)
            new_state, reward, done, _ = self.env.step(action)
            new_state = self._preprocess(new_state)
            current_state = new_state
            
if __name__ == '__main__':
    agent = DQN()
    agent.train()
    agent.run()
    
    
