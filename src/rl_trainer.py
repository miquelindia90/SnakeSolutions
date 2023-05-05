import torch
from numpy.random import random_sample

class RlTrainer:
    def __init__(self, env, dnn, learning_rate=0.0001):
        self.env = env
        self.dnn = dnn

        self.episodes = 1000
        self.epsilon = 0.5
        self._init_optimizer(learning_rate)  

    def _init_optimizer(self, learning_rate):
        self.optimizer = torch.optim.Adam(self.dnn.parameters(), lr=learning_rate)

    def _sample_action(self, states):
        states_tensor = torch.tensor(states).float().flatten().unsqueeze(0)
        dnn_logits = self.dnn(states_tensor)
        action = torch.argmax(dnn_logits).item() if random_sample() > self.epsilon else self.env.action_space.sample()
        return action, dnn_logits

    def _compute_loss(self, action, dnn_logits, reward):
        logit = dnn_logits[0][action]
        return logit * torch.tensor(reward).float()
    
    def _update_weights(self, loss):        
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def _train_episode(self):

        states = self.env.reset()
        done = False
        self.optimizer.zero_grad()
        while not done:
            action, dnn_logits = self._sample_action(states)
            states, reward, _, _, done = self.env.step(action)
            self.env.render()
            loss = self._compute_loss(action, dnn_logits, reward)
            self._update_weights(loss)

    def train(self):

        for episode in range(self.episodes):
            self._train_episode()
            print(f"Episode: {episode}")
            