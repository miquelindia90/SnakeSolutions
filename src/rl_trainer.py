import torch
from numpy.random import random_sample
import matplotlib.pyplot as plt


def sliding_list_average(list, sliding_window=150):
    return [sum(list[max(0,i-sliding_window):i])/sliding_window for i in range(len(list))]

class RlTrainer:
    def __init__(self, env, dnn, learning_rate=0.0001):
        self.env = env
        self.dnn = dnn

        self.episodes = 10_000
        self.batch_size = 256
        self.epsilon = 0.3
        self._init_optimizer(learning_rate)

    def _init_optimizer(self, learning_rate):
        self.optimizer = torch.optim.Adam(self.dnn.parameters(), lr=learning_rate)

    def _init_training_variables(self):
        self.loss = 0
        self.batch_count = 0

    def _init_train_metrics(self):

        self.movements_count = [0]*self.episodes
        self.scores = [0]*self.episodes

    def _update_epsilon(self):
        self.epsilon = min(round(min(1, self.epsilon + 0.0001), 4), 0.9)

    def _sample_action(self, states):
        states_tensor = torch.tensor(states).float().flatten().unsqueeze(0)
        dnn_logits = self.dnn(states_tensor)
        action = torch.argmax(dnn_logits).item() if random_sample() < self.epsilon else self.env.action_space.sample()
        return action, dnn_logits

    def _compute_loss(self, action, dnn_logits, reward):
        logit = dnn_logits[0][action]
        return -logit * torch.tensor(reward).float()
    
    def _update_weights(self):

        self.loss /= self.batch_size
        self.loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.loss = 0

    def _create_metrics_plot(self, episode):

        figure, axes = plt.subplots(2)
        axes[0].plot(list(range(episode)), self.movements_count[:episode], label="Movements")
        axes[0].plot(list(range(episode)), sliding_list_average(self.movements_count[:episode]), label="Average Movements")
        axes[0].set_title("Movements per episode")
        axes[1].plot(list(range(episode)), self.scores[:episode], label="Score")
        axes[1].set_title("Score per episode")

        for ax in axes:
            ax.label_outer()
        plt.savefig("logs/Movements.png")
        figure.clf()

    def _log_metrics(self, episode, movements_count, score):
        self.movements_count[episode] = movements_count
        self.scores[episode] = score
        if episode % 200 == 0:
            self._create_metrics_plot(episode)

    def _train_episode(self):

        states = self.env.reset()
        done = False
        movements_count = 0
        while not done:
            action, dnn_logits = self._sample_action(states)
            states, reward, _, _, done = self.env.step(action)
            #self.env.render()
            self.loss += self._compute_loss(action, dnn_logits, reward)
            self.batch_count += 1
            movements_count += 1
            if self.batch_count % self.batch_size == 0:
                self._update_weights()
        return movements_count, self.env.score
             
    def train(self):

        self._init_training_variables()
        self._init_train_metrics()
        for episode in range(self.episodes):
            movements_count, score = self._train_episode()
            self._log_metrics(episode, movements_count, score)
            self._update_epsilon()
            print(f"Episode: {episode}, Epsilon: {self.epsilon}, Movements: {movements_count}, Score: {score}")