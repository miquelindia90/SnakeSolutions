from progress.bar import Bar

import torch
from numpy.random import random_sample
import matplotlib.pyplot as plt


def sliding_list_average(list: list, sliding_window: int = 150) -> list:
    '''Compute the sliding average of a list.
    Args: list (list): List to compute the sliding average
          sliding_window (int): Sliding window size
    Returns: list: Sliding average of the list
    '''
    return [sum(list[max(0,i-sliding_window):i])/sliding_window for i in range(len(list))]

class RlTrainer:
    ''' Class that trains a DNN to play snake game using Reinforcement Learning '''
    def __init__(self, env, dnn, model_name, learning_rate=0.0001):
        '''Initialize the trainer.
        
        Args: env (SnakeEnv): Snake environment
              dnn (DNN): DNN to train
              learning_rate (float): Learning rate for the optimizer
        '''
        self.env = env
        self.dnn = dnn

        self.model_name = model_name

        self.episodes = 500_000
        self.plot_frequency = self.episodes//1000
        self.batch_size = 20
        self.epsilon = 0.5
        self._init_optimizer(learning_rate)

    def _init_optimizer(self, learning_rate: float):
        '''Initialize the optimizer.'''
        self.optimizer = torch.optim.Adam(self.dnn.parameters(), lr=learning_rate)

    def _init_training_variables(self):
        '''Initialize the training variables.'''
        self.loss = 0
        self.batch_count = 0

    def _init_train_metrics(self):
        '''Initialize the training metrics.'''
        self.movements_count = [0]*self.episodes
        self.scores = [0]*self.episodes
        self.rewards = [0]*self.episodes
        self.epsilons = [0]*self.episodes
        self.best_average_reward = -10000

    def _update_epsilon(self, episode: int):
        '''Update the epsilon value.
        Args: episode (int): Episode number'''
        self.epsilon = min(round(self.epsilon + 1/self.episodes, 8), 0.9)
        self.epsilons[episode] = float(self.epsilon)

    def _sample_action(self, states: list) -> tuple[int, torch.Tensor]:
        '''Sample an action from the DNN or randomly.
        Args: states (list): List of states
        Returns: tuple[int, torch.Tensor]: Action and the DNN logits
        '''
        snake_direction, food_distance, snake_body_danger, snake_wall_danger = states        
        snake_direction_tensor = torch.tensor(snake_direction).float().unsqueeze(0)
        food_distance_tensor = torch.tensor(food_distance).float().unsqueeze(0)
        snake_body_danger_tensor = torch.tensor(snake_body_danger).float().unsqueeze(0)
        snake_wall_danger_tensor = torch.tensor(snake_wall_danger).float().unsqueeze(0)
        dnn_logits = self.dnn(snake_direction_tensor, food_distance_tensor, snake_body_danger_tensor, snake_wall_danger_tensor)
        action = torch.argmax(dnn_logits).item() if random_sample() < self.epsilon else self.env.action_space.sample()
        return action, dnn_logits

    def _compute_loss(self, action: int, dnn_logits: torch.tensor, reward: int) -> torch.tensor:
        '''Compute the loss for the DNN.
        Args: action (int): Action taken
              dnn_logits (torch.tensor): DNN logits
              reward (int): Reward obtained
        Returns: torch.tensor: Loss
        '''
        logit = dnn_logits[0][action]
        return -logit * torch.tensor(reward).float()
    
    def _update_weights(self):
        '''Update the weights of the DNN.'''

        self.loss /= self.batch_count
        self.loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.loss = 0
        self.batch_count = 0

    def _create_metrics_plot(self, episode: int):
        '''Create the metrics plot for the training.
        Args: episode (int): Episode number
        '''

        figure, axes = plt.subplots(2, 2, figsize=(30, 10))
        
        axes[0][0].plot(list(range(episode)), self.scores[:episode], label="Score")
        axes[0][0].plot(list(range(episode)), sliding_list_average(self.scores[:episode]), label="Average Score")
        axes[0][0].set_title("Score")
        axes[1][0].plot(list(range(episode)), self.rewards[:episode], label="Reward")
        axes[1][0].plot(list(range(episode)), sliding_list_average(self.rewards[:episode]), label="Average Reward")
        axes[1][0].set_title("Reward")
        axes[1][0].set_xlabel("Episode")
        axes[0][1].plot(list(range(episode)), self.movements_count[:episode], label="Movements")
        axes[0][1].plot(list(range(episode)), sliding_list_average(self.movements_count[:episode]), label="Average Movements")
        axes[0][1].set_title("Movements")
        axes[1][1].plot(list(range(episode)), self.epsilons[:episode], label="Epsilon")
        axes[1][1].set_title("Epsilon")
        axes[1][1].set_xlabel("Episode")
        
        plt.savefig("logs/{}_metrics.png".format(self.model_name))
        plt.close()

    def _log_metrics(self, episode: int, movements_count: int, score: int, reward: int):
        '''Log the metrics for the training.
        Args: episode (int): Episode number
              movements_count (int): Movements count
              score (int): Score
        '''
        self.movements_count[episode] = movements_count
        self.scores[episode] = score
        self.rewards[episode] = float(reward)
        if episode % self.plot_frequency == 0:
            self._create_metrics_plot(episode)

    def _save_model(self, episode: int):
        '''Save the model.
        Args: episode (int): Episode number
        '''
        if episode > 150:
            current_average_reward = sliding_list_average(self.rewards[:episode])[-1]
            if current_average_reward > self.best_average_reward:
                torch.save(self.dnn.state_dict(), "models/model_{}.pth".format(self.model_name))
                self.best_average_reward = current_average_reward

    def _train_episode(self):
        '''Train an episode.
        Returns: tuple[int, int]: Movements count and score obtained.
        '''

        states = self.env.reset()
        done = False
        movements_count = 0
        rewards = 0
        while not done:
            action, dnn_logits = self._sample_action(states)
            states, reward, _, _, done = self.env.step(action)
            rewards += reward
            self.loss += self._compute_loss(action, dnn_logits, reward)
            self.batch_count += 1
            movements_count += 1
            
        return movements_count, self.env.score, rewards
             
    def train(self):
        '''Train the DNN.'''

        self._init_training_variables()
        self._init_train_metrics()
        with Bar("Training...", max=self.episodes) as bar: 
            for episode in range(self.episodes):
                movements_count, score, rewards = self._train_episode()
                if episode % self.batch_size == 0:
                    self._update_weights()
                self._log_metrics(episode, movements_count, score, rewards)
                self._save_model(episode)
                self._update_epsilon(episode)
                bar.next()

    def test(self, games: int = 1):
        '''Test the DNN.
        Args: games (int): Number of games to play        '''

        self.dnn.load_state_dict(torch.load("models/model_" +self.model_name + ".pth"))
        self.dnn.eval()
        self.epsilon = 1
        for _ in range(games):
            states = self.env.reset()
            done = False
            while not done:
                action, _ = self._sample_action(states)
                satates, _, _, _, done = self.env.step(action)
                self.env.render()
        self.env.close()