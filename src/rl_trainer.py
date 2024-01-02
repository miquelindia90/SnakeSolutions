from progress.bar import Bar

import random
import math
import torch
from torch import nn
from numpy.random import random_sample
import matplotlib.pyplot as plt


def sliding_list_average(list: list, sliding_window: int = 150) -> list:
    '''Compute the sliding average of a list.
    Args: list (list): List to compute the sliding average
          sliding_window (int): Sliding window size
    Returns: list: Sliding average of the list
    '''
    return [sum(list[max(0,i-sliding_window):i])/sliding_window for i in range(len(list))]

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = []
    
    def add_experience(self, experience):
        if len(self.buffer) >= self.buffer_size:
            self.buffer.pop(0)
        self.buffer.append(experience)
    
    def sample_batch(self, batch_size):
        return random.sample(self.buffer, batch_size)

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

        self.max_episodes = 100_000
        self.plot_frequency = self.max_episodes//100
        self.batch_size = 10
        self.gamma = 0.95
        self.initial_epsilon = 0.5  # Initial epsilon value
        self.target_epsilon = 0.01  # Maximum epsilon value
        self.epsilon_converge_ratio = 2/3  # Ratio for epsilon convergence
        self._init_epsilon()        
        self.buffer_size = 3000
        self.replay_buffer = ReplayBuffer(buffer_size=self.buffer_size)
        self._init_optimizer(learning_rate)

    def _init_optimizer(self, learning_rate: float):
        '''Initialize the optimizer.'''
        self.optimizer = torch.optim.Adam(self.dnn.parameters(), lr=learning_rate)

    def _init_epsilon(self):
        '''Initialize the epsilon value.'''
        self.epsilon = self.initial_epsilon

    def _init_training_variables(self):
        '''Initialize the training variables.'''
        self.loss = 0
        self.batch_count = 0

    def _init_train_metrics(self):
        '''Initialize the training metrics.'''
        self.movements_count = [0]*self.max_episodes
        self.scores = [0]*self.max_episodes
        self.rewards = [0]*self.max_episodes
        self.epsilons = [0]*self.max_episodes
        self.best_average_reward = -10000

    def _update_epsilon(self, episode: int):
        '''Update the epsilon value.
        Args: episode (int): Episode number'''
        decay_episodes = self.max_episodes * self.epsilon_converge_ratio
        decay_rate = -math.log(self.target_epsilon / self.initial_epsilon) / decay_episodes
        self.epsilon = max(self.target_epsilon, self.initial_epsilon * math.exp(-decay_rate * episode))
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
        action = torch.argmax(dnn_logits).item() if random_sample() > self.epsilon else self.env.action_space.sample()
        return action, dnn_logits

    def _compute_loss(self, q_value: torch.tensor, target_q_value: torch.tensor) -> torch.tensor:
        '''Compute the loss for the DNN.
        Args: q_value (torch.tensor): Q-value predicted by the network
          target_q_value (torch.tensor): Target Q-value based on the Bellman equation
        Returns: torch.tensor: Loss
        '''
        return nn.MSELoss()(q_value, target_q_value)
    
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
            action, _ = self._sample_action(states)
            prev_states = [torch.tensor(state).float().unsqueeze(0) for state in states]
            states, reward, _, _, done = self.env.step(action)

            # Calculate target Q-value using the Bellman equation
            next_states =[torch.tensor(state).float().unsqueeze(0) for state in states]
            self.replay_buffer.add_experience((prev_states, action, reward, next_states, done))
            target_q_value = reward + self.gamma * torch.max(self.dnn(*next_states))

            # Calculate the loss and update the Q-network
            q_value = self.dnn(*prev_states)[0][action]
            self.loss += self._compute_loss(q_value, target_q_value)
            self.batch_count += 1
            movements_count += 1
            rewards += reward
            

        return movements_count, self.env.score, rewards
    
    def _apply_buffer_replay(self):
        batch = self.replay_buffer.sample_batch(self.batch_size)
        for experience_idx, experience in enumerate(batch):
            prev_states, action, reward, next_states, done = experience

            #Calculate target Q-value using the Bellman equation           
            target_q_value = reward + self.gamma * torch.max(self.dnn(*next_states))
            q_value = self.dnn(*prev_states)[0][action]
            loss = self._compute_loss(q_value, target_q_value)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()                
             
    def train(self):
        '''Train the DNN.'''

        self._init_training_variables()
        self._init_train_metrics()
        with Bar("Training...", max=self.max_episodes) as bar: 
            for episode in range(self.max_episodes):
                movements_count, score, rewards = self._train_episode()
                if episode % self.batch_size == 0:
                    self._update_weights()
                    if len(self.replay_buffer.buffer) >= self.batch_size:
                        self._apply_buffer_replay()
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
                states, _, _, _, done = self.env.step(action)
                self.env.render()
        self.env.close()