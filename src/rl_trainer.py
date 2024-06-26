import os
import random
import math
import time

import matplotlib.pyplot as plt
import torch
from torch import nn
from numpy.random import random_sample
from progress.bar import Bar


def sliding_list_average(list: list, sliding_window: int = 150) -> list:
    """Compute the sliding average of a list.

    Args:
        list (list): List to compute the sliding average
        sliding_window (int): Sliding window size

    Returns:
        list: Sliding average of the list
    """
    return [
        sum(list[max(0, i - sliding_window) : i]) / sliding_window
        for i in range(len(list))
    ]


class ReplayBuffer:
    """Class for storing and sampling experiences for reinforcement learning """

    def __init__(self, buffer_size: int = 1000) -> None:
        """
        Initialize the RLTrainer object.

        Args:
            buffer_size (int): The size of the buffer. Defaults to 1000.
        """
        self.buffer_size = buffer_size
        self.buffer = []

    def add_experience(self, experience: tuple) -> None:
        """
        Adds a new experience to the buffer.

        If the buffer is already full, the oldest experience will be removed.

        Args:
            experience(tuple): The experience to be added to the buffer.

        """
        if len(self.buffer) >= self.buffer_size:
            self.buffer.pop(0)
        self.buffer.append(experience)

    def sample_batch(self, batch_size: int) -> list:
        """
        Samples a batch of experiences from the buffer.

        Args:
            batch_size (int): The number of experiences to sample.

        Returns:
            list: A list of sampled experiences.

        """
        return random.sample(self.buffer, batch_size)


class RlTrainer:
    """ Class that trains a DNN to play snake game using Reinforcement Learning """

    def __init__(self, env, dnn, parameters):
        """Initialize the trainer.
        
        Args: env (SnakeEnv): Snake environment
              dnn (DNN): DNN to train
              parameters (dict): training configuration variables
        """
        self.env = env
        self.dnn = dnn

        self.model_name = parameters["model_name"]
        self.max_episodes = parameters["max_episodes"]
        self.plot_frequency = self.max_episodes // 1000
        self.batch_size = parameters["batch_size"]
        self.gamma = parameters["gamma"]  # Discount factor
        self.initial_epsilon = parameters["initial_epsilon"]  # Initial epsilon value
        self.target_epsilon = parameters["target_epsilon"]  # Maximum epsilon value
        self.epsilon_converge_ratio = parameters["epsilon_converge_ratio"]
        self._init_epsilon()
        self.buffer_size = parameters["buffer_size"]
        self.replay_buffer = ReplayBuffer(buffer_size=self.buffer_size)
        self._init_optimizer(parameters["learning_rate"])

    def _init_optimizer(self, learning_rate: float):
        """Initialize the optimizer."""
        self.optimizer = torch.optim.Adam(self.dnn.parameters(), lr=learning_rate)

    def _init_epsilon(self):
        """Initialize the epsilon value."""
        self.epsilon = self.initial_epsilon

    def _init_training_variables(self):
        """Initialize the training variables."""
        self.loss = 0
        self.batch_count = 0

    def _init_train_metrics(self):
        """Initialize the training metrics."""
        self.movements_count = [0] * self.max_episodes
        self.scores = [0] * self.max_episodes
        self.rewards = [0] * self.max_episodes
        self.epsilons = [0] * self.max_episodes
        self.best_average_score = 0

    def _update_epsilon(self, episode: int):
        """Update the epsilon value."""
        decay_episodes = self.max_episodes * self.epsilon_converge_ratio
        decay_rate = (
            -math.log(self.target_epsilon / self.initial_epsilon) / decay_episodes
        )
        self.epsilon = max(
            self.target_epsilon, self.initial_epsilon * math.exp(-decay_rate * episode)
        )
        self.epsilons[episode] = float(self.epsilon)

    def _sample_action(self, states: list) -> tuple[int, torch.Tensor]:
        """
        Sample an action based on the given states.

        Args:
            states (list): A list of states containing snake_direction, food_distance,
                snake_body_danger, snake_wall_danger, and board_tensor.

        Returns:
            tuple[int, torch.Tensor]: A tuple containing the sampled action and the
                logits output from the DNN model.
        """
        snake_direction, food_distance, snake_body_danger, snake_wall_danger, board_tensor = (
            states
        )
        snake_direction_tensor = torch.tensor(snake_direction).float().unsqueeze(0)
        food_distance_tensor = torch.tensor(food_distance).float().unsqueeze(0)
        snake_body_danger_tensor = torch.tensor(snake_body_danger).float().unsqueeze(0)
        snake_wall_danger_tensor = torch.tensor(snake_wall_danger).float().unsqueeze(0)
        board_tensor = torch.tensor(board_tensor).float().unsqueeze(0)
        dnn_logits = self.dnn(
            snake_direction_tensor,
            food_distance_tensor,
            snake_body_danger_tensor,
            snake_wall_danger_tensor,
            board_tensor,
        )
        action = (
            torch.argmax(dnn_logits).item()
            if random_sample() > self.epsilon
            else self.env.action_space.sample()
        )
        return action, dnn_logits

    def _compute_loss(
        self, q_value: torch.tensor, target_q_value: torch.tensor
    ) -> torch.tensor:
        """Compute the loss for the DNN.
            
            Args:
                q_value (torch.tensor): Q-value predicted by the network
                target_q_value (torch.tensor): Target Q-value based on the Bellman equation
            
            Returns:
                torch.tensor: Loss
            """
        return nn.MSELoss()(q_value, target_q_value)

    def _update_weights(self) -> None:
        """Update the weights of the DNN."""

        self.loss /= self.batch_count
        self.loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.loss = 0
        self.batch_count = 0

    def _create_metrics_plot(self, episode: int) -> None:
        """
        Create a metrics plot for the RL trainer.

        Args:
            episode (int): The current episode number.

        Returns:
            None
        """

        figure, axes = plt.subplots(2, 2, figsize=(30, 10))

        axes[0][0].plot(list(range(episode)), self.scores[:episode], label="Score")
        axes[0][0].plot(
            list(range(episode)),
            sliding_list_average(self.scores[:episode]),
            label="Average Score",
        )
        axes[0][0].set_title("Score")
        axes[1][0].plot(list(range(episode)), self.rewards[:episode], label="Reward")
        axes[1][0].plot(
            list(range(episode)),
            sliding_list_average(self.rewards[:episode]),
            label="Average Reward",
        )
        axes[1][0].set_title("Reward")
        axes[1][0].set_xlabel("Episode")
        axes[0][1].plot(
            list(range(episode)), self.movements_count[:episode], label="Movements"
        )
        axes[0][1].plot(
            list(range(episode)),
            sliding_list_average(self.movements_count[:episode]),
            label="Average Movements",
        )
        axes[0][1].set_title("Movements")
        axes[1][1].plot(list(range(episode)), self.epsilons[:episode], label="Epsilon")
        axes[1][1].set_title("Epsilon")
        axes[1][1].set_xlabel("Episode")

        plt.savefig("models/{}/metrics.png".format(self.model_name))
        plt.close()

    def _log_metrics(
        self, episode: int, movements_count: int, score: int, reward: int
    ) -> None:
        """
            Logs the metrics for each episode.

            Args:
                episode (int): The episode number.
                movements_count (int): The number of movements in the episode.
                score (int): The score achieved in the episode.
                reward (int): The reward obtained in the episode.

            Returns:
                None
            """
        self.movements_count[episode] = movements_count
        self.scores[episode] = score
        self.rewards[episode] = float(reward)
        if episode % self.plot_frequency == 0:
            self._create_metrics_plot(episode)

    def _save_model(self, episode: int) -> None:
        """Save the model.

            This method saves the model if the current average score is higher than the best average score.

            Args:
                episode (int): The episode number.
            
            Returns:
                None
            """
        if episode > 150:
            current_average_score = sliding_list_average(self.scores[:episode])[-1]
            if current_average_score > self.best_average_score:
                torch.save(
                    self.dnn.state_dict(), "models/{}/model.pth".format(self.model_name)
                )
                self.best_average_score = current_average_score

    def _train_episode(self) -> tuple[int, float, float]:
        """Train an episode.

            This method trains a single episode of the reinforcement learning agent.
            It performs the following steps:
            1. Resets the environment and initializes necessary variables.
            2. Executes actions based on the current state and samples the next action.
            3. Calculates the target Q-value using the Bellman equation.
            4. Adds the experience to the replay buffer.
            5. Calculates the loss and updates the Q-network.
            6. Updates the movement count, rewards, and checks if the episode is done.
            7. Returns the movement count, score obtained, and total rewards.

            Returns:
                tuple[int, float, float]: A tuple containing the movement count, score obtained, and total rewards.
            """

        states = self.env.reset()
        done = False
        movements_count = 0
        rewards = 0

        while not done:
            action, _ = self._sample_action(states)
            prev_states = [torch.tensor(state).float().unsqueeze(0) for state in states]
            states, reward, _, _, done = self.env.step(action)

            # Calculate target Q-value using the Bellman equation
            next_states = [torch.tensor(state).float().unsqueeze(0) for state in states]
            self.replay_buffer.add_experience(
                (prev_states, action, reward, next_states, done)
            )
            target_q_value = reward + self.gamma * torch.max(self.dnn(*next_states))

            # Calculate the loss and update the Q-network
            q_value = self.dnn(*prev_states)[0][action]
            self.loss += self._compute_loss(q_value, target_q_value)
            self.batch_count += 1
            movements_count += 1
            rewards += reward

        return movements_count, self.env.score, rewards

    def _apply_buffer_replay(self) -> None:
        """
        Applies buffer replay to update the neural network model.

        This method samples a batch of experiences from the replay buffer and performs the following steps:
        1. Calculates the target Q-value using the Bellman equation.
        2. Computes the loss between the predicted Q-value and the target Q-value.
        3. Backpropagates the loss and updates the model's weights.
        4. Resets the gradients of the optimizer.

        """
        batch = self.replay_buffer.sample_batch(self.batch_size)
        for experience_idx, experience in enumerate(batch):
            prev_states, action, reward, next_states, done = experience

            # Calculate target Q-value using the Bellman equation
            target_q_value = reward + self.gamma * torch.max(self.dnn(*next_states))
            q_value = self.dnn(*prev_states)[0][action]
            loss = self._compute_loss(q_value, target_q_value)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

    def train(self) -> None:
        """Train the DNN.

        This method trains the DNN by running multiple episodes of the game. It initializes the training variables
        and metrics, and then iterates over the episodes. For each episode, it trains the agent, updates the weights,
        applies buffer replay, logs the metrics, saves the model, and updates the epsilon value. Finally, it displays
        a progress bar to track the training progress.

        """
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

    def test(self, games: int = 1, display: bool = False) -> None:
        """
        Run the testing phase of the RL trainer.

        Args:
            games (int): The number of games to play during testing. Default is 1.
            display (bool): Whether to display the game environment during testing. Default is False.
        
        Returns:
            None
        """
        self.dnn.load_state_dict(torch.load("models/" + self.model_name + "/model.pth"))
        self.dnn.eval()
        self.epsilon = 0
        scores = list()
        for _ in range(games):
            states = self.env.reset()
            done = False
            if display:
                self.env.render()
                time.sleep(0.5)
            while not done:
                action, _ = self._sample_action(states)
                states, _, score, _, done = self.env.step(action)
                if display:
                    self.env.render()
            scores.append(score)
        print(
            "Mean Score: {}, Max Score: {}".format(
                sum(scores) / len(scores), max(scores)
            )
        )
        self.env.close()
