import gym
import numpy as np

from src.snake_game import SnakeGame

class SnakeEnv(gym.Env):
    
    ACTIONS_DICT = {
        0: "RIGHT",
        1: "LEFT",
        2: "UP",
        3: "DOWN"
    }

    def __init__(self, board_size = 350):
        super(SnakeEnv, self).__init__()
        self.board_size = board_size
        self.score = 0
        self.game = SnakeGame(board_size=board_size)
        self.action_space = gym.spaces.Discrete(4)        
        self._update_observation_space()
        

    def _update_observation_space(self):
        snake_position, snake_body, food_position = self.game.get_elements_space()
        self.observation_space = -np.ones((self.board_size, self.board_size), dtype=np.int8)
        for snake_point in snake_body[1:]:
            self.observation_space[snake_point[0], snake_point[1]] = 0
        self.observation_space[snake_position[0], snake_position[1]] = 1
        self.observation_space[food_position[0], food_position[1]] = 2

    def _compute_reward(self, score, done):
        if done:
            return -20
        elif score > self.score:
            self.score = score
            return 20
        else:
            return 0.1

    def reset(self):
        self.game.reset()
        self.score = 0
        self._update_observation_space()
        return self.observation_space

    def step(self, action):
        states, score, _, _, done = self.game.step(self.ACTIONS_DICT[action])
        snake_position, _, food_position = self.game.get_elements_space()
        return self.observation_space, self._compute_reward(score, done), _, _, done

    def render(self):
        self.game.render()