import gym
import numpy as np

from src.snake_game import SnakeGame

class SnakeEnv(gym.Env):
    '''Snake Environment.'''
    ACTIONS_DICT = {
        0: "RIGHT",
        1: "LEFT",
        2: "UP",
        3: "DOWN"
    }
    def __init__(self, board_size = 350, display = False):
        '''Initialize the Snake Environment.
        Args: board_size (int): Size of the board
        '''
        super(SnakeEnv, self).__init__()
        self.board_size = board_size
        self.score = 0
        self.game = SnakeGame(board_size=board_size, display=display)
        self.action_space = gym.spaces.Discrete(4)        
        self._update_observation_space()
        

    def _update_observation_space(self):
        '''Update the observation space.'''
        snake_position, snake_body, food_position = self.game.get_elements_space()
        self.observation_space = -np.ones((self.board_size, self.board_size), dtype=np.int8)
        for snake_point in snake_body[1:]:
            self.observation_space[snake_point[0], snake_point[1]] = 0
        self.observation_space[snake_position[0], snake_position[1]] = 1
        self.observation_space[food_position[0], food_position[1]] = 2

    def _compute_reward(self, score: int, done: bool, snake_position: list, food_position: list) -> float():
        '''Compute the reward.
        Args: score (int): Score
              done (bool): Done
        Returns: float: Reward
        '''
        if done:
            return -10
        elif score > self.score:
            self.score = score
            return 10
        else:
            return 0

    def reset(self):
        '''Reset the environment.
        Returns: tuple[np.array, list, list, list]: Observation, snake position, snake body, food position
        '''
        self.game.reset()
        self.score = 0
        self._update_observation_space()
        snake_position, snake_body, food_position = self.game.get_elements_space()

        return self.observation_space, snake_position, snake_body, food_position

    def step(self, action: int)-> tuple[tuple[np.array, list, list, list], float, bool, dict, bool]:
        '''Take a step in the environment.
        Args: action (int): Action
        Returns: tuple[tuple[np.array, list, list, list], float, bool, dict, bool]: Observation, reward, done, info, done
        '''
        states, score, _, _, done = self.game.step(self.ACTIONS_DICT[action])
        snake_position, snake_body, food_position = self.game.get_elements_space()
        return [self.observation_space, snake_position, snake_body, food_position], self._compute_reward(score, done, snake_position, food_position), _, _, done

    def render(self):
        '''Render the environment.'''
        self.game.render()