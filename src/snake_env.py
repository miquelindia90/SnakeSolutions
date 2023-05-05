import gym
import numpy as np

from snake_game import SnakeGame

class SnakeEnv(gym.Env):
    def __init__(self, board_size = 350):
        super(SnakeEnv, self).__init__()
        self.board_size = board_size
        self.game = SnakeGame(board_size=board_size)
        self.action_space = gym.spaces.Discrete(4)        
        self._init_observation_space()

    def _init_observation_space(self):
        self.observation_space = -np.ones((self.board_size, self.board_size), dtype=np.int8)
        self._update_observation_space()

    def _update_observation_space(self):
        _, snake_body, food_position = self.game.get_elements_space()
        for snake_point in snake_body:
            self.observation_space[snake_point[0], snake_point[1]] = 0
        self.observation_space[food_position[0], food_position[1]] = 1

    def reset(self):
        self.game.reset()
        self._init_observation_space()
        return self.observation_space

    def step(self, action):
        pass    

    def render(self):
        pass

def main():

    env = SnakeEnv()
    states = env.reset()
    print(states)
    #for _ in range(1000):
    #    env.render()
    #    env.step(env.action_space.sample())

if __name__ == '__main__':
    main()