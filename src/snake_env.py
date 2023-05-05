import gym
import numpy as np

from snake_game import SnakeGame

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

    def _compute_reward(self, snake_position, food_position):
        return np.array(self.board_size - (snake_position[0]-food_position[0] + snake_position[1]-food_position[1])/2)


    def reset(self):
        self.game.reset()
        self._init_observation_space()
        return self.observation_space

    def step(self, action):
        states, _, _, _, done = self.game.step(self.ACTIONS_DICT[action])
        snake_position, _, food_position = self.game.get_elements_space()
        return self.observation_space, self._compute_reward(states[0], states[2]), _, _, done

    def render(self):
        self.game.render()

def main():

    env = SnakeEnv()
    states = env.reset()
    done = False
    while not done:
        #states, reward, _, _, done = env.step(env.action_space.sample())
        states, reward, _, _, done = env.step(0)
        print(reward, done)
        env.render()

if __name__ == '__main__':
    main()