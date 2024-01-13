import gym
import numpy as np

from snake_game import SnakeGame


class SnakeEnv(gym.Env):
    """Snake Environment."""

    ACTIONS_DICT = {0: "RIGHT", 1: "LEFT", 2: "UP", 3: "DOWN"}

    def __init__(self, board_size=350, display=False):
        """Initialize the Snake Environment.
        Args: board_size (int): Size of the board
        """
        super(SnakeEnv, self).__init__()
        self.board_size = board_size
        self.score = 0
        self.game = SnakeGame(board_size=board_size, display=display)
        self.action_space = gym.spaces.Discrete(4)

    def _compute_reward(
        self, score: int, done: bool, snake_position: list, food_position: list
    ) -> float():
        """Compute the reward.
        Args: score (int): Score
              done (bool): Done
        Returns: float: Reward
        """
        if done:
            return -10
        elif score > self.score:
            self.score = score
            return 10
        else:
            return -1

    def _get_snake_body_danger(self, snake_position: list, snake_body: list) -> list:
        """Get the danger of the snake in each direction in relation to its body.
        Args: snake_position (list): Snake position
              snake_body (list): Snake body
        Returns: list: Danger of the snake in each direction
        """
        danger = [0, 0, 0, 0]
        for body_part in snake_body:
            if snake_position[0] == body_part[0]:
                if snake_position[1] < body_part[1]:
                    danger[0] += 1
                else:
                    danger[1] += 1
            elif snake_position[1] == body_part[1]:
                if snake_position[0] < body_part[0]:
                    danger[2] += 1
                else:
                    danger[3] += 1
        danger = np.array(danger) - max(danger)
        danger = np.sign(danger) + 1
        return danger

    def _get_snake_wall_danger(self, snake_position: list) -> list:
        """Get the danger of the snake in each direction in relation to the walls.
        Args: snake_position (list): Snake position
        Returns: list: Danger of the snake in each direction
        """
        danger = [0, 0, 0, 0]
        if snake_position[0] == 0:
            danger[2] = 1
        elif snake_position[0] == self.board_size:
            danger[3] = 1
        if snake_position[1] == 0:
            danger[0] = 1
        elif snake_position[1] == self.board_size:
            danger[1] = 1
        return np.array(danger)
    
    def __get_board_tensor(self, snake_body: list, food_position: list) -> np.array:
        """
        Get the 3D tensor representing the game board.
        """
        board_tensor = np.zeros((2, self.board_size, self.board_size))
        for body_part in snake_body:
            board_tensor[0, body_part[0], body_part[1]] = 1
        board_tensor[1, food_position[0], food_position[1]] = 1

        return board_tensor

    def _calculate_states(self):
        snake_position, snake_body, food_position = self.game.get_elements_space()
        snake_direction = np.sign(
            np.array(
                [
                    snake_position[0] - snake_body[1][0],
                    snake_position[1] - snake_body[1][1],
                ]
            )
        )
        food_distance = np.sign(
            np.array(
                [
                    food_position[0] - snake_position[0],
                    food_position[1] - snake_position[1],
                ]
            )
        )
        snake_direction_danger = self._get_snake_body_danger(snake_position, snake_body)
        snake_wall_danger = self._get_snake_wall_danger(snake_position)
        board_tensor = self.__get_board_tensor(snake_body, food_position)
        return snake_direction, food_distance, snake_direction_danger, snake_wall_danger, board_tensor

    def reset(self):
        """Reset the environment.
        Returns: tuple[np.array, list, list, list]: Observation, snake position, snake body, food position
        """
        self.game.reset()
        self.score = 0
        return self._calculate_states()

    def step(
        self, action: int
    ) -> tuple[tuple[np.array, list, list, list], float, bool, dict, bool]:
        """Take a step in the environment.
        Args: action (int): Action
        Returns: tuple[tuple[np.array, list, list, list], float, bool, dict, bool]: Observation, reward, done, info, done
        """
        states, score, _, _, done = self.game.step(self.ACTIONS_DICT[action])
        snake_position, _, food_position = self.game.get_elements_space()
        return (
            self._calculate_states(),
            self._compute_reward(score, done, snake_position, food_position),
            score,
            _,
            done,
        )

    def render(self):
        """Render the environment."""
        self.game.render()
