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
        self.surface_dimensions = self.game.get_surface_dimensions()
        self.action_space = gym.spaces.Discrete(4)

    def get_surface_dimensions(self) -> list:
        """
            Returns the dimensions of the surface.
            
            Returns:
                list: A list containing the width and height of the surface.
            """
        return self.surface_dimensions

    def _compute_reward(
        self, score: int, done: bool, snake_position: list, food_position: list
    ) -> float():
        """
            Compute the reward based on the score, game completion status, snake position, and food position.

            Parameters:
            score (int): The current score of the game.
            done (bool): A flag indicating whether the game is completed or not.
            snake_position (list): The current position of the snake.
            food_position (list): The position of the food.

            Returns:
            float: The computed reward value.

            If the game is completed (done=True), the reward is -10.
            If the score is higher than the previous score, the reward is 10.
            Otherwise, the reward is -1.
            """
        if done:
            return -10
        elif score > self.score:
            self.score = score
            return 10
        else:
            return -1

    def _get_snake_body_danger(self, snake_position: list, snake_body: list) -> list:
        """
        Calculates the danger level for each direction around the snake's body.

        Args:
            snake_position (list): The current position of the snake.
            snake_body (list): The body parts of the snake.

        Returns:
            list: A list representing the danger level in each direction. The order of the directions is [up, down, left, right].
                  The danger level is calculated based on the proximity of the snake's body parts in each direction.
                  Higher values indicate higher danger.

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
        """
        Calculates the danger of the snake being near the walls of the game board.

        Args:
            snake_position (list): The current position of the snake as a list [x, y].

        Returns:
            list: A list representing the danger of the snake being near the walls.
                  The list contains four elements [left, right, up, down], where each
                  element is either 0 or 1. A value of 1 indicates that the snake is
                  near the corresponding wall, while a value of 0 indicates that the
                  snake is not near that wall.
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

            Parameters:
            snake_body (list): The list of coordinates representing the snake's body.
            food_position (list): The coordinates of the food on the game board.

            Returns:
            np.array: The 3D tensor representing the game board.
            """
        board_tensor = np.zeros(
            (2, self.surface_dimensions[0], self.surface_dimensions[1])
        )
        for body_part in snake_body:
            board_tensor[0, body_part[0] - 1, body_part[1] - 1] = 1
        board_tensor[1, food_position[0], food_position[1]] = 1

        return board_tensor

    def _calculate_states(self):
        """
        Calculates the states for the snake environment.

        This method calculates various states that are used to represent the current state of the snake environment.
        It retrieves the snake's position, body, and the position of the food from the game. It then calculates the
        snake's direction, food distance, snake direction danger, snake wall danger, and the board tensor.

        Returns:
            tuple: A tuple containing the calculated states:
                - snake_direction (numpy.ndarray): The direction of the snake.
                - food_distance (numpy.ndarray): The distance between the snake and the food.
                - snake_direction_danger (numpy.ndarray): The danger level of the snake's direction.
                - snake_wall_danger (numpy.ndarray): The danger level of the snake's proximity to the wall.
                - board_tensor (numpy.ndarray): The tensor representation of the game board.
        """
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
        return (
            snake_direction,
            food_distance,
            snake_direction_danger,
            snake_wall_danger,
            board_tensor,
        )

    def reset(self) -> tuple[np.array, list, list, list]:
        """
        Reset the environment.

        Returns:
            tuple[np.array, list, list, list]: Observation, snake position, snake body, food position
        """
        self.game.reset()
        self.score = 0
        return self._calculate_states()

    def step(
        self, action: int
    ) -> tuple[tuple[np.array, list, list, list], float, bool, dict, bool]:
        """
        Executes a single step in the Snake game environment.

        Args:
            action (int): The action to be taken by the agent.

        Returns:
            tuple: A tuple containing the following elements:
                - states (tuple): A tuple containing the state information of the environment.
                - reward (float): The reward obtained from the step.
                - score (bool): The current score in the game.
                - _ (dict): Placeholder for additional information (not used).
                - done (bool): Indicates whether the game is over or not.
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
