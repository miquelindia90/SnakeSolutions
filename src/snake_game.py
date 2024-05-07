import random

import pygame


class SnakeGame:
    """Class that contains all the methods functions to run the Snake Game."""

    def __init__(self, board_size: int = 300, display: bool = False):
        """
        Initialize the Snake Game.

        Args:
            board_size (int): Size of the board.
            display (bool): Display the game.
        """
        self.board_size = board_size
        self.board_padding = 10
        self.display = display
        self.pixel_size = 10
        self.title_box_size = [self.board_size, self.board_size // 10]
        self.game_surface_dimensions = [
            self.title_box_size[0] + self.board_padding,
            self.board_size + self.title_box_size[1] + self.board_padding,
        ]
        self._init_game()

    def _init_game(self) -> None:
        """Initialize the game.
            
        This method initializes the game by calling other initialization methods.
        If the `display` attribute is True, it also initializes the game board.
        """
        if self.display:
            self._init_board()
        self._init_elements()
        self._init_score()

    def reset(self) -> None:
        """
        Resets the game by initializing the elements and score.
        """
        self._init_elements()
        self._init_score()

    def get_surface_dimensions(self) -> list:
        """
        Returns the dimensions of the game surface.

        Returns:
            list: A list containing the width and height of the game surface.
        """
        return self.game_surface_dimensions

    def _init_board(self) -> None:
        """
        Initializes the game board.

        This method initializes the game board by initializing the Pygame module,
        creating a game surface with the specified dimensions, and setting up the
        frame rate for the game.

        Returns:
            None
        """
        pygame.init()
        self.game_surface = pygame.display.set_mode(
            (self.game_surface_dimensions[0], self.game_surface_dimensions[1])
        )
        self.fps = pygame.time.Clock()

    def _init_elements(self) -> None:
        """Initialize the elements.

        This method initializes the elements of the snake game, including the snake's position and body,
        the current action, and the status of the game.

        The snake's position is randomly generated within the game board, while the snake's body is
        initialized with three segments. The action is set to "RIGHT" by default, and the game status
        is set to False (not done).

        This method also calls the `_update_position_food` method to update the position of the food
        within the game board.

        Returns:
            None
        """
        x_postion = int(
            random.randint(
                3 + self.board_padding // self.pixel_size,
                (self.board_size) // self.pixel_size - 3,
            )
            * self.pixel_size
        )
        y_postion = int(
            random.randint(
                3 + self.board_padding // self.pixel_size,
                (self.board_size + self.title_box_size[1]) // self.pixel_size - 3,
            )
            * self.pixel_size
        )
        self.snake_position = [x_postion, y_postion]
        self.snake_body = [
            [x_postion, y_postion],
            [x_postion - self.pixel_size, y_postion - self.pixel_size],
            [x_postion - 2 * self.pixel_size, y_postion - 2 * self.pixel_size],
        ]

        self.action = "RIGHT"
        self.done = False
        self._update_position_food()

    def get_elements_space(self) -> tuple[list, list, list]:
        """Get the elements space.

        This method returns a tuple containing the snake position, snake body, and food position.

        Returns:
            tuple[list, list, list]: A tuple containing the snake position, snake body, and food position.
        """
        return self.snake_position, self.snake_body, self.food_position

    def _init_score(self) -> None:
        """Initialize the score."""
        self.score = 0

    def _update_position_food(self) -> None:
        """Update the position of the food.

        This method generates a random position for the food on the game board.
        It ensures that the food does not overlap with the snake's body.

        Returns:
            None
        """
        valid_position = False
        while not valid_position:
            x_random_position = (
                random.randint(
                    self.board_padding // self.pixel_size,
                    (self.board_size) // self.pixel_size - 1,
                )
                * self.pixel_size
            )
            y_random_position = (
                random.randint(
                    (self.title_box_size[1] + self.board_padding) // self.pixel_size,
                    (self.board_size + self.title_box_size[1]) // self.pixel_size - 1,
                )
                * self.pixel_size
            )
            self.food_position = [x_random_position, y_random_position]
            if self.food_position not in self.snake_body:
                valid_position = True

    def _register_actions(self) -> None:
        """Register the actions from the keyboard.

        This method listens for keyboard events and updates the `action` attribute
        based on the keys pressed. It checks for arrow key presses and updates the
        `action` attribute accordingly, ensuring that the snake does not reverse
        its direction.

        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RIGHT and self.action != "LEFT":
                    self.action = "RIGHT"
                if event.key == pygame.K_LEFT and self.action != "RIGHT":
                    self.action = "LEFT"
                if event.key == pygame.K_UP and self.action != "DOWN":
                    self.action = "UP"
                if event.key == pygame.K_DOWN and self.action != "UP":
                    self.action = "DOWN"

    def _update_board_elements(self, action: str) -> None:
        """
        Update the board elements based on the given action.

        Args:
            action (str): The action to be performed. Can be "RIGHT", "LEFT", "UP", or "DOWN".

        Returns:
            None
        """

        if action == "RIGHT":
            self.snake_position[0] += self.pixel_size
        elif action == "LEFT":
            self.snake_position[0] -= self.pixel_size
        elif action == "UP":
            self.snake_position[1] -= self.pixel_size
        elif action == "DOWN":
            self.snake_position[1] += self.pixel_size

        self.snake_body.insert(0, list(self.snake_position))
        if self.snake_position == self.food_position:
            self._update_position_food()
            self.score += 1
        else:
            self.snake_body.pop()

    def render(self):
        """Render the game."""
        self._update_display()

    def step(
        self, action: int
    ) -> tuple[tuple[list, list, list], int, bool, bool, bool]:
        """
        Takes a step in the game based on the given action.

        Args:
            action (int): The action to be taken.

        Returns:
            tuple[tuple[list, list, list], int, bool, bool, bool]: A tuple containing the updated elements space,
            the score, a flag indicating if the game is over, a flag indicating if the snake collided with itself,
            and a flag indicating if the snake collided with the wall.
        """
        self._update_board_elements(action)
        self._update_game_status()
        return self.get_elements_space(), self.score, False, False, self.done

    def _update_display(self) -> None:
        """
        Update the display of the snake game.

        This method updates the game surface by filling it with the background color,
        drawing the title and score outside the box, drawing the snake and food inside the box,
        adding borders to the snake box, and updating the display.

        Returns:
            None
        """
        # Background color
        self.game_surface.fill((0, 0, 0))

        # Draw title outside the box
        title_font = pygame.font.Font(None, self.board_size // 10)
        title_text = title_font.render("Snake Game", True, (255, 255, 255))
        title_rect = title_text.get_rect(
            center=(self.title_box_size[0] // 3, self.title_box_size[1] // 2)
        )
        self.game_surface.blit(title_text, title_rect)

        # Draw score outside the box
        title_font = pygame.font.Font(None, self.board_size // 10)
        score_text = title_font.render(
            "Score: " + str(self.score), True, (255, 255, 255)
        )
        score_rect = score_text.get_rect(
            center=(
                self.title_box_size[0] - self.board_size // 5,
                self.title_box_size[1] // 2,
            )
        )
        self.game_surface.blit(score_text, score_rect)

        # Draw snake and food inside the box
        for position in self.snake_body:
            pygame.draw.rect(
                self.game_surface,
                (200, 200, 200),
                pygame.Rect(position[0], position[1], self.pixel_size, self.pixel_size),
            )
        pygame.draw.rect(
            self.game_surface,
            (255, 0, 0),
            pygame.Rect(
                self.food_position[0],
                self.food_position[1],
                self.pixel_size,
                self.pixel_size,
            ),
        )

        # Add borders to the snake box
        pygame.draw.rect(
            self.game_surface,
            (255, 255, 255),
            (
                0,
                0 + self.title_box_size[1],
                self.board_size + self.board_padding,
                self.board_size + self.board_padding,
            ),
            self.board_padding,
        )

        # Update display
        self.fps.tick(self.pixel_size)
        pygame.display.flip()

    def _update_game_status(self) -> None:
        """
        Update the game status based on the snake's position and body.

        If the snake goes out of bounds or collides with its own body, the game is considered done.

        Returns:
            None
        """
        if (
            self.snake_position[0] < self.board_padding
            or self.snake_position[0]
            > self.board_size + self.board_padding - self.pixel_size
        ):
            self.done = True
        if (
            self.snake_position[1] < self.title_box_size[1] + self.board_padding
            or self.snake_position[1]
            > self.board_size + self.title_box_size[1] - self.pixel_size
        ):
            self.done = True
        if self.snake_position in self.snake_body[1:]:
            self.done = True

    def _update_board(self, action: str) -> None:
        """
        Updates the game board based on the given action.

        Args:
            action (str): The action to be performed on the game board.

        Returns:
            None
        """
        self._update_board_elements(action)
        if self.display:
            self._update_display()
        self._update_game_status()

    def start(self) -> None:
        """Start the game."""
        while not self.done:
            self._register_actions()
            self._update_board(self.action)
        if self.display:
            pygame.quit()
