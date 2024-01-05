import random

import pygame


class SnakeGame:
    """Snake Game."""

    def __init__(self, board_size: int = 400, display: bool = False):
        """Initialize the Snake Game.
        Args: board_size (int): Size of the board
              display (bool): Display the game
        """
        self.board_size = board_size
        self.display = display
        self.pixel_size = 10
        self.title_box_size = [max(self.board_size, 100), 40]
        self._init_game()

    def _init_game(self):
        """Initialize the game."""
        if self.display:
            self._init_board()
        self._init_elements()
        self._init_score()

    def reset(self):
        """Reset the game."""
        self._init_elements()
        self._init_score()

    def _init_board(self):
        """Initialize the board."""
        pygame.init()
        self.game_surface = pygame.display.set_mode((self.title_box_size[0], self.board_size+self.title_box_size[1]))
        self.fps = pygame.time.Clock()
        self.font = pygame.font.Font(None, 25)

    def _init_elements(self):
        """Initialize the elements."""
        x_postion = int(random.randint(3, self.board_size / 10 - 3) * 10)
        y_postion = int(random.randint(3, self.board_size / 10 - 3 + self.title_box_size[1]) * 10)
        self.snake_position = [x_postion, y_postion]
        self.snake_body = [
            [x_postion, y_postion],
            [x_postion - self.pixel_size, y_postion - self.pixel_size],
            [x_postion - 2 * self.pixel_size, y_postion - 2 * self.pixel_size],
        ]

        self.action = "RIGHT"
        self.done = False
        self._update_position_food()

    def get_elements_space(self):
        """Get the elements space.
        Returns: tuple[list, list, list]: Snake position, snake body, food position"""
        return self.snake_position, self.snake_body, self.food_position

    def _init_score(self):
        """Initialize the score."""
        self.score = 0

    def _update_position_food(self):
        """Update the position of the food."""
        valid_position = False
        while not valid_position:
            x_random_position = random.randint(0, self.board_size // 10 - 1) * 10
            y_random_position = random.randint(self.title_box_size[1]//10, self.board_size // 10 + self.title_box_size[1]//10 - 1) * 10
            self.food_position = [x_random_position, y_random_position]
            if self.food_position not in self.snake_body:
                valid_position = True

    def _register_actions(self):
        """Register the actions from the keyboard."""
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

    def _update_board_elements(self, action: str):
        """Update the board elements.
        Args: action (str): Action
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
        """Step the game.
        Args: action (int): Action
        Returns: tuple[tuple[list, list, list], int, bool, bool, bool]: Elements space, score, truncated, info, done
        """
        self._update_board_elements(action)
        self._update_game_status()
        return self.get_elements_space(), self.score, False, False, self.done

    def _update_display(self):
        """Update the display with improved graphics."""
        # Background color
        self.game_surface.fill((0, 0, 0))

        # Draw title outside the box
        title_font = pygame.font.Font(None, 36)
        title_text = title_font.render("Snake Game", True, (255, 255, 255))
        title_rect = title_text.get_rect(center=(self.title_box_size[0] // 3, self.title_box_size[1] // 2))
        self.game_surface.blit(title_text, title_rect)

        # Draw score outside the box
        score_text = title_font.render("Score: " + str(self.score), True, (255, 255, 255))
        score_rect = score_text.get_rect(center=(self.title_box_size[0] - 100, self.title_box_size[1] // 2))
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
        pygame.draw.rect(self.game_surface, (255, 255, 255), (5, 5 + self.title_box_size[1], 390, 390), 3)

        # Update display
        self.fps.tick(self.pixel_size)
        pygame.display.flip()

    def _update_game_status(self):
        """Update the game status."""
        if (
            self.snake_position[0] < 0
            or self.snake_position[0] > self.board_size - self.pixel_size
        ):
            self.done = True
        if (
            self.snake_position[1] < 0
            or self.snake_position[1] > self.board_size + self.title_box_size[1]- self.pixel_size
        ):
            self.done = True
        if self.snake_position in self.snake_body[1:]:
            self.done = True

    def _update_board(self, action: str):
        """Update the board.
        Args: action (str): Action
        """
        self._update_board_elements(action)
        if self.display:
            self._update_display()
        self._update_game_status()

    def start(self):
        """Start the game."""
        while not self.done:
            self._register_actions()
            self._update_board(self.action)
        if self.display:
            pygame.quit()
