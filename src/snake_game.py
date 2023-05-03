import sys
import time
import random

import pygame


class SnakeGame:
    def __init__(self, board_size = 350, display=True):

        self.board_size = board_size
        self.display = display
        self.pixel_size = 10
        if self.display:
            self._init_board()
        self._init_elements()
        self._init_rewards()

    def _init_board(self):
        pygame.init()
        self.game_surface = pygame.display.set_mode((self.board_size, self.board_size))
        self.fps = pygame.time.Clock()
        self.font = pygame.font.Font(None, 25)
    
    def _init_elements(self):
        self.snake_position = [100, 50]
        self.snake_body = [[100,50], [90, 50], [80, 50]]

        self.action = "RIGHT"
        self.done = True
        self._update_position_food()
        
    def _init_rewards(self):
        self.score = 0

    def _update_position_food(self):
        random_position = random.randint(0, self.board_size//10 - 1)*10
        self.food_position = [random_position, random_position]

    def _register_actions(self):

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.done = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RIGHT and self.action!="LEFT":
                    self.action = "RIGHT"
                if event.key == pygame.K_LEFT and self.action!="RIGHT":
                    self.action = "LEFT"
                if event.key == pygame.K_UP and self.action!="DOWN":
                    self.action = "UP"
                if event.key == pygame.K_DOWN and self.action!="UP":
                    self.action = "DOWN"

    def _update_board_elements(self, action):
        
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

    def _update_display(self):

        self.game_surface.fill((0,0,0))
        for position in self.snake_body:
            pygame.draw.rect(self.game_surface, (200,200,200), pygame.Rect(position[0], position[1], self.pixel_size, self.pixel_size))
        pygame.draw.rect(self.game_surface, (255,0,0), pygame.Rect(self.food_position[0], self.food_position[1], self.pixel_size, self.pixel_size))
        text = self.font.render(str(self.score), 0, (250,60,80))
        self.game_surface.blit(text, (self.board_size-2*self.pixel_size,20))
        self.fps.tick(self.pixel_size)
        pygame.display.flip()

    def _update_game_status(self):

        if self.snake_position[0] < 0 or self.snake_position[0] > self.board_size-self.pixel_size:
            self.done = False
        if self.snake_position[1] < 0 or self.snake_position[1] > self.board_size-self.pixel_size:
            self.done = False
        if self.snake_position in self.snake_body[1:]:
            self.done = False

    def _update_board(self, action):

        self._update_board_elements(action)
        if self.display:
            self._update_display()
        self._update_game_status()
            

    def start(self):
        
        while self.done:
            
            self._register_actions()
            self._update_board(self.action)    

        pygame.quit()