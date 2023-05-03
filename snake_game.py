import sys
import time
import random

import pygame

pygame.init()
game_surface = pygame.display.set_mode((500,500))
fps = pygame.time.Clock()
font = pygame.font.Font(None, 25)

def food():
    random_position = random.randint(0, 49)*10
    food_position = [random_position, random_position]
    return food_position

def main():
    snake_position = [100, 50]
    snake_body = [[100,50], [90, 50], [80, 50]]

    change = "RIGHT"
    run = True
    food_position = food()
    score = 0

    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RIGHT:
                    change = "RIGHT"
                if event.key == pygame.K_LEFT:
                    change = "LEFT"
                if event.key == pygame.K_UP:
                    change = "UP"
                if event.key == pygame.K_DOWN:
                    change = "DOWN"

        if change == "RIGHT":
            snake_position[0] += 10
        if change == "LEFT":
            snake_position[0] -= 10
        if change == "UP":  
            snake_position[1] -= 10
        if change == "DOWN":
            snake_position[1] += 10

        snake_body.insert(0, list(snake_position))
        if snake_position == food_position:
            food_position = food()
            score += 1
        else:
            snake_body.pop()

        game_surface.fill((0,0,0))
        for position in snake_body:
            pygame.draw.rect(game_surface, (200,200,200), pygame.Rect(position[0], position[1], 10, 10))
        pygame.draw.rect(game_surface, (255,0,0), pygame.Rect(food_position[0], food_position[1], 10, 10))
        text = font.render(str(score), 0, (250,60,80))
        game_surface.blit(text, (480,20))
        if score < 10:
            fps.tick(10)
        if score >= 10 and score < 20:
            fps.tick(20)
        if score >= 20:
            fps.tick(30)

        if snake_position[0] < 0 or snake_position[0] > 490:
            run = False
        if snake_position[1] < 0 or snake_position[1] > 490:
            run = False

        pygame.display.flip()

    pygame.quit()
if __name__ == '__main__':
    main()