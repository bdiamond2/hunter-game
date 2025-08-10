# FROM CHATGPT:

import pygame

pygame.init()

# Window setup
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Control with Keyboard")

# Object setup
rect_x, rect_y = 400, 300
rect_speed = 5
rect_width, rect_height = 50, 50

# Clock for frame rate
clock = pygame.time.Clock()

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Get all keys currently pressed
    keys = pygame.key.get_pressed()

    # Move rectangle with arrow keys
    if keys[pygame.K_LEFT]:
        rect_x -= rect_speed
    if keys[pygame.K_RIGHT]:
        rect_x += rect_speed
    if keys[pygame.K_UP]:
        rect_y -= rect_speed
    if keys[pygame.K_DOWN]:
        rect_y += rect_speed

    # Draw
    screen.fill((0, 0, 0))  # clear screen
    pygame.draw.rect(screen, (255, 0, 0), (rect_x, rect_y, rect_width, rect_height))
    pygame.display.flip()

    clock.tick(60)  # limit to 60 FPS

pygame.quit()
