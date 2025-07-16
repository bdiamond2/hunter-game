import pygame
import random
import sys
import math


class Circle:
    def __init__(self, x, y, radius, color):
        self.x = x
        self.y = y
        self.radius = radius
        self.color = color

        # mass
        self.mass = math.pi * (self.radius**2)

        # velocity
        self.vx = 0
        self.vy = 0

        # momentum
        self.px = 0
        self.py = 0

    def draw(self, surface):
        pygame.draw.circle(surface, self.color, (self.x, self.y), self.radius)

    def set_velocity(self, vx, vy):
        self.vx = vx
        self.vy = vy
        
        self.px = vx * self.mass
        self.py = vy * self.mass


# Initialize Pygame
pygame.init()
screen_width, screen_height = 800, 600
screen = pygame.display.set_mode((screen_width, screen_height))
clock = pygame.time.Clock()

circles = []
for _ in range(100):
    x = random.uniform(screen.get_width() * 0.1, screen.get_width() * 0.9)
    y = random.uniform(screen.get_height() * 0.1, screen.get_height() * 0.9)

    radius = (random.random() * 5) + 5
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    c = Circle(x, y, radius, color)

    c.set_velocity(random.random() - 0.5, random.random() - 0.5)

    circles.append(c)


# --- Main loop ---
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    screen.fill((255, 255, 255))  # dark background

    # Draw all the circles
    for c1 in circles:
        c1.draw(screen)

        # first add a little jiggle
        # c1.x += (random.random() * 2) - 1
        # c1.y += (random.random() * 2) - 1

        c1.x += c1.vx
        c1.y += c1.vy

        if c1.x + c1.radius >= screen.get_width() or c1.x - c1.radius <= 0:
            c1.set_velocity(c1.vx * -1, c1.vy)

        if c1.y + c1.radius >= screen.get_height() or c1.y - c1.radius <= 0:
            c1.set_velocity(c1.vx, c1.vy * -1)

        # then make them interact
        for c2 in circles:
            if c2 == c1:
                continue

            # if they're touching
            if math.dist([c1.x, c1.y], [c2.x, c2.y]) <= c1.radius + c2.radius:
                dx = c2.x - c1.x
                dy = c2.y - c1.y

                mag = math.dist([0, 0], [dx, dy])
                damp = 10
                uv = [dx / (mag * damp), dy / (mag * damp)]

                c2.set_velocity(c2.vx + uv[0], c2.vy + uv[1])
                c1.set_velocity(c1.vx - uv[0], c1.vy - uv[1])

    pygame.display.flip()
    clock.tick(60)
