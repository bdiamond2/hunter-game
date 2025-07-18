import numpy as np
import pygame
import sys

class Creature:
    # pos: np.array

    def __init__(self, x=0.0, y=0.0, speed=1, detect_range=100):
        self.pos = np.array([x, y])
        self.speed = speed
        self.detect_range = detect_range

    def can_detect(self, other):
        # magnitude of the difference vector = distance
        return np.linalg.norm(other.pos - self.pos) <= self.detect_range

    def vector_to(self, other):
        # unit vector pointing from self to other
        uv = (other.pos - self.pos) / np.linalg.norm(other.pos - self.pos)
        # make magnitude of speed
        return uv * self.speed


class Hunter(Creature):
    def step(self, creatures):
        for c in creatures:
            if isinstance(c, Prey) and self.can_detect(c):
                self.chase(c)
    
    def chase(self, prey):
        vt = self.vector_to(prey)
        self.pos += vt


class Prey(Creature):
    def __init__(self, x=0.0, y=0.0, speed=1, detect_range=100):
        self.is_alert = False
        super().__init__(x=x, y=y, speed=speed, detect_range=detect_range)

    def step(self, creatures):
        for c in creatures:
            if isinstance(c, Hunter) and self.can_detect(c):
                self.make_alert()
                self.flee(c)
                break
            else:
                self.lower_alert()

    def flee(self, hunter):
        vt = self.vector_to(hunter)
        self.pos -= vt

    def make_alert(self):
        if self.is_alert == False:
            self.is_alert = True
            self.detect_range *= 2

    def lower_alert(self):
        if self.is_alert:
            self.is_alert = False
            self.detect_range *= 0.5

def draw_creature(c, screen):
    color = [0, 0, 0]
    if isinstance(c, Prey):
        color = [50, 50, 255]
    elif isinstance(c, Hunter):
        color = [255, 50, 50]

    pygame.draw.circle(screen, color, (c.pos[0], c.pos[1]), 5)

def main():
    # Initialize Pygame
    pygame.init()
    screen_width, screen_height = 600, 600
    screen = pygame.display.set_mode((screen_width, screen_height))
    clock = pygame.time.Clock()

    h = Hunter(x=10.0, y=10.0, speed=1, detect_range=500)
    p = Prey(x=100.0, y=100.0, speed=3, detect_range=100)
    creatures = [h, p]

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        screen.fill((255, 255, 255))

        for c in creatures:
            c.step(creatures)

        for c in creatures:
            draw_creature(c, screen)

        # draw_creature(p, screen)
        # draw_creature(h, screen)

        # pygame.draw.circle(screen, (255, 0, 0), (30, 30), 10)

        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    main()