import numpy as np
import pygame
import sys
from typing import TypedDict, List, Tuple

class CreatureData(TypedDict):
    pos_x: float
    pos_y: float
    speed: float
    detect_range: float

class Creature:
    def __init__(self, creature_data: CreatureData):
        cd = creature_data
        self.pos = np.array([cd.get("pos_x", 0.0), cd.get("pos_y", 0.0)])
        self.speed = cd.get("speed", 1.0)
        self.detect_range = cd.get("detect_range", 10.0)

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
    def __init__(self, prey_data: CreatureData):
        self.is_alert = False
        super().__init__(prey_data)

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

class GameData(TypedDict):
    creatures: List[Creature]
    map_dim: Tuple[float, float]

class HuntersGame:
    # game_data keys
    CREATURES = 0
    MAP_DIM = 1

    def __ini__(self, game_data):
        self.creatures = game_data["creatures"]
        self.map_dim = game_data["map_dim"]

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

    hunter_data: CreatureData = {
        "pos_x": 10.0,
        "pos_y": 10.0,
        "speed": 1.0,
        "detect_range": 500
    }

    prey_data: CreatureData = {
        "pos_x": 100.0,
        "pos_y": 150.0,
        "speed": 3.0,
        "detect_range": 100.0
    }

    h = Hunter(hunter_data)
    p = Prey(prey_data)
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

        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    main()