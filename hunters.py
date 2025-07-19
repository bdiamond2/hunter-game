from __future__ import annotations
import numpy as np
import pygame
import sys
from typing import TypedDict, Tuple


class CreatureInit(TypedDict, total=False):
    pos_x: float
    pos_y: float
    speed: float
    detect_range: float


class Creature:
    def __init__(self, creature_init: CreatureInit, game_data: HuntersGame):
        ci = creature_init
        self.game_data = game_data
        self.game_data.add_creature(self)  # add the creature to game_data's list

        self.pos = np.array([ci.get("pos_x", 0), ci.get("pos_y", 0)], dtype=float)
        self.speed = ci.get("speed", 1)
        self.detect_range = ci.get("detect_range", 10)

    def can_detect(self, other):
        # magnitude of the difference vector = distance
        return np.linalg.norm(other.pos - self.pos) <= self.detect_range

    def vector_to(self, other):
        # unit vector pointing from self to other
        uv = (other.pos - self.pos) / np.linalg.norm(other.pos - self.pos)
        # make magnitude of speed
        return uv * self.speed
    
    def move_to(self, new_pos: np.ndarray):
        if (new_pos > self.game_data.map_dim).any():  # either dimension beyond map_dim
            raise ValueError("Invalid coordinates")
        else:
            self.pos = new_pos


class Hunter(Creature):
    def __init__(self, hunter_data: CreatureInit, game_data: HuntersGame):
        self.is_stealth = False
        super().__init__(hunter_data, game_data)

    def step(self):
        for c in self.game_data.creatures:
            if c == self:
                continue
                
            if isinstance(c, Prey) and self.can_detect(c):
                self.chase(c)

    def chase(self, prey):
        vt = self.vector_to(prey)
        # self.pos += vt
        self.move_to(self.pos + vt)


class Prey(Creature):
    def __init__(self, prey_init: CreatureInit, game_data: HuntersGame):
        self.is_alert = False
        super().__init__(prey_init, game_data)

    def step(self):
        for c in self.game_data.creatures:
            if c == self:
                continue

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


class GameInit(TypedDict):
    # creatures: List[Creature]
    map_dim: Tuple[float, float]


class HuntersGame:
    def __init__(self, game_init):
        self.creatures = []
        self.map_dim = np.array([game_init["map_dim"][0], game_init["map_dim"][1]])

    def step(self):
        for c in self.creatures:
            c.step()  # pass game data

    def add_creature(self, c: Creature):
        if c not in self.creatures:
            self.creatures.append(c)


def draw_creature(c, screen):
    color = [0, 0, 0]
    if isinstance(c, Prey):
        color = [50, 50, 255]
    elif isinstance(c, Hunter):
        color = [255, 50, 50]

    pygame.draw.circle(screen, color, (c.pos[0], c.pos[1]), 5)


def init_game_data():
    hunter_init: CreatureInit = {
        "pos_x": 10,
        "pos_y": 10,
        # "speed": 1,
        "detect_range": 500,
    }

    prey_init: CreatureInit = {
        "pos_x": 100,
        "pos_y": 150,
        "speed": 3,
        "detect_range": 100,
    }

    game_init: GameInit = {
        # "creatures": [Hunter(hunter_data), Prey(prey_data)],
        "map_dim": (500, 500),
    }

    game_data = HuntersGame(game_init)
    Hunter(hunter_init, game_data)
    Prey(prey_init, game_data)

    return game_data


def main():
    pygame.init()
    screen_width, screen_height = 600, 600
    screen = pygame.display.set_mode((screen_width, screen_height))
    clock = pygame.time.Clock()
    
    game_data = init_game_data()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        screen.fill((255, 255, 255))

        game_data.step()

        for c in game_data.creatures:
            draw_creature(c, screen)

        pygame.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    main()
