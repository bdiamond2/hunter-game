from __future__ import annotations
import numpy as np
import pygame
import sys
from typing import TypedDict, Tuple


def rotate_vector(v: np.ndarray, theta: float) -> np.ndarray:
    rotation_matrix = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    return rotation_matrix @ v  # matrix multiply


class CreatureInit(TypedDict, total=False):
    pos_x: float
    pos_y: float
    speed: float
    detect_range: float
    max_stamina: float
    stamina_threshold: float
    stamina_recharge: float


class Creature:
    def __init__(self, creature_init: CreatureInit, game_data: HuntersGame):
        ci = creature_init
        self.game_data = game_data
        self.game_data.add_creature(self)  # add the creature to game_data's list

        self.pos = np.array([ci.get("pos_x", 0), ci.get("pos_y", 0)], dtype=float)
        self.speed = ci.get("speed", 1)

        self.detect_range = ci.get("detect_range", 50)

        self.max_stamina = ci.get("max_stamina", 100)
        self.stamina = self.max_stamina
        self.on_stamina_recharge = False
        self.stamina_threshold = ci.get("stamina_threshold", 10)
        self.stamina_recharge = ci.get("stamina_recharge", 1)

        self.is_alive = True

    def can_detect(self, other):
        # magnitude of the difference vector = distance
        return np.linalg.norm(other.pos - self.pos) <= self.detect_range

    def vector_to(self, other):
        # unit vector pointing from self to other
        uv = (other.pos - self.pos) / np.linalg.norm(other.pos - self.pos)
        # make magnitude of speed
        return uv * self.speed

    def increase_stamina(self, amount: float=1):
        self.stamina += amount

        if self.stamina > self.max_stamina:
            self.stamina = self.max_stamina

        if self.stamina >= self.stamina_threshold:
            self.on_stamina_recharge = False

    def decrease_stamina(self, amount=1):
        self.stamina -= amount
        if self.stamina <= 0:
            self.stamina = 0
            self.on_stamina_recharge = True

    def try_move_to(self, new_pos: np.ndarray):
        if not self.game_data.valid_pos(new_pos):
            raise ValueError("Invalid coordinates")
        else:
            pass

        if self.on_stamina_recharge:
            self.increase_stamina(self.stamina_recharge)
            return False
        else:
            self.pos = new_pos
            self.decrease_stamina()
            return True


class Hunter(Creature):
    def __init__(self, hunter_data: CreatureInit, game_data: HuntersGame):
        self.is_stealth = False
        super().__init__(hunter_data, game_data)

    def step(self):
        if not self.is_alive:
            return

        for c in self.game_data.creatures:
            if c == self:
                continue

            if isinstance(c, Prey) and self.can_detect(c) and c.is_alive:
                self.try_chase(c)
            else:
                self.increase_stamina(self.stamina_recharge)

    def try_chase(self, prey: Prey):
        vt = self.vector_to(prey)
        self.try_move_to(self.pos + vt)

        if self.has_caught(prey):
            prey.is_alive = False

    def has_caught(self, prey):
        return np.linalg.norm(prey.pos - self.pos) <= 1



class Prey(Creature):
    def __init__(self, prey_init: CreatureInit, game_data: HuntersGame):
        self.is_alert = False
        super().__init__(prey_init, game_data)

    def step(self):
        if not self.is_alive:
            return

        for c in self.game_data.creatures:
            if c == self:
                continue

            if isinstance(c, Hunter) and self.can_detect(c):
                self.make_alert()
                self.try_flee(c)
                break  # don't flee from multiple predators (for now)
            else:
                self.lower_alert()
                self.increase_stamina(self.stamina_recharge)

    def try_flee(self, hunter):
        vt = self.vector_to(hunter)
        new_pos = self.pos - vt
        loop_count = 0

        # if invalid position, find a new valid one
        while not self.game_data.valid_pos(new_pos):
            vt = rotate_vector(vt, np.pi / 8)
            new_pos = self.pos - vt
            loop_count += 1
            if loop_count > 1000:
                raise RuntimeError(
                    "Infinite loop while searching for valid flee position"
                )

        return self.try_move_to(new_pos)

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

    def valid_pos(self, pos: np.ndarray):
        return (
            not (pos > self.map_dim).any()
            and (pos > np.array([0, 0], dtype=float)).all()
        )


def draw_creature(c: Creature, screen):
    color = [0, 0, 0]
    if not c.is_alive:
        return
    
    if isinstance(c, Prey):
        color = [50, 50, 255]
    elif isinstance(c, Hunter):
        color = [255, 50, 50]

    pygame.draw.circle(screen, color, (c.pos[0], c.pos[1]), 5)


def init_game_data():
    hunter_init: CreatureInit = {
        "pos_x": 10,
        "pos_y": 10,
        "speed": 3,
        "detect_range": 500,
        "max_stamina": 80,
        "stamina_threshold": 70,
    }

    prey_init: CreatureInit = {
        "pos_x": 100,
        "pos_y": 150,
        "speed": 2.5,
        "detect_range": 100,
        "max_stamina": 200,
        "stamina_threshold": 20,
        "stamina_recharge": 10
    }

    game_init: GameInit = {
        "map_dim": (500, 500),
    }

    game_data = HuntersGame(game_init)

    # invoking the constructor adds them to the game
    Hunter(hunter_init, game_data)
    Prey(prey_init, game_data)

    return game_data


def main():
    game_data = init_game_data()

    pygame.init()
    w, h = game_data.map_dim[0], game_data.map_dim[1]
    screen = pygame.display.set_mode((w, h))
    clock = pygame.time.Clock()

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
