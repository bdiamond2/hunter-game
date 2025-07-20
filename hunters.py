from __future__ import annotations
import numpy as np
import pygame
import sys
from typing import TypedDict, Tuple, List, cast
import random as rdm
import math


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
        self.stamina_recharge_threshold = ci.get("stamina_threshold", 10)
        self.stamina_recharge_rate = ci.get("stamina_recharge", 1)

        self.is_alive = True

    def step(self):
        pass  # default behavior do nothing

    def wander(self):
        if rdm.random() < 0.1: # 10% random chance of moving
            pass
        else:
            pass

    def can_detect(self, other: Creature):
        # magnitude of the difference vector = distance
        return np.linalg.norm(other.pos - self.pos) <= self.detect_range

    def run_vector_towards(self, other: Creature):
        # unit vector pointing from self to other
        uv = (other.pos - self.pos) / self.distance_to(other)
        # make magnitude of speed
        return uv * self.speed
    
    def distance_to(self, other: Creature):
        return np.linalg.norm(other.pos - self.pos)

    def increase_stamina(self, amount: float = 1):
        self.stamina += amount

        if self.stamina > self.max_stamina:
            self.stamina = self.max_stamina

        if self.stamina >= self.stamina_recharge_threshold:
            self.on_stamina_recharge = False

    def decrease_stamina(self, amount=1):
        self.stamina -= amount
        if self.stamina <= 0:
            self.stamina = 0
            self.on_stamina_recharge = True

    def find_alt_vector(self, vec: np.ndarray):
        new_pos = self.pos + vec

        loop_count = 0
        while not self.game_data.valid_pos(new_pos):
            vec = rotate_vector(vec, np.pi / 16)
            new_pos = self.pos + vec
            loop_count += 1
            if loop_count > 1000:
                raise RuntimeError(
                    "Infinite loop while searching for valid position"
                )
        return vec

    def try_move_to(self, new_pos: np.ndarray):
        if not self.game_data.valid_pos(new_pos):
            raise ValueError("Invalid coordinates")
        else:
            pass

        if self.on_stamina_recharge:
            return False
        else:
            self.pos = new_pos
            return True


class Hunter(Creature):
    def __init__(self, hunter_data: CreatureInit, game_data: HuntersGame):
        self.is_stealth = False
        self.target = None
        self.target_change_factor = 0.5
        super().__init__(hunter_data, game_data)

    def eval_target(self, p: Prey | None):
        if p == None:
            return False
        
        if not p.is_alive:
            if self.target == p:
                self.target = None
            return False

        # acquire first lock
        if self.target == None:
            self.target = p
            return True
        # already locked onto this
        elif self.target == p:
            # assess if prey is dead
            self.target = None if not p.is_alive else self.target
            return True
        # check if we should change targets
        elif self.target != p:
            # if the distance to this new target is less than the tgt chg factor...
            if self.distance_to(p) < self.target_change_factor * self.distance_to(self.target):
                # switch to this target
                self.target = p
                return True
            else:
                return False
        else:
            # this shouldn't happen
            raise RuntimeError("eval_target() is broken")

    def step(self):
        if not self.is_alive:
            return

        # flip to True if chasing happens
        did_chase = False

        for c in self.game_data.creatures:
            if self.can_chase(c):
                c = cast(Prey, c)

                # determine if we need to change or keep target
                self.eval_target(c)
            else:
                pass  # do nothing for this - not chasable
        
        did_chase = self.try_chase(self.target)
        
        if did_chase:
            self.decrease_stamina()
            self.eval_target(self.target) # check if we should keep this target
        else:
            self.increase_stamina(self.stamina_recharge_rate)
            self.wander()
        

    def can_chase(self, c: Creature):
        return isinstance(c, Prey) and self.can_detect(c) and c.is_alive and c != self

    def try_chase(self, prey: Prey | None):
        if prey == None:
            return False
        
        vec = self.run_vector_towards(prey)
        new_pos = self.pos + vec
        if not self.game_data.valid_pos(new_pos):
            vec = self.find_alt_vector(vec)
            new_pos = self.pos + vec
        
        did_chase = self.try_move_to(self.pos + vec)

        if did_chase and self.has_caught(prey):
            prey.is_alive = False
        else:
            pass
        return did_chase

    def has_caught(self, prey: Prey):
        return np.linalg.norm(prey.pos - self.pos) <= 1


class Prey(Creature):
    def __init__(self, prey_init: CreatureInit, game_data: HuntersGame):
        self.is_alert = False
        super().__init__(prey_init, game_data)

    def step(self):
        if not self.is_alive:
            return
        
        # satisfaction from next decision
        util: float = 0
        
        did_flee = False
        max_threat_level = 0
        max_threat_hunter: Hunter | None = None

        for c in self.game_data.creatures:
            if self.is_threat(c):
                c = cast(Hunter, c)
                threat_level = max(self.get_threat_level(c), 0)  # extra precaution against negative threats
                if threat_level > max_threat_level:
                    max_threat_level = threat_level
                    max_threat_hunter = c
                else:
                    pass # not highest threat
            else:
                pass  # not a threat

        if max_threat_level == 0:  # no threat found
            self.lower_alert()
        else:
            self.make_alert()
            did_flee = self.try_flee(max_threat_hunter)

        if did_flee:
            self.decrease_stamina()
        else:    
            self.increase_stamina(self.stamina_recharge_rate)
            self.wander()

    def get_threat_level(self, hunter: Hunter):
        if not self.is_threat(hunter):
            return 0

        # "normalized" distance to predator, as pct of detect_range
        d_norm = self.distance_to(hunter) / self.detect_range
        
        # logistic signmoid function
        if d_norm <= 0.01:
            threat_level = 1
        elif d_norm >= 0.99:
            threat_level = 0
        else:
            # sigmoid logistic function
            k = 10
            threat_level = 1 / (1 + math.exp((k * (d_norm - 0.5))))

        return threat_level

    def get_utility(self, new_pos: np.ndarray):
        pass
        # 1.) highest utility: gaining max distance from predator
        # 2.) gaining less than max distance from predator

    def try_flee(self, hunter: Hunter | None):
        if hunter == None:
            return False

        vec = self.run_vector_towards(hunter) * -1
        new_pos = self.pos + vec

        # if invalid position, find a new valid one
        if not self.game_data.valid_pos(new_pos):
            vec = self.find_alt_vector(vec)
            new_pos = self.pos + vec

        return self.try_move_to(new_pos)
    
    def is_threat(self, c: Creature):
        return c.is_alive and c != self and isinstance(c, Hunter) and self.can_detect(c)

    def make_alert(self):
        if self.is_alert == False:
            self.is_alert = True
            self.detect_range *= 2

    def lower_alert(self):
        if self.is_alert:
            self.is_alert = False
            self.detect_range *= 0.5


class GameInit(TypedDict):
    map_dim: Tuple[float, float]


class HuntersGame:
    def __init__(self, game_init):
        self.creatures: List[Creature] = []
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
        if c.is_alert:
            color = [50, 50, 255]
        else:
            color = [50, 255, 50]
    elif isinstance(c, Hunter):
        if c.on_stamina_recharge:
            color = [255, 0, 255]
        elif c.target != None:
            color = [255, 0, 0]
        else:
            color = [255, 200, 0]

    pygame.draw.circle(screen, color, (c.pos[0], c.pos[1]), 5)


def init_game_data():

    game_init: GameInit = {
        "map_dim": (500, 500),
    }
    game_data = HuntersGame(game_init)

    for i in range(0, 2):
        hunter_init: CreatureInit = {
            "pos_x": 50 * (i + 1),
            "pos_y": 50 * (i + 3),
            "speed": 3,
            "detect_range": 400,
            "max_stamina": 200,
            "stamina_threshold": 180,
        }
        # invoking constructor adds it to the game object
        Hunter(hunter_init, game_data)

    for i in range(0, 20):
        prey_init: CreatureInit = {
            "pos_x": rdm.random() * 100,
            "pos_y": rdm.random() * 150,
            "speed": 2,
            "detect_range": 100,
            "max_stamina": 200,
            "stamina_threshold": 20,
            "stamina_recharge": 5,
        }
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
