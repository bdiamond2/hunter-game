from __future__ import annotations
import numpy as np
import pygame
import sys
from typing import TypedDict, List, cast
import random as rdm
# from noise import pnoise1


def rotate_vector(v: np.ndarray, theta: float) -> np.ndarray:
    """
    Takes a given vector and rotates it by a specified angle theta in radians
    """
    rotation_matrix = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    return rotation_matrix @ v  # matrix multiply


def distance_to(p1: np.ndarray, p2: np.ndarray):
    return np.linalg.norm(p1 - p2)


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

        # initialize heading as a unit vector in a random direction
        theta = np.random.uniform(0, 2 * np.pi)
        self.heading = np.array([np.cos(theta), np.sin(theta)])  # remember your unit circle

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
        if rdm.random() < 0.1:  # 10% random chance of moving
            MAX_DTHETA = 0.3
            dtheta = np.random.uniform(-1 * MAX_DTHETA, MAX_DTHETA)
            vec = rotate_vector(self.heading, dtheta)
            new_pos = self.pos + vec
            if not self.game_data.is_valid_pos(new_pos):
                new_pos = self.game_data.center
            # wandering doesn't consume stamina
            return self.try_move_to(new_pos)
        else:
            return False

    def can_detect(self, other: Creature):
        # magnitude of the difference vector = distance
        return np.linalg.norm(other.pos - self.pos) <= self.detect_range

    def run_vector_towards(self, other: Creature):
        # unit vector pointing from self to other
        uv = (other.pos - self.pos) / distance_to(self.pos, other.pos)
        # make magnitude of speed
        return uv * self.speed

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
        while not self.game_data.is_valid_pos(new_pos):
            vec = rotate_vector(vec, np.pi / 16)
            new_pos = self.pos + vec
            loop_count += 1
            if loop_count > 1000:
                raise RuntimeError("Infinite loop while searching for valid position")
        return vec

    def try_move_to(self, new_pos: np.ndarray):
        if not self.game_data.is_valid_pos(new_pos):
            # raise ValueError("Invalid coordinates")
            pass
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

    def step(self):
        if not self.is_alive:
            return

        # flip to True if chasing happens
        did_chase = False

        self.wander()
        # vec = rotate_vector(np.array([1, 0]), 2 * np.pi * rdm.random()) * self.speed
        # new_pos = self.pos + vec
        # if not self.game_data.is_valid_pos(new_pos):
        #     new_pos = np.array([self.game_data.width / 2, self.game_data.height / 2])
        # self.try_move_to(new_pos)

        # for c in self.game_data.prey_list:
        #     if self.can_chase(c):
        #         c = cast(Prey, c)

        #         # determine if we need to change or keep target
        #         self.eval_target(c)
        #     else:
        #         pass  # do nothing for this - not chasable

        # did_chase = self.try_chase(self.target)

        if did_chase:
            self.decrease_stamina()
            # self.eval_target(self.target)  # check if we should keep this target
        else:
            self.increase_stamina(self.stamina_recharge_rate)
            self.wander()

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
            if distance_to(self.pos, p.pos) < self.target_change_factor * distance_to(
                self.pos, self.target.pos
            ):
                # switch to this target
                self.target = p
                return True
            else:
                return False
        else:
            # this shouldn't happen
            raise RuntimeError("eval_target() is broken")

    def can_chase(self, c: Creature):
        return isinstance(c, Prey) and self.can_detect(c) and c.is_alive and c != self

    def try_chase(self, prey: Prey | None):
        if prey == None:
            return False

        vec = self.run_vector_towards(prey)
        new_pos = self.pos + vec
        if not self.game_data.is_valid_pos(new_pos):
            vec = self.find_alt_vector(vec)
            new_pos = self.pos + vec

        did_chase = self.try_move_to(self.pos + vec)

        if did_chase and self.has_caught(prey):
            prey.is_alive = False
        else:
            pass
        return did_chase

    def has_caught(self, prey: Prey):
        return np.linalg.norm(prey.pos - self.pos) <= self.speed


class Prey(Creature):
    def __init__(self, prey_init: CreatureInit, game_data: HuntersGame):
        super().__init__(prey_init, game_data)
        self.is_alert = False
        self.threat_cluster_radius = self.detect_range * 0.1

    def step(self):
        if not self.is_alive:
            return

        did_flee = False

        # if self.get_threat_level() > 0.75:
        if True:
            self.make_alert()
            vec = self.get_util_max_vector()
            if np.isnan(vec).any():
                vec = np.array([0, 0])

            new_pos = self.pos + vec
            if not self.game_data.is_valid_pos(new_pos):
                new_pos = np.array(
                    [
                        self.game_data.width * rdm.random(),
                        self.game_data.height * rdm.random(),
                    ]
                )

            did_flee = self.try_move_to(new_pos)
        else:
            self.lower_alert()
            self.wander()

        if did_flee:
            self.decrease_stamina()
        else:
            self.increase_stamina(self.stamina_recharge_rate)

    def get_threat_level(self):
        x = int(self.pos[0])
        y = int(self.pos[1])
        return self.game_data.threat_field[y, x]

    def get_flee_vector(self):
        x = int(self.pos[0])
        y = int(self.pos[1])
        vec = self.game_data.threat_gradient[y, x]
        mag = np.linalg.norm(vec)
        return -1 * self.speed * vec / mag

    def get_util_max_vector(self):
        x = int(self.pos[0])
        y = int(self.pos[1])
        vec = self.game_data.prey_utility_gradient[y, x]
        mag = np.linalg.norm(vec)
        return self.speed * vec / mag

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
    height: float
    width: float


class HuntersGame:
    def __init__(self, game_init):
        self.creature_list: List[Creature] = []
        self.prey_list: List[Prey] = []
        self.hunter_list: List[Hunter] = []
        self.width = game_init["width"]
        self.height = game_init["height"]
        self.map_dim = np.array([self.width, self.height])
        self.center = self.map_dim / 2.0

        self.threat_field: np.ndarray
        self.threat_gradient: np.ndarray

        self.hunter_utility_field: np.ndarray
        self.hunter_utility_field: np.ndarray

        self.prey_utility_field: np.ndarray
        self.prey_utility_gradient: np.ndarray
        self.UTILITY_AMPLITUDE = 2

    def step(self):
        # print("game_data.step() called")
        for c in self.creature_list:
            c.step()
        self.calc_prey_utility_field()

    def add_creature(self, c: Creature):
        if c not in self.creature_list:
            self.creature_list.append(c)

        if isinstance(c, Prey):
            c = cast(Prey, c)
            self.prey_list.append(c)
        elif isinstance(c, Hunter):
            c = cast(Hunter, c)
            self.hunter_list.append(c)

    def is_valid_pos(self, pos: np.ndarray):
        return (
            not (pos > self.map_dim).any()
            and (pos > np.array([0, 0], dtype=float)).all()
        )

    def calc_prey_utility_field(self):
        # print("\tcalc_prey_utility_field() called")
        h = self.height
        w = self.width

        xs = np.arange(w)
        ys = np.arange(h)
        X, Y = np.meshgrid(xs, ys, indexing="xy")

        # layer representing the disutility from being near hunters
        def uf1():
            # Initialize the product term (start at 1 for multiplication)
            product_term = np.ones((h, w), dtype=np.float64)

            for hunter in self.hunter_list:
                px, py = hunter.pos[0], hunter.pos[1]
                sigma = hunter.detect_range / 3

                dist_sq = (X - px) ** 2 + (Y - py) ** 2
                threat = np.exp(-dist_sq / (2 * sigma**2))

                product_term *= 1 - threat / self.UTILITY_AMPLITUDE
            return self.UTILITY_AMPLITUDE * (1 - product_term) * -1

        # layer representing the utility of being near the center of the map
        def uf2():
            a = w / 2
            b = h / 2
            k = 10
            term_x = ((2 * X - a * 2) / w) ** k
            term_y = ((2 * Y - b * 2) / h) ** k
            return self.UTILITY_AMPLITUDE * np.exp(-1 * (term_x + term_y))
        
        # layer representing disutility from being very close to other prey
        def uf3():
            # Initialize the product term (start at 1 for multiplication)
            product_term = np.ones((h, w), dtype=np.float64)

            for prey in self.prey_list:
                px, py = prey.pos[0], prey.pos[1]
                sigma = 1

                dist_sq = (X - px) ** 2 + (Y - py) ** 2
                disutil = np.exp(-dist_sq / (2 * sigma**2))

                product_term *= 1 - disutil / self.UTILITY_AMPLITUDE
            return self.UTILITY_AMPLITUDE * (1 - product_term) * -1

        self.prey_utility_field = uf1() + uf2()
        # also update partial derivative gradient fields
        gy, gx = np.gradient(self.prey_utility_field)
        self.prey_utility_gradient = np.stack((gx, gy), axis=-1)


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

    # text_val = f"{round(c.pos[0])}, {round(c.pos[1])}"
    # text_surface = pygame.font.SysFont(None, 24).render(text_val, True, (0,0,0))
    # screen.blit(text_surface, (c.pos[0], c.pos[1]))

    pygame.draw.circle(screen, color, (c.pos[0], c.pos[1]), 5)
    # pygame.draw.circle(screen, (0,0,0), (c.pos[0], c.pos[1]), c.detect_range, width=1)


def draw_arr(
    arr: np.ndarray,
    screen,
    low_val: float = 0.0,
    high_val: float = 1.0,
    low_level=0,
    high_level=255,
):
    def value_to_color(val: float):
        val = np.clip(val, low_val, high_val)
        level = int(np.interp(val, [low_val, high_val], [low_level, high_level]))
        return (level, level, level)

    h, w = arr.shape
    step = 20
    px = 20

    for i in range(0, w, step):
        for j in range(0, h, step):
            val = arr[j, i]
            color = value_to_color(val)
            pygame.draw.rect(screen, color, (i - px / 2, j - px / 2, px, px))


def init_game_data():

    # TODO: draw_arr throws exception for non-square worlds
    game_init: GameInit = {"height": 400, "width": 300}
    game_data = HuntersGame(game_init)

    for i in range(0, 2):
        hunter_init: CreatureInit = {
            # "pos_x": rdm.random() * game_data.width,
            # "pos_y": rdm.random() * game_data.height,
            "pos_x": game_data.width / 2,
            "pos_y": game_data.height / 2,
            "speed": 3,
            "detect_range": 400,
            "max_stamina": 1000,
            "stamina_threshold": 1000,
            "stamina_recharge": 10,
        }
        # invoking constructor adds it to the game object
        Hunter(hunter_init, game_data)

    for i in range(0, 200):
        prey_init: CreatureInit = {
            "pos_x": (rdm.random() * game_data.width / 2) + game_data.width / 4,
            "pos_y": (rdm.random() * game_data.height / 2) + game_data.height / 4,
            "speed": 3,
            "detect_range": 100,
            "max_stamina": 100 + rdm.random() * 100,
            "stamina_threshold": 100,
            "stamina_recharge": 5,
        }
        Prey(prey_init, game_data)

    game_data.calc_prey_utility_field()

    return game_data


def main():
    game_data = init_game_data()

    pygame.init()
    w, h = game_data.width, game_data.height
    screen = pygame.display.set_mode((w, h))
    clock = pygame.time.Clock()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        screen.fill((255, 255, 255))

        game_data.step()

        arr = game_data.prey_utility_field
        draw_arr(arr, screen, low_val=np.min(arr), high_val=np.max(arr))

        for c in game_data.creature_list:
            draw_creature(c, screen)

        pygame.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    main()
