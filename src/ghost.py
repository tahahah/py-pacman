import os
from typing import Optional, Tuple
import math

import pygame as pg
from pygame.surface import SurfaceType

from src.constants import TILE_SIZE, VULNERABLE_GHOST_COLOR, WHITE_GHOST_COLOR, ROOT_DIR
from src.map import Map
from src.pacman import Pacman
from src.utils.functions import get_image_surface
from src.utils.game_mode import GameMode
from src.utils.ghost_state import GhostState
from src.utils.path_finder import PathFinder


class Ghost(object):
    img_glasses: SurfaceType

    def __init__(self, ghost_id, ghost_color, path_finder: PathFinder):
        self.x = 0
        self.y = 0
        self.vel_x = 0
        self.vel_y = 0
        self.speed = 2
        self.nearest_row = 0
        self.nearest_col = 0
        self.id = ghost_id
        self.state = GhostState.normal
        self.value = 0
        self.path_finder = path_finder

        self.home_x = 0
        self.home_y = 0
        self.respawn_x = None
        self.respawn_y = None

        self.current_path = None
        self.ghost_color = ghost_color

        self.anim = {}
        self.anim_fram = 1
        self.anim_delay = 0

    @staticmethod
    def load_ghost_animation(color):
        anim = {}

        for i in range(1, 7):
            anim[i] = pg.image.load(
                os.path.join(ROOT_DIR, "res", "sprite", "ghost " + str(i) + ".gif")).convert()

            # change the ghost color in this frame
            for y in range(0, TILE_SIZE):
                for x in range(0, TILE_SIZE):

                    if anim[i].get_at((x, y)) == (255, 0, 0, 255):
                        # default, red ghost body color
                        anim[i].set_at((x, y), color)

        return anim

    def load_assets(self):
        self.img_glasses = get_image_surface(os.path.join(ROOT_DIR, "res", "tiles", "glasses.gif"))

    def draw(self, screen, game, player: Pacman):

        pupil_set = None

        if game.game_mode == GameMode.game_over:
            return False

        # ghost eyes --
        for y in range(6, 12):
            for x in [5, 6, 8, 9]:
                self.anim[self.anim_fram].set_at((x, y), (248, 248, 248, 255))
                self.anim[self.anim_fram].set_at((x + 9, y), (248, 248, 248, 255))

                if player.x > self.x and player.y > self.y:
                    # player is to lower-right
                    pupil_set = (8, 9)
                elif player.x < self.x and player.y > self.y:
                    # player is to lower-left
                    pupil_set = (5, 9)
                elif player.x > self.x and player.y < self.y:
                    # player is to upper-right
                    pupil_set = (8, 6)
                elif player.x < self.x and player.y < self.y:
                    # player is to upper-left
                    pupil_set = (5, 6)
                else:
                    pupil_set = (5, 9)

        for y in range(pupil_set[1], pupil_set[1] + 3):
            for x in range(pupil_set[0], pupil_set[0] + 2, 1):
                self.anim[self.anim_fram].set_at((x, y), (0, 0, 255, 255))
                self.anim[self.anim_fram].set_at((x + 9, y), (0, 0, 255, 255))

        if self.state == GhostState.normal:
            # draw regular ghost (this one)
            screen.blit(self.anim[self.anim_fram],
                        (self.x, self.y))
        elif self.state == GhostState.vulnerable:
            if game.is_at_least_a_ghost_vulnerable() and game.ghosts_timer < 260:
                # blue
                screen.blit(Ghost.load_ghost_animation(VULNERABLE_GHOST_COLOR)[self.anim_fram],
                            (self.x, self.y))
            else:
                # blue/white flashing
                temp_timer_i = int((360 - game.ghosts_timer) / 10)
                if temp_timer_i == 1 or temp_timer_i == 3 or temp_timer_i == 5 or temp_timer_i == 7 or temp_timer_i == 9:
                    screen.blit(Ghost.load_ghost_animation(WHITE_GHOST_COLOR)[self.anim_fram],
                                (self.x, self.y))
                else:
                    screen.blit(Ghost.load_ghost_animation(VULNERABLE_GHOST_COLOR)[self.anim_fram],
                                (self.x, self.y))

        elif self.state == GhostState.spectacles:
            # draw glasses
            screen.blit(self.img_glasses,
                        (self.x, self.y))

        if game.game_mode == GameMode.wait_after_finishing_level or game.game_mode == GameMode.flash_maze:
            # don't animate ghost if the level is complete
            return False

        self.anim_delay += 1

        if self.anim_delay == 2:
            self.anim_fram += 1

            if self.anim_fram == 7:
                # wrap to beginning
                self.anim_fram = 1

            self.anim_delay = 0

    def move(self, player: Pacman):
        self.x += self.vel_x
        self.y += self.vel_y

        self.nearest_row = int(((self.y + TILE_SIZE / 2) / TILE_SIZE))
        self.nearest_col = int(((self.x + TILE_SIZE / 2) / TILE_SIZE))

        self.check_ghost_state(self.path_finder, player)

        if (self.x % TILE_SIZE) == 0 and (self.y % TILE_SIZE) == 0:
            # ghost is lined up with the grid
            if self.current_path is not None and len(self.current_path) > 0:
                self.follow_next_path()
            else:
                self.find_path(path_finder=self.path_finder, player=player)

    def find_path(self, path_finder: PathFinder, player: Optional[Pacman], random=False):
        if random:
            rnd_col, rnd_row = path_finder.get_random_allow_position()
            self.current_path = path_finder.get_min_path(
                self.nearest_col,
                self.nearest_row,
                rnd_col,
                rnd_row
            )
        elif self.state in [GhostState.normal, GhostState.vulnerable]:
            self.current_path = path_finder.get_min_path(
                self.nearest_col,
                self.nearest_row,
                player.nearest_col,
                player.nearest_row
            )
        elif self.state == GhostState.spectacles:
            self.current_path = path_finder.get_min_path(
                self.nearest_col,
                self.nearest_row,
                self.respawn_x,
                self.respawn_y
            )
        else:
            rnd_col, rnd_row = path_finder.get_random_allow_position()
            self.current_path = path_finder.get_min_path(
                self.nearest_col,
                self.nearest_row,
                rnd_col,
                rnd_row
            )
        self.follow_next_path()

    def set_normal(self):
        self.state = GhostState.normal
        self.value = 0
        self.speed = 2

    def duplicate_value(self):
        self.value *= 2

    def init_home(self, home_x: int, home_y: int):
        self.home_x = home_x
        self.home_y = home_y
        self.x = self.home_x * TILE_SIZE
        self.y = self.home_y * TILE_SIZE
        self.nearest_col = home_x
        self.nearest_row = home_y

    def set_vulnerable(self):
        self.state = GhostState.vulnerable
        self.value = 200
        self.speed = 2

    def is_vulnerable(self) -> bool:
        return self.state == GhostState.vulnerable

    def set_spectacles(self, path_finder: PathFinder, player: Pacman):
        self.state = GhostState.spectacles
        self.value = 0
        self.speed = 8
        self.x = self.nearest_col * TILE_SIZE
        self.y = self.nearest_row * TILE_SIZE
        self.find_path(path_finder, player)

    def follow_next_path(self):
        if self.current_path is not None and len(self.current_path) > 0:
            if self.current_path[0] == "L":
                self.vel_x, self.vel_y = -self.speed, 0
            elif self.current_path[0] == "R":
                self.vel_x, self.vel_y = self.speed, 0
            elif self.current_path[0] == "U":
                self.vel_x, self.vel_y = 0, -self.speed
            elif self.current_path[0] == "D":
                self.vel_x, self.vel_y = 0, self.speed
            self.current_path = self.current_path[1:]

    def init_respawn_home(self, resp_x: int, resp_y: int):
        self.respawn_x = resp_x
        self.respawn_y = resp_y

    def init_for_game(self, path_finder: PathFinder, player: Pacman):
        self.vel_x = 0
        self.vel_y = 0
        self.state = GhostState.normal
        self.find_path(path_finder, player, True)

    def check_ghost_state(self, path_finder: PathFinder, player: Pacman):
        if self.nearest_row == self.respawn_y \
            and self.nearest_col == self.respawn_x \
                and self.state == GhostState.spectacles:
            self.state = GhostState.normal
            self.speed = 2
            self.find_path(path_finder, player)

    def check_ghost_position(self, maze: Map):
        try:
            maze.state_matrix[self.nearest_row][self.nearest_col]
        except IndexError:
            self.x = self.home_x * TILE_SIZE
            self.y = self.home_y * TILE_SIZE
            self.nearest_col = self.home_x
            self.nearest_row = self.home_y
            self.find_path(self.path_finder, None, True)

    def get_pixel_position(self) -> Tuple[int, int]:
        """
        :return: the (x, y) pixel position of the ghost
        """
        return self.x, self.y

    def get_position(self) -> Tuple[int, int]:
        """
        :return: the (x, y) position of the ghost
        """
        return self.nearest_col, self.nearest_row

    def print_position(self):
        print(f"Ghost_{self.id} col: {self.nearest_col}, row: {self.nearest_row}")

    def distance_to_pacman(self, pacman_x, pacman_y):
        return math.sqrt((self.x - pacman_x)**2 + (self.y - pacman_y)**2)
