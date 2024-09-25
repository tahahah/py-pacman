import os
from typing import Dict, List, Tuple

import numpy as np
import pygame as pg

from .constants import (EDGE_LIGHT_COLOR, EDGE_SHADOW_COLOR, FILL_COLOR,
                        GHOST_VALUE, GHOST_VULNERABLE_VALUE, PELLET_COLOR,
                        ROOT_DIR, STATE_COLOR_LOOKUP_TABLE, STATE_LOOKUP_TABLE,
                        TILE_LOOKUP_TABLE, TILE_SIZE, WHITE_EDGE_LIGHT_COLOR,
                        WHITE_EDGE_SHADOW_COLOR, WHITE_FILL_COLOR)
from .utils.functions import get_image_surface
from .utils.ghost_state import GhostState


class Map:

    def __init__(self, layout_name):
        self.map_matrix = np.loadtxt(os.path.join(ROOT_DIR, 'res', 'layouts', layout_name + '.lay')).astype(int)
        self.shape = self.map_matrix.shape
        self.edge_light_color = (0, 0, 255, 255)
        self.edge_shadow_color = (0, 0, 255, 255)
        self.fill_color = (0, 0, 0, 255)
        self.pellet_color = (255, 255, 255, 255)
        self.layout_name = layout_name
        self.state_matrix = self.matrix_from_lookup_table(STATE_LOOKUP_TABLE)

    def is_wall(self, row: int, col: int) -> bool:
        if row > self.shape[0] or row < 0:
            return True

        if col > self.shape[1] or col < 0:
            return True

        if 0 <= row < self.shape[0] and 0 <= col < self.shape[1]:
            return self.map_matrix[row][col] in range(16, 33)
        else:
            return False

    def is_ghost(self, x: int, y: int) -> bool:
        return self.state_matrix[x:int, y:int] == -1

    def is_biscuit(self, x: int, y: int) -> bool:
        return self.state_matrix[x:int, y:int] == 1

    def is_pill(self, x: int, y: int) -> bool:
        return self.map_matrix[x:int, y:int] == 15

    def remove_biscuit(self, x: int, y: int, is_screen_active: bool = True):
        self.map_matrix[x][y] = 10
        self.state_matrix[x][y] = 0
        if is_screen_active:
            self.tile_map[x, y] = get_image_surface(os.path.join(
                ROOT_DIR,
                "res",
                "tiles",
                TILE_LOOKUP_TABLE[self.map_matrix[x][y]]
            ))

    def get_player_home(self) -> Tuple[int, int]:
        home_y, home_x = np.where(self.map_matrix == 40)
        return int(home_x[0]), int(home_y[0])

    def get_number_of_ghosts(self):
        number_of_ghosts = 0

        for i in range(33, 37):
            if np.any(self.map_matrix == i):
                number_of_ghosts += 1

        return number_of_ghosts

    def draw(self, screen, draw_state: bool):
        for row in range(self.shape[0]):
            for col in range(self.shape[1]):
                screen.blit(self.tile_map[row, col], (col * TILE_SIZE, row * TILE_SIZE))

        if draw_state:
            for y in range(self.shape[0]):
                for x in range(self.shape[1] + 1, (self.shape[1] * 2) + 1):
                    real_x = x - self.shape[1] - 1
                    poly = [(x * TILE_SIZE, y * TILE_SIZE),
                            ((x + 1) * TILE_SIZE, y * TILE_SIZE),
                            ((x + 1) * TILE_SIZE, (y + 1) * TILE_SIZE),
                            (x * TILE_SIZE, (y + 1) * TILE_SIZE)]
                    pg.draw.polygon(screen, STATE_COLOR_LOOKUP_TABLE[self.state_matrix[y][real_x]], poly, 0)

    def build_tile_map(self):
        self.tile_map = {}

        for i in range(self.map_matrix.shape[0]):
            for j in range(self.map_matrix.shape[1]):
                if self.map_matrix[i][j] in [40, 11, 12, 33, 34, 35, 36]:
                    # position of pacman, the ghost or the doors
                    self.tile_map[i, j] = get_image_surface(
                        os.path.join(ROOT_DIR, "res", "tiles", TILE_LOOKUP_TABLE[10]))
                else:
                    self.tile_map[i, j] = get_image_surface(os.path.join(
                        ROOT_DIR,
                        "res",
                        "tiles",
                        TILE_LOOKUP_TABLE[self.map_matrix[i][j]]
                    ))
                self.recolor_tile(self.tile_map[i, j])

    def matrix_from_lookup_table(self, lookup_table: Dict[int, int]) -> np.ndarray:
        # Using np.empty with dtype=int to avoid uninitialized values and casting issues
        matrix = np.empty(shape=(self.map_matrix.shape[0], self.map_matrix.shape[1]), dtype=int)

        for i in range(self.map_matrix.shape[0]):
            for j in range(self.map_matrix.shape[1]):
                matrix[i][j] = lookup_table[self.map_matrix[i][j]]

        return matrix

    def get_number_of_pellets(self) -> int:
        return (self.state_matrix == 1).sum()

    def recolor_tile(self, tile):
        for y in range(0, TILE_SIZE, 1):
            for x in range(0, TILE_SIZE, 1):

                if tile.get_at((x, y)) == EDGE_LIGHT_COLOR:
                    # wall edge
                    tile.set_at((x, y), self.edge_light_color)

                elif tile.get_at((x, y)) == FILL_COLOR:
                    # wall fill
                    tile.set_at((x, y), self.fill_color)

                elif tile.get_at((x, y)) == EDGE_SHADOW_COLOR:
                    # pellet color
                    tile.set_at((x, y), self.edge_shadow_color)

                elif tile.get_at((x, y)) == PELLET_COLOR:
                    # pellet color
                    tile.set_at((x, y), self.pellet_color)

    def get_map_sizes(self) -> Tuple[int, int]:
        return self.shape[1] * TILE_SIZE, (self.shape[0] + 1) * TILE_SIZE

    def get_ghosts_home(self, num_ghosts: int) -> List[Dict[str, int]]:
        ghosts_home = []

        for i in range(0, num_ghosts):
            home_y, home_x = np.where(self.map_matrix == 33 + i)
            home_y, home_x = int(home_y[0]), int(home_x[0])
            ghosts_home.append({"x": home_x, "y": home_y})

        return ghosts_home

    def reinit_map(self):
        self.map_matrix = np.loadtxt(os.path.join(ROOT_DIR, 'res', 'layouts', self.layout_name + '.lay')).astype(int)
        self.state_matrix = self.matrix_from_lookup_table(STATE_LOOKUP_TABLE)
        self.build_tile_map()

    def get_ghost_respawn_home(self) -> Tuple[int, int]:
        g_home = 35 if np.any(self.map_matrix == 35) else 33
        home_y, home_x = np.where(self.map_matrix == g_home)
        return int(home_x[0]), int(home_y[0])

    def set_white_color(self):
        self.edge_light_color = WHITE_EDGE_LIGHT_COLOR
        self.edge_shadow_color = WHITE_EDGE_SHADOW_COLOR
        self.fill_color = WHITE_FILL_COLOR
        self.build_tile_map()

    def set_normal_color(self):
        self.edge_light_color = (0, 0, 255, 255)
        self.edge_shadow_color = (0, 0, 255, 255)
        self.fill_color = (0, 0, 0, 255)
        self.build_tile_map()

    def get_state_matrix(self) -> np.ndarray:
        return self.state_matrix

    def update_ghosts_position(self, ghosts: List):

        self.state_matrix[self.state_matrix == GHOST_VALUE] = -99999
        self.state_matrix[self.state_matrix == GHOST_VULNERABLE_VALUE] = -99999

        a = np.where(self.state_matrix == -99999)
        pos = [(x, y) for x, y in zip(a[1], a[0])]

        for x, y in pos:
            self.state_matrix[y][x] = STATE_LOOKUP_TABLE[self.map_matrix[y][x]]

        for ghost in ghosts:
            value = GHOST_VULNERABLE_VALUE \
                if ghost.state == GhostState.vulnerable or ghost.state == GhostState.spectacles \
                else GHOST_VALUE
            try:
                self.state_matrix[ghost.nearest_row][ghost.nearest_col] = value
            except IndexError:
                id = ghost.id
                positions = np.where(self.map_matrix == 33 + id)
                pos = [(x, y) for x, y in zip(positions[1], positions[0])][0]
                self.state_matrix[pos[1]][pos[0]] = GHOST_VALUE
            if ghost.nearest_row != ghost.home_y and ghost.nearest_col != ghost.home_x:
                self.state_matrix[ghost.home_y][ghost.home_x] = 0
