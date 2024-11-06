import os
import sys
import threading
from typing import Union, Tuple, Dict, Any

import pygame as pg
from pygame.mixer import SoundType
from pygame.surface import SurfaceType

from src.pacman import Pacman
from .constants import GHOST_COLORS, TILE_SIZE, SCORE_COLWIDTH, MODES_TO_ZERO, PATH_FINDER_LOOKUP_TABLE, MOVE_MODES, \
    ROOT_DIR
from .env.agent import Agent
from .ghost import Ghost
from .map import Map
from .utils.action import Action
from .utils.functions import get_image_surface, check_if_hit
from .utils.game_mode import GameMode
from .utils.ghost_state import GhostState
from .utils.path_finder import PathFinder


class Game(object):
    ai_agent: Agent
    channel_background: pg.mixer.Channel
    clock: pg.time.Clock
    snd_intro: SoundType
    snd_default: SoundType
    snd_death: SoundType
    snd_extra_pac: SoundType
    snd_pellet: Dict[int, SoundType]
    snd_eat_gh: SoundType
    snd_eat_fruit: SoundType
    snd_power_pellet: SoundType
    screen_bg: object
    num_digits: dict
    img_game_over: SurfaceType
    img_ready: SurfaceType
    img_life: SurfaceType
    prev_screen: SurfaceType

    def __init__(self, maze: Map, screen: Union[pg.SurfaceType, Any], sounds_active: bool, state_active: bool,
                 **kwargs):
        """

        :param maze:
        :param screen:
        :param sounds_active:
        :param state_active:
        """
        self.ai_agent = kwargs['agent']
        self.screen = screen
        if self.screen is not None:
            self.screen_size = {
                "height": pg.display.Info().current_h,
                "width": pg.display.Info().current_w if not state_active else (pg.display.Info().current_w // 2) - 24
            }
        self.score = 0
        self.total_rewards = 0
        self.mode_timer = 0
        self.ghosts_timer = 0
        self.value_to_draw = 0
        self.pellet_snd_num = 0
        self.sounds_active = sounds_active
        self.state_active = state_active
        self.maze = maze

        self.is_run = True
        self.is_game_run = False
        self.pause = False
        self.draw_value = False

        self.player = Pacman()
        self.path_finder = PathFinder(self.maze.matrix_from_lookup_table(PATH_FINDER_LOOKUP_TABLE))
        self.ghosts = [Ghost(i, GHOST_COLORS[i], self.path_finder) for i in range(0, self.maze.get_number_of_ghosts())]

        self.consecutive_pellets = 0  # Add this line to initialize the counter

        if self.sounds_active:
            self.init_mixer()
            self.load_sounds()

        if self.screen is not None:
            self.load_assets()
            self.maze.build_tile_map()

    def load_assets(self):
        self.screen_bg = get_image_surface(os.path.join(ROOT_DIR, 'res', 'backgrounds', '1.gif'))
        self.num_digits = {
            i: get_image_surface(os.path.join(ROOT_DIR, "res", "text", str(i) + ".gif"))
            for i in range(0, 10)
        }
        self.img_game_over = get_image_surface(os.path.join(ROOT_DIR, "res", "text", "gameover.gif"))
        self.img_ready = get_image_surface(os.path.join(ROOT_DIR, "res", "text", "ready.gif"))
        self.img_life = get_image_surface(os.path.join(ROOT_DIR, "res", "text", "life.gif"))
        self.player.load_frames()
        for ghost in self.ghosts:
            ghost.anim = Ghost.load_ghost_animation(ghost.ghost_color)
            ghost.load_assets()

    def load_sounds(self):
        self.snd_intro = pg.mixer.Sound(os.path.join(ROOT_DIR, "res", "sounds", "levelintro.wav"))
        self.snd_default = pg.mixer.Sound(os.path.join(ROOT_DIR, "res", "sounds", "default.wav"))
        self.snd_death = pg.mixer.Sound(os.path.join(ROOT_DIR, "res", "sounds", "death.wav"))
        self.snd_extra_pac = pg.mixer.Sound(os.path.join(ROOT_DIR, "res", "sounds", "extrapac.wav"))
        self.snd_pellet = {
            0: pg.mixer.Sound(os.path.join(ROOT_DIR, "res", "sounds", "pellet1.wav")),
            1: pg.mixer.Sound(os.path.join(ROOT_DIR, "res", "sounds", "pellet2.wav"))
        }
        self.snd_eat_gh = pg.mixer.Sound(os.path.join(ROOT_DIR, "res", "sounds", "eatgh2.wav"))
        self.snd_eat_fruit = pg.mixer.Sound(os.path.join(ROOT_DIR, "res", "sounds", "eatfruit.wav"))
        self.snd_power_pellet = pg.mixer.Sound(os.path.join(ROOT_DIR, "res", "sounds", "powerpellet.wav"))

    def init_mixer(self):
        pg.mixer.init()
        pg.mixer.set_num_channels(7)
        self.channel_background = pg.mixer.Channel(6)

    def init_game(self):
        self.clock = pg.time.Clock()
        pg.mouse.set_visible(True)

    def play_bkg_sound(self, snd, loops=-1):
        self.channel_background.stop()
        self.channel_background.play(snd, loops=loops)

    def init_players_in_map(self):
        home_x, home_y = self.maze.get_player_home()
        self.player.init_home(home_x, home_y)
        ghosts_home = self.maze.get_ghosts_home(len(self.ghosts))
        resp_x, resp_y = self.maze.get_ghost_respawn_home()
        for i, ghost in enumerate(self.ghosts):
            ghost.init_home(ghosts_home[i]["x"], ghosts_home[i]["y"])
            ghost.init_for_game(path_finder=self.path_finder, player=self.player)
            ghost.init_respawn_home(resp_x, resp_y)

    def init_screen(self):
        self.screen.blit(self.screen_bg, (0, 0))

    def start_game(self, restart=False):
        if restart:
            self.maze.reinit_map()
            self.player.lives = 3

        self.set_mode(0)
        self.init_game()
        self.init_players_in_map()
        self.player.set_vel_to_zero()
        self.game_loop()

    def game_loop(self):
        while self.is_run:
            self.init_screen()
            self.event_loop()

            self.check_game_mode()

            # control pacman
            if self.game_mode in MOVE_MODES:
                self.move_players()

            self.draw()
            pg.display.flip()
            self.clock.tick(60)

            # Add a small positive reward for each time step to encourage longer survival
            self.add_reward(0.01)

    def event_loop(self):
        if self.game_mode in MOVE_MODES:
            if self.ai_agent is None:
                action = Game.check_keyboard_inputs()
            else:
                action = self.ai_agent.act(
                    player_pos=self.player.get_position(),
                    player_pixel_pos=self.player.get_pixel_pos(),
                    matrix=self.maze.get_state_matrix(),
                    ghost_positions=[(ghost.nearest_col, ghost.nearest_row) \
                                     for ghost in self.ghosts],
                    screen=pg.surfarray.pixels3d(self.prev_screen),
                    player_action=self.player.current_action)
                action = int(action)

            if action is not None:
                self.player.change_player_vel(action, self)

        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.quit_game()
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    self.is_run = False
                    self.start_game(restart=True)
                    if self.sounds_active:
                        self.channel_background.stop()

    @staticmethod
    def check_keyboard_inputs() -> Action:
        if pg.key.get_pressed()[pg.K_LEFT]:
            return Action.LEFT
        elif pg.key.get_pressed()[pg.K_RIGHT]:
            return Action.RIGHT
        elif pg.key.get_pressed()[pg.K_UP]:
            return Action.UP
        elif pg.key.get_pressed()[pg.K_DOWN]:
            return Action.DOWN

    def quit_game(self):
        pg.quit()
        sys.exit(0)

    def move_players(self):
        self.player.move(self)
        self.move_ghosts()

        self.update_ghosts_position_in_map()

    def move_ghosts(self):
        for ghost in self.ghosts:
            ghost.move(player=self.player)

    def update_ghosts_position_in_map(self):
        self.maze.update_ghosts_position(self.ghosts)

    def draw(self):
        draw_maze_th = threading.Thread(target=self.maze.draw, args=(self.screen, self.state_active))
        draw_maze_th.start()
        self.player.draw(self.screen, self.game_mode)

        for ghost in self.ghosts:
            ghost.draw(self.screen, self, self.player)

        draw_maze_th.join()
        self.draw_texts()
        self.prev_screen = self.screen.copy()

    def draw_texts(self):
        self.draw_number(self.score, 0, self.maze.shape[0] * TILE_SIZE)

        for i in range(0, self.player.lives):
            self.screen.blit(
                self.img_life, (
                    (self.maze.shape[1] // 2 - 1) * TILE_SIZE + i * 10,
                    self.maze.shape[0] * TILE_SIZE))

        if self.game_mode == 3:
            self.screen.blit(self.img_game_over, self.set_text_center(self.img_game_over))
        elif self.game_mode in [GameMode.ready, GameMode.wait_to_start]:
            self.screen.blit(self.img_ready, self.set_text_center(self.img_ready))

    def set_text_center(self, img: SurfaceType) -> Tuple[int, int]:
        return self.screen_size["width"] // 2 - (img.get_width() // 2), \
               self.screen_size["height"] // 2 - (img.get_height() // 2)

    def set_mode(self, mode: Union[int, GameMode]):
        self.game_mode = GameMode(mode) if type(mode) is int else mode
        if self.game_mode in MODES_TO_ZERO:
            self.mode_timer = 0
        if self.sounds_active:
            self.set_proper_bkg_music()

    def add_score(self, score_to_add: int):
        self.score += score_to_add

    def add_reward(self, reward_to_add: int):
        self.total_rewards += reward_to_add

    def set_proper_bkg_music(self):
        if self.game_mode == GameMode.ready:
            self.play_bkg_sound(self.snd_intro)
        elif self.game_mode == GameMode.normal:
            self.play_bkg_sound(self.snd_default)
        elif self.game_mode == GameMode.hit_ghost:
            self.play_bkg_sound(self.snd_death, 1)
        elif self.game_mode == GameMode.change_ghosts \
                or self.is_at_least_a_ghost_vulnerable():
            self.play_bkg_sound(self.snd_extra_pac)
        else:
            self.channel_background.stop()

    def draw_number(self, num: int, x: int, y: int):
        str_num = str(num)

        for i in range(0, len(str_num)):
            digit = int(str_num[i])
            self.screen.blit(self.num_digits[digit], (x + i * SCORE_COLWIDTH, y))

    def check_game_mode(self):
        if self.game_mode == GameMode.ready:
            if self.mode_timer == 264:
                self.set_mode(1)
        elif self.game_mode == GameMode.normal:
            pass
        elif self.game_mode == GameMode.hit_ghost:
            if self.mode_timer == 90:
                self.restart()
                self.player.lives -= 1
                if self.player.lives == -1:
                    self.set_mode(GameMode.game_over)
                else:
                    self.set_mode(GameMode.wait_to_start)
        elif self.game_mode == GameMode.game_over:
            if pg.key.get_pressed()[pg.K_RETURN]:
                self.start_game(restart=True)
        elif self.game_mode == GameMode.wait_to_start:
            if self.mode_timer == 60:
                self.set_mode(GameMode.normal)
                self.player.vel_x = self.player.speed
        elif self.game_mode == GameMode.wait_after_eating_ghost:
            if self.draw_value and self.mode_timer < 20:
                self.draw_number(self.value_to_draw, self.player.x - 10, self.player.y - 20)
            else:
                self.draw_value = False
                self.value_to_draw = 0
        elif self.game_mode == GameMode.wait_after_finishing_level:
            if self.mode_timer == 40:
                self.set_mode(GameMode.flash_maze)
        elif self.game_mode == GameMode.flash_maze:
            white_set = [10, 30, 50, 70]
            normal_set = [20, 40, 60, 80]

            if not white_set.count(self.mode_timer) == 0:
                self.maze.set_white_color()
            elif not normal_set.count(self.mode_timer) == 0:
                self.maze.set_normal_color()
            elif self.mode_timer == 100:
                self.set_mode(GameMode.black_screen)
        elif self.game_mode == GameMode.wait_after_eating_ghost:

            self.move_ghosts()

            if self.maze.get_number_of_pellets() == 0:
                self.set_mode(GameMode.wait_after_finishing_level)
            elif self.are_all_ghosts_vulnerable():
                self.set_mode(GameMode.change_ghosts)
            elif self.are_all_ghosts_normal():
                self.set_mode(GameMode.normal)
        elif self.game_mode == GameMode.change_ghosts:
            pass
        elif self.game_mode == GameMode.black_screen:
            pass

        if self.game_mode not in [GameMode.wait_after_finishing_level, GameMode.wait_to_start, GameMode.black_screen]:
            self.check_ghosts_state()

        self.mode_timer += 1

    def check_ghosts_state(self):
        if self.is_at_least_a_ghost_vulnerable():
            if self.ghosts_timer == 360:
                self.set_mode(GameMode.normal)
                for ghost in self.ghosts:
                    ghost.set_normal()
            else:
                self.ghosts_timer += 1

        for ghost in self.ghosts:
            ghost.check_ghost_position(self.maze)

    def is_at_least_a_ghost_vulnerable(self) -> bool:
        return [ghost.state == GhostState.vulnerable for ghost in self.ghosts].count(True) > 0

    def are_all_ghosts_vulnerable(self) -> bool:
        return [ghost.state == GhostState.vulnerable for ghost in self.ghosts].count(True) == 4

    def are_all_ghosts_normal(self) -> bool:
        return [ghost.state == GhostState.normal for ghost in self.ghosts].count(True) == 4

    def is_at_least_a_ghost_spectacles(self) -> bool:
        return [ghost.state == GhostState.spectacles for ghost in self.ghosts].count(True) > 0

    def make_ghosts_vulnerable(self):
        self.ghosts_timer = 0
        for ghost in self.ghosts:
            ghost.set_vulnerable()

    def make_ghosts_normal(self):
        self.ghosts_timer = 0
        for ghost in self.ghosts:
            ghost.set_normal()

    def duplicate_vulnerable_ghosts_value(self):
        for ghost in self.ghosts:
            if ghost.state == GhostState.vulnerable:
                ghost.duplicate_value()

    def restart(self):
        self.init_players_in_map()
        self.player.set_start_anim()
        self.player.set_vel_to_zero()

    def draw_ghost_value(self, value):
        self.draw_value = True
        self.value_to_draw = value
    def check_if_hit_something(self):
        for row in range(self.player.nearest_row - 1, self.player.nearest_row + 2):
            for col in range(self.player.nearest_col - 1, self.player.nearest_col + 2):
                if ((self.player.x - (col * TILE_SIZE) < TILE_SIZE) and
                        (self.player.x - (col * TILE_SIZE) > -TILE_SIZE) and
                        (self.player.y - (row * TILE_SIZE) > -TILE_SIZE) and
                        (self.player.y - (row * TILE_SIZE) < TILE_SIZE)):

                    if self.maze.map_matrix[row][col] == 14:
                        # got a pellet
                        self.maze.remove_biscuit(row, col, self.screen is not None)
                        if self.sounds_active:
                            self.snd_pellet[self.pellet_snd_num].play()
                        self.pellet_snd_num = 1 - self.pellet_snd_num
                        self.add_score(10)
                        self.consecutive_pellets += 1
                        self.add_reward(1 + self.consecutive_pellets * 0.5)  # Increased reward for consecutive pellets

                        if self.maze.get_number_of_pellets() == 0:
                            self.set_mode(6)
                    elif self.maze.map_matrix[row][col] == 15:
                        # got a power pellet
                        self.set_mode(9)
                        self.maze.remove_biscuit(row, col, self.screen is not None)
                        if self.sounds_active:
                            self.snd_power_pellet.play()
                        self.add_score(100)
                        self.add_reward(2)
                        self.make_ghosts_vulnerable()
                    elif self.maze.map_matrix[row][col] == 11:
                        # ran into a horizontal door
                        for i in range(self.maze.shape[1]):
                            if not i == col:
                                if self.maze.map_matrix[row][i] == 11:
                                    self.player.x = i * TILE_SIZE
                                    if self.player.vel_x > 0:
                                        self.player.x += TILE_SIZE
                                    else:
                                        self.player.x -= TILE_SIZE
                    elif self.maze.map_matrix[row][col] == 12:
                        # ran into a vertical door
                        for i in range(self.maze.shape[0]):
                            if not i == row:
                                if self.maze.map_matrix[i][col] == 12:
                                    self.player.y = i * TILE_SIZE
                                    if self.player.vel_y > 0:
                                        self.player.y += TILE_SIZE
                                    else:
                                        self.player.y -= TILE_SIZE
                else: 
                    self.consecutive_pellets = 0  # Reset consecutive pellets counter


    def check_collision_with_ghosts(self):
        for ghost in self.ghosts:
            if check_if_hit(self.player.x, self.player.y, ghost.x, ghost.y, (3 * TILE_SIZE) // 4):
                if ghost.state == GhostState.normal:
                    self.set_mode(GameMode.hit_ghost)
                    self.add_reward(-5)
                    self.add_reward(-self.maze.get_number_of_pellets()*0.2)
                elif ghost.state == GhostState.vulnerable:
                    self.add_score(ghost.value)
                    self.add_reward(5)
                    self.draw_ghost_value(ghost.value)
                    self.duplicate_vulnerable_ghosts_value()
                    if self.sounds_active:
                        self.snd_eat_gh.play()

                    ghost.set_spectacles(self.path_finder, self.player)
                    self.set_mode(GameMode.wait_after_eating_ghost)

    def check_if_player_hit_wall(self, x: int, y: int) -> bool:
        num_collision = 0

        for row in range(self.player.nearest_row - 1, self.player.nearest_row + 2):
            for col in range(self.player.nearest_col - 1, self.player.nearest_col + 2):
                if ((x - (col * TILE_SIZE) < TILE_SIZE) and
                        (x - (col * TILE_SIZE) > -TILE_SIZE) and
                        (y - (row * TILE_SIZE) > -TILE_SIZE) and
                        (y - (row * TILE_SIZE) < TILE_SIZE)):
                    try:
                        if self.maze.is_wall(row, col):
                            num_collision += 1
                    except Exception as e:
                        print(e)

        return num_collision > 0

    def get_rba_array(self):
        return pg.surfarray.array3d(self.screen)
