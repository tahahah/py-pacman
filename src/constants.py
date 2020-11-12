GHOST_COLORS = {
    0: (255, 0, 0, 255),
    1: (255, 128, 255, 255),
    2: (128, 255, 255, 255),
    3: (255, 128, 0, 255),
    4: (50, 50, 255, 255),
    5: (255, 255, 255, 255)
}

IMG_EDGE_LIGHT_COLOR = (255, 206, 255, 255)
IMG_FILL_COLOR = (132, 0, 132, 255)
IMG_EDGE_SHADOW_COLOR = (255, 0, 255, 255)
IMG_PELLET_COLOR = (128, 0, 128, 255)

SCREEN_TILE_SIZE_HEIGHT: int = 23
SCREEN_TILE_SIZE_WIDTH: int = 19

TILE_SIZE: int = 24

SCORE_COLWIDTH: int = 13

SCREEN_WIDTH: int = SCREEN_TILE_SIZE_WIDTH * TILE_SIZE
SCREEN_HEIGHT: int = SCREEN_TILE_SIZE_HEIGHT * TILE_SIZE

TILE_LOOKUP_TABLE = {
    10: 'blank.gif',
    11: 'door-h.gif',
    12: 'door-v.gif',
    13: 'ghost-door.gif',
    14: 'pellet.gif',
    15: 'pellet-power.gif',
    16: 'wall-corner-ll.gif',
    17: 'wall-corner-lr.gif',
    18: 'wall-corner-ul.gif',
    19: 'wall-corner-ur.gif',
    20: 'wall-end-b.gif',
    21: 'wall-end-l.gif',
    22: 'wall-end-r.gif',
    23: 'wall-end-t.gif',
    24: 'wall-nub.gif',
    25: 'wall-straight-horiz.gif',
    26: 'wall-straight-vert.gif',
    27: 'wall-t-bottom.gif',
    28: 'wall-t-left.gif',
    29: 'wall-t-right.gif',
    30: 'wall-t-top.gif',
    31: 'wall-x.gif',
    32: 'x-paintwall.gif'
}

STATE_LOOKUP_TABLE = {
    10: 0,
    11: 0,
    12: 0,
    13: 2,
    14: 1,
    15: 1,
    16: 2,
    17: 2,
    18: 2,
    19: 2,
    20: 2,
    21: 2,
    22: 2,
    23: 2,
    24: 2,
    25: 2,
    26: 2,
    27: 2,
    28: 2,
    29: 2,
    30: 2,
    31: 2,
    32: 2,
    33: -1,
    40: 0
}
