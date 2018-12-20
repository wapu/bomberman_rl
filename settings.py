
from collections import namedtuple
import pygame
from pygame.locals import *


settings = {
    # Display
    'width': 1000,
    'height': 600,
    'gui': True,
    'fps': 25,

    # Main loop
    'update_interval': 0.5,
    'turn_based': False,

    # Game properties
    'cols': 17,
    'rows': 17,
    'grid_size': 30,
    'actions': ['UP', 'DOWN', 'LEFT', 'RIGHT', 'BOMB', 'WAIT'],
    'max_agents': 4,
    'coin_drop_rate': 0.1,

    # Rules for agents
    'timeout': 2.0,
    'reward_kill': 2,
    'reward_coin': 1,
    'reward_last': 3,
    'reward_slow': -1,

    # User input
    'input_map': {
        K_UP: 'UP',
        K_DOWN: 'DOWN',
        K_LEFT: 'LEFT',
        K_RIGHT: 'RIGHT',
        K_RETURN: 'WAIT',
        K_SPACE: 'BOMB',
    },

}
settings['grid_offset'] = [(settings['height'] - settings['rows']*settings['grid_size'])//2] * 2
s = namedtuple("Settings", settings.keys())(*settings.values())
