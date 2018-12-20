
from collections import namedtuple

settings = {
    # Display
    'width': 1000,
    'height': 600,
    'gui': True,
    'fps': 25,

    # Main loop
    'update_interval': 0.5,
    'wait_for_keyboard': False,

    # Game properties
    'cols': 17,
    'rows': 17,
    'grid_size': 30,
    'actions': ['UP', 'DOWN', 'LEFT', 'RIGHT', 'BOMB', 'WAIT'],
    'coin_drop_rate': 0.1,

    # Rules for agents
    'timeout': 2.0,
    'reward_kill': 2,
    'reward_coin': 1,
    'reward_last': 3,
    'reward_slow': -1,
}
settings['grid_offset'] = [(settings['height'] - settings['rows']*settings['grid_size'])//2] * 2
s = namedtuple("Settings", settings.keys())(*settings.values())
