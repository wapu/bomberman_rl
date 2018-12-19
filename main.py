
import contextlib
with contextlib.redirect_stdout(None):
    import pygame
from pygame.locals import *
import numpy as np
import multiprocessing as mp

from environment import BombeRLeWorld


FPS = 3
FAST_MODE = False
RENDER = True
INTERACTIVE = False


def main():
    # Emulate Windows process spawning behaviour under Unix (for testing)
    # mp.set_start_method('spawn')

    # Initialize screen
    pygame.init()
    screen = pygame.display.set_mode((600, 600))
    pygame.display.set_caption('BombeRLe')

    # Initialize environment
    world = BombeRLeWorld(screen)

    # Initialize agents
    world.add_agent('example_agent', train=True)
    world.add_agent('example_agent')
    world.add_agent('example_agent')
    world.add_agent('example_agent')

    # Initial render
    if RENDER:
        world.render()
        pygame.display.flip()

    # Event loop
    clock = pygame.time.Clock()
    quit = False
    while not quit:
        # grab inputs
        pressed_key = None
        for event in pygame.event.get():
            if event.type == QUIT:
                world.wrap_up()
                quit = True
            elif event.type == pygame.locals.KEYDOWN:
                pressed_key = event.key
                if pressed_key in (K_q, K_ESCAPE):
                    world.wrap_up()
                    quit = True
        if INTERACTIVE:
            useragent.input = pressed_key

        if (not INTERACTIVE) or pressed_key:
            # game logic
            if world.state == 'RUNNING':
                try:
                    world.update()
                except Exception as e:
                    print(f'Wrapping up game...')
                    world.wrap_up()
                    raise

            # render screen
            if RENDER:
                world.render()
                pygame.display.flip()

                if not (FAST_MODE or INTERACTIVE):
                    clock.tick(FPS)

if __name__ == '__main__':
    main()
