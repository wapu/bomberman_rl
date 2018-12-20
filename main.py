
from time import time
import contextlib
with contextlib.redirect_stdout(None):
    import pygame
from pygame.locals import *
import numpy as np
import multiprocessing as mp

from environment import BombeRLeWorld
from settings import s


def main():
    pygame.init()

    # Emulate Windows process spawning behaviour under Unix (for testing)
    # mp.set_start_method('spawn')

    # Initialize environment
    world = BombeRLeWorld()

    # Initialize agents
    world.add_agent('example_agent', train=True)
    world.add_agent('example_agent')
    world.add_agent('example_agent')
    world.add_agent('example_agent')

    # Initial render
    if s.gui:
        world.render()
        pygame.display.flip()

    quit = False
    last_update = time()
    last_frame = time()

    # Main game loop
    while not quit:
        # Grab events
        key_pressed = None
        for event in pygame.event.get():
            if event.type == QUIT:
                world.wrap_up()
                quit = True
            elif event.type == pygame.locals.KEYDOWN:
                key_pressed = event.key
                if key_pressed in (K_q, K_ESCAPE):
                    world.wrap_up()
                    quit = True

        # Game logic
        if ((s.wait_for_keyboard and not key_pressed)
                or (s.gui and (time()-last_update < s.update_interval))):
            pass
        else:
            if world.running:
                last_update = time()
                try:
                    world.update()
                except Exception as e:
                    world.wrap_up()
                    raise

        if not world.running and not s.gui:
            quit = True

        # Rendering
        if s.gui and (time()-last_frame >= 1./s.fps):
            world.render()
            pygame.display.flip()


if __name__ == '__main__':
    main()
