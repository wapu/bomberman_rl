
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

    # Initialize environment and agents
    world = BombeRLeWorld([
        ('example_agent', True),
        ('example_agent', False),
        ('example_agent', False),
        ('user_agent', False)
    ])

    # Run one or more games
    for i in range(3):
        if not world.running:
            world.new_round()

        # First render
        if s.gui:
            world.render()
            pygame.display.flip()

        round_finished = False
        last_update = time()
        last_frame = time()
        user_inputs = []

        # Main game loop
        while not round_finished:
            # Grab events
            key_pressed = None
            for event in pygame.event.get():
                if event.type == QUIT:
                    world.end_round()
                    return
                elif event.type == pygame.locals.KEYDOWN:
                    key_pressed = event.key
                    if key_pressed in (K_q, K_ESCAPE):
                        world.end_round()
                        round_finished = True
                    # Convert keyboard input into actions
                    if s.input_map.get(key_pressed):
                        if s.turn_based:
                            user_inputs = [s.input_map.get(key_pressed),]
                        else:
                            user_inputs.append(s.input_map.get(key_pressed))

            # Game logic
            if ((s.turn_based and not key_pressed)
                    or (s.gui and (time()-last_update < s.update_interval))):
                pass
            else:
                if world.running:
                    last_update = time()
                    try:
                        world.do_step(user_inputs.pop(0) if len(user_inputs) else 'WAIT')
                    except Exception as e:
                        world.end_round()
                        raise

            if not world.running and not s.gui:
                round_finished = True

            # Rendering
            if s.gui and (time()-last_frame >= 1./s.fps):
                world.render()
                pygame.display.flip()

    world.end()

if __name__ == '__main__':
    main()
