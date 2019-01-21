
from time import time, sleep
import os, signal
import multiprocessing as mp
import importlib
import pygame
import logging
from pygame.locals import *

from items import *
from settings import s, e


class IgnoreKeyboardInterrupt(object):
    """Context manager that protects enclosed code from Interrupt signals."""
    def __enter__(self):
        self.old_handler = signal.signal(signal.SIGINT, self.handler)
    def handler(self, sig, frame):
        pass
    def __exit__(self, type, value, traceback):
        signal.signal(signal.SIGINT, self.old_handler)


class AgentProcess(mp.Process):
    """Wrapper class that runs custom agent code in a separate process."""

    def __init__(self, pipe_to_world, ready_flag, name, agent_dir, train_flag):
        super(AgentProcess, self).__init__(name=name)
        self.pipe_to_world = pipe_to_world
        self.ready_flag = ready_flag
        self.agent_dir = agent_dir
        self.train_flag = train_flag

    def run(self):
        # Set up individual loggers for the wrapper and the custom code
        self.wlogger = logging.getLogger(self.name + '_wrapper')
        self.wlogger.setLevel(s.log_agent_wrapper)
        self.logger = logging.getLogger(self.name + '_code')
        self.logger.setLevel(s.log_agent_code)
        log_dir = f'agent_code/{self.agent_dir}/logs/'
        if not os.path.exists(log_dir): os.makedirs(log_dir)
        handler = logging.FileHandler(f'{log_dir}{self.name}.log', mode='w')
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s [%(name)s] %(levelname)s: %(message)s')
        handler.setFormatter(formatter)
        self.wlogger.addHandler(handler)
        self.logger.addHandler(handler)

        # Import custom code for the agent from provided script
        self.wlogger.info(f'Import agent code from "agent_code/{self.agent_dir}/callbacks.py"')
        self.code = importlib.import_module('agent_code.' + self.agent_dir + '.callbacks')

        # Initialize custom code
        self.wlogger.info('Initialize agent code')
        try:
            self.code.setup(self)
        except Exception as e:
            self.wlogger.exception(f'Error in callback function: {e}')
        self.wlogger.debug('Set flag to indicate readiness')
        self.ready_flag.set()

        # Play one game after the other until global exit message is received
        while True:
            # Receive round number and check for exit message
            self.wlogger.debug('Wait for new round')
            self.round = self.pipe_to_world.recv()
            if self.round is None:
                self.wlogger.info('Received global exit message')
                break
            self.wlogger.info(f'STARTING ROUND #{self.round}')

            # Take steps until exit message for current round is received
            while True:
                # Receive new game state and check for exit message
                self.wlogger.debug('Receive game state')
                self.game_state = self.pipe_to_world.recv()
                if self.game_state['died']:
                    self.ready_flag.set()
                    self.wlogger.info('Received exit message for round')
                    break
                self.wlogger.info(f'STARTING STEP {self.game_state["step"]}')

                # Process game events for rewards if in training mode
                if self.train_flag.is_set():
                    self.wlogger.debug('Receive event queue')
                    self.events = self.pipe_to_world.recv()
                    self.wlogger.debug(f'Received event queue {self.events}')
                    self.wlogger.info('Process intermediate rewards')
                    try:
                        self.code.reward_update(self)
                    except Exception as e:
                        self.wlogger.exception(f'Error in callback function: {e}')
                    self.wlogger.debug('Set flag to indicate readiness')
                    self.ready_flag.set()

                # Come up with an action to perform
                self.wlogger.debug('Begin choosing an action')
                self.next_action = 'WAIT'
                t = time()
                try:
                    self.code.act(self)
                except KeyboardInterrupt:
                    self.wlogger.warn(f'Got interrupted by timeout')
                except Exception as e:
                    self.wlogger.exception(f'Error in callback function: {e}')

                # Send action and time taken back to main process
                with IgnoreKeyboardInterrupt():
                    t = time() - t
                    self.wlogger.info(f'Chose action {self.next_action} after {t:.3f}s of thinking')
                    self.wlogger.debug('Send action and time to main process')
                    self.pipe_to_world.send((self.next_action, t))
                    while self.ready_flag.is_set():
                        sleep(0.01)
                    self.wlogger.debug('Set flag to indicate readiness')
                    self.ready_flag.set()

            # Process final events and learn from episode if in training mode
            if self.train_flag.is_set():
                self.wlogger.info('Finalize agent\'s training')
                self.wlogger.debug('Receive final event queue')
                self.events = self.pipe_to_world.recv()
                self.wlogger.debug(f'Received final event queue {self.events}')
                try:
                    self.code.end_of_episode(self)
                except Exception as e:
                    self.wlogger.exception(f'Error in callback function: {e}')
                self.ready_flag.set()

            self.wlogger.info(f'Round #{self.round} finished')

        self.wlogger.info('SHUT DOWN')


class Agent(object):
    """Class representing agents as game objects."""

    def __init__(self, process, pipe_to_agent, ready_flag, color, train_flag):
        """Set up agent, process for custom code and inter-process communication."""
        self.name = process.name
        self.process = process
        self.pipe = pipe_to_agent
        self.ready_flag = ready_flag
        self.color = color
        self.train_flag = train_flag
        self.events = []

        # Load custom avatar or standard robot avatar of assigned color
        try:
            self.avatar = pygame.image.load(f'agent_code/{self.process.agent_dir}/avatar.png')
            assert self.avatar.get_size() == (30,30)
        except Exception as e:
            self.avatar = pygame.image.load(f'assets/robot_{self.color}.png')

        # Prepare overlay that will indicate dead agent on the scoreboard
        self.shade = pygame.Surface((30,30), SRCALPHA)
        self.shade.fill((0,0,0,208))

        self.x, self.y = 1, 1
        self.total_score = 0

        self.bomb_timer = 5
        self.explosion_timer = 3
        self.bomb_power = 3
        self.bomb_type = Bomb

        self.reset()

    def reset(self):
        """Make agent ready for a new game round."""
        self.times = []
        self.mean_time = 0
        self.dead = False
        self.score = 0
        self.events = []
        self.bombs_left = 1

    def get_state(self):
        """Provide information about this agent for the global game state."""
        return (self.x, self.y, self.name, self.bombs_left)

    def update_score(self, delta):
        """Add delta to both the current round's score and the total score."""
        self.score += delta
        self.total_score += delta

    def make_bomb(self):
        """Create a new Bomb object at current agent position."""
        return self.bomb_type((self.x, self.y), self, self.bomb_timer, self.bomb_power, self.color)

    def render(self, screen, x, y):
        """Draw the agent's avatar to the screen at the given coordinates."""
        screen.blit(self.avatar, (x, y))
        if self.dead:
            screen.blit(self.shade, (x, y))
