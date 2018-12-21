
from time import time
import os, signal
import multiprocessing as mp
import importlib
import pygame
import logging
from pygame.locals import *

from items import *


class AgentProcess(mp.Process):

    def __init__(self, pipe_to_world, ready_flag, name, agent_dir, train_flag):
        super(AgentProcess, self).__init__(name=name)
        self.pipe_to_world = pipe_to_world
        self.ready_flag = ready_flag
        self.agent_dir = agent_dir
        self.train_flag = train_flag

    def run(self):
        # Set up logging
        self.wlogger = logging.getLogger(self.name + '_wrapper')
        self.wlogger.setLevel(logging.INFO)
        self.logger = logging.getLogger(self.name + '_code')
        self.logger.setLevel(logging.DEBUG)
        log_dir = f'agent_code/{self.agent_dir}/logs/'
        if not os.path.exists(log_dir): os.makedirs(log_dir)
        handler = logging.FileHandler(f'{log_dir}{self.name}.log', mode='w')
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s [%(name)s] %(levelname)s: %(message)s')
        handler.setFormatter(formatter)
        self.wlogger.addHandler(handler)
        self.logger.addHandler(handler)

        # Import custom code for the agent
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

        # Repeat until global exit message is received
        while True:
            # Receive round number and check for exit message
            self.wlogger.debug('Wait for new round')
            self.round = self.pipe_to_world.recv()
            if self.round is None:
                self.wlogger.info('Received global exit message')
                break
            self.wlogger.info(f'STARTING ROUND #{self.round}')

            # Repeat until exit message for this round is received
            while True:
                # Receive new world state and check for exit message
                self.wlogger.debug('Receive game state')
                self.game_state = self.pipe_to_world.recv()
                if self.game_state is None:
                    self.wlogger.info('Received exit message for round')
                    break
                self.wlogger.info(f'STARTING STEP {self.game_state["step"]}')

                # Process intermediate rewards if in training mode
                if self.train_flag.is_set():
                    self.wlogger.debug('Receive global reward')
                    self.reward = self.pipe_to_world.recv()
                    self.wlogger.debug(f'Received global reward {self.reward}')
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
                finally:
                    # Send action and time taken back to main process
                    t = time() - t
                    self.wlogger.info(f'Chose action {self.next_action} after {t:.3f}s of thinking')
                    self.wlogger.debug('Send action and time to main process')
                    self.pipe_to_world.send((self.next_action, t))
                    self.wlogger.debug('Set flag to indicate readiness')
                    self.ready_flag.set()

            # Learn from episode if in training mode
            if self.train_flag.is_set():
                self.wlogger.info('Finalize agent\'s training')
                self.wlogger.debug('Receive final reward')
                self.reward = self.pipe_to_world.recv()
                self.wlogger.debug(f'Received final reward {self.reward}')
                try:
                    self.code.end_of_episode(self)
                except Exception as e:
                    self.wlogger.exception(f'Error in callback function: {e}')

            self.wlogger.info(f'Round #{self.round} finished')

        self.wlogger.info('SHUT DOWN')


class Agent(object):

    def __init__(self, process, pipe_to_agent, ready_flag, color, train_flag):
        self.name = process.name
        self.process = process
        self.pipe = pipe_to_agent
        self.ready_flag = ready_flag
        self.color = color
        self.train_flag = train_flag

        # Load custom avatar or robot avatar of assigned color
        try:
            self.avatar = pygame.image.load(f'agent_code/{self.process.agent_dir}/avatar.png')
            assert self.avatar.get_size() == (30,30)
        except Exception as e:
            self.avatar = pygame.image.load(f'assets/robot_{self.color}.png')

        # Prepare overlay to indicate dead agent on the scoreboard
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
        self.times = []
        self.mean_time = 0
        self.dead = False
        self.score = 0
        self.reward = 0
        self.bombs_left = 1

    def get_state(self):
        # return ((self.x, self.y), self.bomb_timer, self.explosion_timer, self.bomb_power, self.bombs_left, self.name)
        return (self.x, self.y, self.name)

    def update_score(self, delta):
        self.score += delta
        self.total_score += delta
        self.reward += delta

    def make_bomb(self):
        return self.bomb_type((self.x, self.y), self, self.bomb_timer, self.bomb_power, self.color)

    def render(self, screen, x, y):
        screen.blit(self.avatar, (x, y))
        if self.dead:
            screen.blit(self.shade, (x, y))
