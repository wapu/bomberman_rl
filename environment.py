
from time import time
import multiprocessing as mp
import numpy as np
import random
import pygame
from pygame.locals import *

import logging

from agents import *
from items import *
from settings import s


class BombeRLeWorld(object):

    def __init__(self):
        # Set up logging
        self.logger = logging.getLogger('BombeRLeWorld')
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler('logs/game.log', mode='w')
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s [%(name)s] %(levelname)s: %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.info('Initializing game world')

        # Arena with wall layout
        self.arena = (np.random.rand(s.cols, s.rows) > 0.25).astype(int)
        self.arena[:1, :] = -1
        self.arena[-1:,:] = -1
        self.arena[:, :1] = -1
        self.arena[:,-1:] = -1
        for x in range(s.cols):
            for y in range(s.rows):
                if (x+1)*(y+1) % 2 == 1:
                    self.arena[x,y] = -1

        # Available robot colors and starting positions
        self.colors = ['blue', 'green', 'yellow', 'pink']
        self.start_positions = [(1,1), (1,s.rows-2), (s.cols-2,1), (s.cols-2,s.rows-2)]
        # Clear some space around starting positions
        for (x,y) in self.start_positions:
            for (xx,yy) in [(x,y), (x-1,y), (x+1,y), (x,y-1), (x,y+1)]:
                if self.arena[xx,yy] == 1:
                    self.arena[xx,yy] = 0

        if s.gui:
            # Initialize screen
            self.screen = pygame.display.set_mode((s.width, s.height))
            pygame.display.set_caption('BombeRLe')

            # Background and tiles
            self.background = pygame.Surface((s.width, s.height))
            self.background = self.background.convert()
            self.background.fill((0,0,0))
            self.t_wall = pygame.image.load('assets/brick.png')
            self.t_crate = pygame.image.load('assets/crate.png')

            # Font for scores and such
            font_name = pygame.font.match_font('roboto')
            self.fonts = {
                'huge': pygame.font.Font(font_name, 32),
                'big': pygame.font.Font(font_name, 20),
                'medium': pygame.font.Font(font_name, 16),
                'small': pygame.font.Font(font_name, 12),
            }

        # Bookkeeping
        self.running = True
        self.step = 0
        self.agents = []
        self.active_agents = []
        self.coins = []
        self.bombs = []
        self.explosions = []


    def add_agent(self, filename, train=False):
        if len(self.start_positions) > 0:
            # Add unique suffix to name
            name = filename + '_' + str(list([a.process.filename for a in self.agents]).count(filename))

            # Set up a new process to run the agent's code
            pipe_to_world, pipe_to_agent = mp.Pipe()
            ready_flag = mp.Event()
            train_flag = mp.Event()
            if train:
                train_flag.set()
            p = AgentProcess(pipe_to_world, ready_flag, name, filename, train_flag)
            self.logger.info(f'Starting process for agent <{name}>')
            p.start()

            # Create the agent container and assign random starting slot
            agent = Agent(p, pipe_to_agent, ready_flag, self.colors.pop(), train_flag)
            random.shuffle(self.start_positions)
            agent.x, agent.y = self.start_positions.pop()
            self.agents.append(agent)
            self.active_agents.append(agent)

            # Make sure process setup is finished
            self.logger.debug(f'Waiting for setup of agent <{agent.name}>')
            agent.ready_flag.wait()
            agent.ready_flag.clear()
            self.logger.debug(f'Setup finished for agent <{agent.name}>')


    def get_state_for_agent(self, agent):
        state = {}
        state['step'] = self.step
        state['arena'] = np.array(self.arena)
        state['self'] = agent.get_state()
        state['others'] = [other.get_state() for other in self.active_agents if other is not agent]
        state['bombs'] = [bomb.get_state() for bomb in self.bombs]
        explosion_map = np.zeros(self.arena.shape)
        for e in self.explosions:
            for (x,y) in e.blast_coords:
                explosion_map[x,y] = max(explosion_map[x,y], e.timer)
        state['explosions'] = explosion_map
        return state


    def tile_is_free(self, x, y):
        is_free = (self.arena[x,y] == 0)
        if is_free:
            for bomb in self.bombs:
                is_free = is_free and (bomb.x != x or bomb.y != y)
        return is_free


    def update(self):
        self.step += 1
        self.logger.info(f'STARTING STEP {self.step}')

        # Send world state to all agents
        for a in self.active_agents:
            self.logger.debug(f'Sending game state to agent <{a.name}>')
            a.pipe.send(self.get_state_for_agent(a))

        # Send reward to all agents that expect it, then reset it and wait for them
        for a in self.active_agents:
            if a.train_flag.is_set():
                self.logger.debug(f'Sending reward {a.reward} to agent <{a.name}>')
                a.pipe.send(a.reward)
            a.reward = 0
        for a in self.active_agents:
            if a.train_flag.is_set():
                self.logger.debug(f'Waiting for agent <{a.name}> to process rewards')
                a.ready_flag.wait()
                self.logger.debug(f'Clearing flag for agent <{a.name}>')
                a.ready_flag.clear()

        # Give agents time to decide and set their ready flags; interrupt after time limit
        deadline = time() + s.timeout
        for a in self.active_agents:
            if not a.ready_flag.wait(deadline - time()):
                self.logger.warn(f'Interrupting agent <{a.name}>')
                if os.name == 'posix':
                    os.kill(a.process.pid, signal.SIGINT)
                else:
                    # Special case for Windows
                    os.kill(a.process.pid, signal.CTRL_C_EVENT)

        # Perform decided agent actions
        for a in self.active_agents:
            self.logger.debug(f'Collecting action from agent <{a.name}>')
            (action, t) = a.pipe.recv()
            self.logger.info(f'Agent <{a.name}> chose action {action} in {t:.2f}s.')
            a.times.append(t)
            a.mean_time = np.mean(a.times)
            if action == 'UP'    and self.tile_is_free(a.x, a.y - 1):
                a.y -= 1
            if action == 'DOWN'  and self.tile_is_free(a.x, a.y + 1):
                a.y += 1
            if action == 'LEFT'  and self.tile_is_free(a.x - 1, a.y):
                a.x -= 1
            if action == 'RIGHT' and self.tile_is_free(a.x + 1, a.y):
                a.x += 1
            if action == 'BOMB' and a.bombs_left > 0:
                self.logger.info(f'Agent <{a.name}> drops bomb at {(a.x, a.y)}')
                self.bombs.append(a.make_bomb())
                a.bombs_left -= 1

        # Reset agent flags
        for a in self.active_agents:
            self.logger.debug(f'Clearing flag for agent <{a.name}>')
            a.ready_flag.clear()

        # Coins
        for coin in self.coins:
            for a in self.active_agents:
                if a.x == coin.x and a.y == coin.y:
                    coin.picked_up = True
                    self.logger.info(f'Agent <{a.name}> picked up coin at {(a.x, a.y)} and receives 1 point')
                    a.update_score(s.reward_coin)
        self.coins = [c for c in self.coins if not c.picked_up]

        # Bombs
        for bomb in self.bombs:
            bomb.timer -= 1
            # Explode when timer is finished
            if bomb.timer < 0:
                self.logger.info(f'Agent <{bomb.owner.name}>\'s bomb at {(bomb.x, bomb.y)} explodes')
                blast_coords = bomb.get_blast_coords(self.arena)
                # Clear crates
                for (x,y) in blast_coords:
                    if self.arena[x,y] == 1:
                        self.arena[x,y] = 0
                        # Maybe spawn a coin
                        if np.random.rand() < s.coin_drop_rate:
                            self.logger.info(f'Coin dropped at {(x,y)}')
                            self.coins.append(Coin((x,y)))
                # Create explosion
                screen_coords = [(s.grid_offset[0] + s.grid_size*x, s.grid_offset[1] + s.grid_size*y) for (x,y) in blast_coords]
                self.explosions.append(Explosion(blast_coords, screen_coords, bomb.owner))
                bomb.active = False
                bomb.owner.bombs_left += 1
        self.bombs = [b for b in self.bombs if b.active]

        # Explosions
        agents_hit = []
        for explosion in self.explosions:
            explosion.timer -= 1
            if explosion.timer < 0:
                explosion.active = False
            # Kill agents
            if explosion.timer > 0:
                for a in self.active_agents:
                    if (not a.dead) and (a.x, a.y) in explosion.blast_coords:
                        agents_hit.append(a)
                        # Note who killed whom, adjust scores
                        if a is explosion.owner:
                            self.logger.info(f'Agent <{a.name}> blown up by own bomb')
                        else:
                            self.logger.info(f'Agent <{a.name}> blown up by agent <{explosion.owner.name}>\'s bomb')
                            self.logger.info(f'Agent <{explosion.owner.name}> receives 1 point')
                            explosion.owner.update_score(s.reward_kill)
        for a in agents_hit:
            a.dead = True
            self.active_agents.remove(a)
            # Send exit message to shut down agent
            a.pipe.send(None)
        self.explosions = [e for e in self.explosions if e.active]

        if len(self.active_agents) <= 1:
            self.logger.debug(f'Only {len(self.active_agents)} agent(s) left, wrap up game')
            self.wrap_up()


    def wrap_up(self):
        if self.running:
            self.logger.info('WRAPPING UP GAME')
            for a in self.active_agents:
                # Reward survivor(s)
                self.logger.info(f'Agent <{a.name}> receives 1 point for surviving')
                a.update_score(s.reward_last)
                # Send exit message to shut down agent
                self.logger.debug(f'Sending exit message to agent <{a.name}>')
                a.pipe.send(None)
            for a in self.agents:
                # Send final reward to agent if it expects one
                if a.train_flag.is_set():
                    self.logger.debug(f'Sending final reward {a.reward} to agent <{a.name}>')
                    a.pipe.send(a.reward)
            # Penalty for agent who spent most time thinking
            slowest = max(self.agents, key=lambda a: a.mean_time)
            self.logger.info(f'Agent <{slowest.name}> loses 1 point for being slowest (avg. {slowest.mean_time:.3f}s)')
            slowest.update_score(s.reward_slow)

            self.running = False


    def render_text(self, text, x, y, color, hcenter=False, vcenter=False, size='medium'):
        if not s.gui: return
        text_surface = self.fonts[size].render(text, True, color)
        text_rect = text_surface.get_rect()
        if     hcenter   and (not vcenter): text_rect.midtop  = (x,y)
        if     hcenter   and       vcenter: text_rect.mid     = (x,y)
        if (not hcenter) and       vcenter: text_rect.midleft = (x,y)
        if (not hcenter) and (not vcenter): text_rect.topleft = (x,y)
        self.screen.blit(text_surface, text_rect)


    def render(self):
        if not s.gui: return
        self.screen.blit(self.background, (0,0))

        # World
        self.logger.debug(f'RENDERING game world')
        for x in range(self.arena.shape[1]):
            for y in range(self.arena.shape[0]):
                if self.arena[x,y] == -1:
                    self.screen.blit(self.t_wall, (s.grid_offset[0] + s.grid_size*x, s.grid_offset[1] + s.grid_size*y))
                if self.arena[x,y] == 1:
                    self.screen.blit(self.t_crate, (s.grid_offset[0] + s.grid_size*x, s.grid_offset[1] + s.grid_size*y))

        # Items
        self.logger.debug(f'RENDERING items')
        for bomb in self.bombs:
            bomb.render(self.screen, s.grid_offset[0] + s.grid_size*bomb.x, s.grid_offset[1] + s.grid_size*bomb.y)
        for coin in self.coins:
            coin.render(self.screen, s.grid_offset[0] + s.grid_size*coin.x, s.grid_offset[1] + s.grid_size*coin.y)

        # Agents
        self.logger.debug(f'RENDERING agents')
        for agent in self.active_agents:
            agent.render(self.screen, s.grid_offset[0] + s.grid_size*agent.x, s.grid_offset[1] + s.grid_size*agent.y)

        # Explosions
        self.logger.debug(f'RENDERING explosions')
        for explosion in self.explosions:
            explosion.render(self.screen)

        # Scores
        self.agents.sort(key=lambda a: (a.score, -a.mean_time), reverse=True)
        y_base = s.grid_offset[1] + 15
        for i, a in enumerate(self.agents):
            a.render(self.screen, 600, y_base + 50*i - 15)
            self.render_text(a.name, 650, y_base + 50*i, (200,200,200), vcenter=True)
            self.render_text(f'{a.score: d}', 850, y_base + 50*i, (255,255,255), vcenter=True, size='big')
            self.render_text(f'({a.mean_time: .3f})', 900, y_base + 50*i, (100,100,100), vcenter=True, size='small')
