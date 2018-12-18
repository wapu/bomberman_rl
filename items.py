
import pygame
from pygame.locals import *


class Item(object):

    def __init__(self):
        self.avatar = pygame.Surface((10,10))
        self.avatar.fill((180,180,180))
        self.offset = (10,10)

        self.x, self.y = 1, 1

    def render(self, screen, x, y):
        screen.blit(self.avatar, (x + self.offset[0], y + self.offset[1]))


class Bomb(Item):

    def __init__(self, pos, owner, timer, power):
        super(Bomb, self).__init__()
        self.x = pos[0]
        self.y = pos[1]
        self.owner = owner
        self.timer = timer
        self.power = power
        self.active = True

    def get_state(self):
        return ((self.x, self.y), self.timer, self.power, self.active, self.owner.name)

    def get_blast_coords(self, arena):
        x, y = self.x, self.y
        blast_coords = [(x,y)]

        for i in range(1, self.power+1):
            if arena[x+i,y] == -1: break
            blast_coords.append((x+i,y))
        for i in range(1, self.power+1):
            if arena[x-i,y] == -1: break
            blast_coords.append((x-i,y))
        for i in range(1, self.power+1):
            if arena[x,y+i] == -1: break
            blast_coords.append((x,y+i))
        for i in range(1, self.power+1):
            if arena[x,y-i] == -1: break
            blast_coords.append((x,y-i))

        return blast_coords


class Explosion(Item):

    def __init__(self, blast_coords, screen_coords, owner):
        self.blast_coords = blast_coords
        self.screen_coords = screen_coords
        self.owner = owner
        self.timer = owner.explosion_timer
        self.active = True

        self.stages = [pygame.Surface((4*i,4*i)) for i in range(5)]
        for s in self.stages: s.fill((255,128,0))
        self.offsets = [(15-2*i,15-2*i) for i in range(5)]

    def render(self, screen):
        for (x,y) in self.screen_coords:
            screen.blit(self.stages[self.timer+2], (x + self.offsets[self.timer+2][0], y + self.offsets[self.timer+2][1]))
