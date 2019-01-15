
import numpy as np
from random import shuffle
from time import time, sleep
from collections import deque

from settings import s


def look_for_targets(free_space, start, targets):
    frontier = [start]
    parent_dict = {start: start}
    best = start
    best_dist = np.sum(np.abs(np.subtract(targets, start)), axis=1).min()

    while len(frontier) > 0:
        current = frontier.pop(0)

        d = np.sum(np.abs(np.subtract(targets, current)), axis=1).min()
        if d <= best_dist:
            best = current
            best_dist = d
        if d == 0: break

        x, y = current
        neighbors = [(x,y) for (x,y) in [(x+1,y), (x-1,y), (x,y+1), (x,y-1)] if free_space[x,y]]
        for n in neighbors:
            if n not in parent_dict:
                frontier.append(n)
                parent_dict[n] = current

    current = best
    while True:
        if parent_dict[current] == start: return current
        current = parent_dict[current]


def setup(agent):
    agent.logger.debug('Successfully entered setup code')
    np.random.seed()
    agent.history = deque(['WAIT',])


from numpy.linalg import inv
def act(agent):
    agent.logger.info('Picking action according to rule set')

    # if np.random.rand() > 0.9:
    #     # waste time
    #     t = time()
    #     while time() - t < s.timeout:
    #         foo = inv(np.random.rand(1000,1000))

    # Get info
    arena = agent.game_state['arena']
    x, y, _, bombs_left = agent.game_state['self']
    bombs = agent.game_state['bombs']
    bomb_xys = [(x,y) for (x,y,t) in bombs]
    others = [(x,y) for (x,y,n,b) in agent.game_state['others']]
    coins = agent.game_state['coins']
    bomb_map = np.ones(arena.shape) * 5
    for xb,yb,t in bombs:
        for (i,j) in [(xb+h, yb) for h in range(-3,4)] + [(xb, yb+h) for h in range(-3,4)]:
            if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
                bomb_map[i,j] = min(bomb_map[i,j], t)

    # Check which moves make sense
    directions = [(x,y), (x+1,y), (x-1,y), (x,y+1), (x,y-1)]
    valid, valid_actions = [], []
    for d in directions:
        if ((arena[d] == 0) and
            (agent.game_state['explosions'][d] <= 1) and
            (bomb_map[d] > 0) and
            (not d in others) and
            (not d in bomb_xys)):
            valid.append(d)
    if (x-1,y) in valid: valid_actions.append('LEFT')
    if (x+1,y) in valid: valid_actions.append('RIGHT')
    if (x,y-1) in valid: valid_actions.append('UP')
    if (x,y+1) in valid: valid_actions.append('DOWN')
    if bombs_left > 0: valid_actions.append('BOMB')
    agent.logger.debug(f'Valid actions: {valid_actions}')

    # Check for cycles in previous behaviour to avoid repetition in the late game
    # NEEDS TO BE BETTER
    last_action = agent.history[0]
    if len(agent.history) > 40:
        try:
            prev_occurrence = agent.history.index(last_action, 1)
            if prev_occurrence < len(agent.history) / 2:
                length = max(prev_occurrence, 20)
                cycle = True
                for i in range(1, length):
                    if agent.history[i] != agent.history[prev_occurrence + i]:
                        cycle = False
                        break
                if cycle:
                    # Disallow the action that followed after the last time
                    a = agent.history[prev_occurrence - 1]
                    agent.logger.debug(f'Encountered cycle of length {length}, avoid action {a}')
                    if a in valid_actions:
                        valid_actions.remove(a)
        except ValueError:
            # Action never occurred before, no problem
            pass

    # Collect actions in a queue
    action_ideas = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    shuffle(action_ideas)

    # Head towards nearest coin or opponent
    d = look_for_targets(arena == 0, (x,y), others+coins)
    if d == (x,y-1): action_ideas.append('UP')
    if d == (x,y+1): action_ideas.append('DOWN')
    if d == (x-1,y): action_ideas.append('LEFT')
    if d == (x+1,y): action_ideas.append('RIGHT')
    if d == (x,y): action_ideas.append('BOMB')

    # Place bomb if at dead end or next to opponent
    if len(valid) == 2 or (min(abs(xy[0] - x) + abs(xy[1] - y) for xy in others)) <= 1:
        action_ideas.append('BOMB')

    # Run away from any dangerous bomb
    for xb,yb,t in bombs:
        if (xb == x) and (abs(yb-y) < 4):
            # Run away
            if (yb > y): action_ideas.append('UP')
            if (yb < y): action_ideas.append('DOWN')
            # if possible turn a corner
            action_ideas.append('LEFT')
            action_ideas.append('RIGHT')
        if (yb == y) and (abs(xb-x) < 4):
            # Run away
            if (xb > x): action_ideas.append('LEFT')
            if (xb < x): action_ideas.append('RIGHT')
            # if possible turn a corner
            action_ideas.append('UP')
            action_ideas.append('DOWN')

    # Pick first action from queue that is valid
    while len(action_ideas) > 0:
        a = action_ideas.pop()
        if a in valid_actions:
            agent.next_action = a
            break

    agent.history.appendleft(agent.next_action)


def reward_update(agent):
    agent.logger.debug(f'Found reward of {agent.reward}')


def end_of_episode(agent):
    agent.logger.debug(f'Found final reward of {agent.reward}')
