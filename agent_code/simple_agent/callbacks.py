
import numpy as np
from random import shuffle
from queue import PriorityQueue
from time import sleep


def a_star_search(free_space, start, goal):
    frontier = PriorityQueue()
    frontier.put(start, 0)
    parent_dict = {start: None}
    cost_dict = {start: 0}

    best = start
    best_dist = abs(start[0] - goal[0]) + abs(start[1] - goal[1])

    while not frontier.empty():
        current = frontier.get()
        if current == goal: break

        x, y = current
        neighbors = [(x,y) for (x,y) in [(x+1,y), (x-1,y), (x,y+1), (x,y-1)] if free_space[x,y]]

        for n in neighbors:
            cost_n = cost_dict[current] + 1
            if n not in cost_dict or cost_n < cost_dict[n]:
                cost_dict[n] = cost_n
                heuristic = abs(n[0] - goal[0]) + abs(n[1] - goal[1])
                if heuristic < best_dist:
                    best = n
                    best_dist = heuristic
                frontier.put(n, cost_n + heuristic)
                parent_dict[n] = current

    p = best
    if p == start: return start
    while True:
        if parent_dict[p] == start: return p
        p = parent_dict[p]


def setup(agent):
    agent.logger.debug('Successfully entered setup code')
    np.random.seed()

def act(agent):
    agent.logger.info('Picking action according to rule set')

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

    # Collect actions in a queue
    action_ideas = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    shuffle(action_ideas)

    # Head towards nearest coin or opponent
    xn, yn = min(others+coins, key=lambda xy: abs(xy[0] - x) + abs(xy[1] - y))
    d = a_star_search(arena == 0, (x,y), (xn,yn))
    if d == (x,y-1): action_ideas.append('UP')
    if d == (x,y+1): action_ideas.append('DOWN')
    if d == (x-1,y): action_ideas.append('LEFT')
    if d == (x+1,y): action_ideas.append('RIGHT')
    # if (yn < y): action_ideas.append('UP')
    # if (yn > y): action_ideas.append('DOWN')
    # if (xn < x): action_ideas.append('LEFT')
    # if (xn > x): action_ideas.append('RIGHT')
    # if abs(yn-y) > abs(xn-x): action_ideas.append(action_ideas[-2])

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
            return

def reward_update(agent):
    agent.logger.debug(f'Found reward of {agent.reward}')

def end_of_episode(agent):
    agent.logger.debug(f'Found final reward of {agent.reward}')
