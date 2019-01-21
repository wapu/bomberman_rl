
import numpy as np
from random import shuffle
from time import time, sleep
from collections import deque

from settings import s


def look_for_targets(free_space, start, targets):
    """Find direction of closest target that can be reached via free tiles.

    Performs a breadth-first search of the reachable free tiles until a target is encountered.
    If no target is found, the tile that is closest to any target is chosen.

    Args:
        free_space: Boolean numpy array. True for free tiles and False for obstacles.
        start: the coordinate from which to begin the search.
        targets: list or array holding the coordinates of all target tiles.
    Returns:
        coordinate of first step towards closest target or towards tile closest to any target.
    """
    frontier = [start]
    parent_dict = {start: start}
    best = start
    best_dist = np.sum(np.abs(np.subtract(targets, start)), axis=1).min()

    while len(frontier) > 0:
        current = frontier.pop(0)
        # Find distance from current position to all targets, track closest
        d = np.sum(np.abs(np.subtract(targets, current)), axis=1).min()
        if d <= best_dist:
            best = current
            best_dist = d
        if d == 0: break
        # Add unexplored free neighboring tiles to the queue
        x, y = current
        neighbors = [(x,y) for (x,y) in [(x+1,y), (x-1,y), (x,y+1), (x,y-1)] if free_space[x,y]]
        for neighbor in neighbors:
            if neighbor not in parent_dict:
                frontier.append(neighbor)
                parent_dict[neighbor] = current
    # Determine the first step towards the best found target tile
    current = best
    while True:
        if parent_dict[current] == start: return current
        current = parent_dict[current]


def setup(agent):
    """Called once before a set of games to initialize data structures etc.

    The 'agent' object passed to this method will be the same in all other
    callback methods. You can assign new properties (like agent.history below)
    here or later on and they will be persistent even across multiple games.
    You can also use the agent.logger object at any time to write to the log
    file for debugging (see https://docs.python.org/3.7/library/logging.html).
    """
    agent.logger.debug('Successfully entered setup code')
    np.random.seed()
    agent.history = deque(['WAIT',])


def act(agent):
    """Called each game step to determine the agent's next action.

    You can find out about the state of the game environment via agent.game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    Set the action you wish to perform by assigning the relevant string to
    agent.next_action. You can assign to this variable multiple times during
    your computations. If this method takes longer than the time limit specified
    in settings.py, execution is interrupted by the game and the current value
    of agent.next_action will be used. The default value is 'WAIT'.
    """
    agent.logger.info('Picking action according to rule set')

    # Gather information about the game state
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

    # Check which moves make sense at all
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

    # Check for cycles in previous behaviour, remove repeating actions from
    # valid actions list (this is not ideal yet)
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

    # Collect basic action proposals in a queue
    # Later on, the last added action that is also valid will be chosen
    action_ideas = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    shuffle(action_ideas)

    # Add action proposal to head towards nearest coin or opponent
    d = look_for_targets(arena == 0, (x,y), others+coins)
    if d == (x,y-1): action_ideas.append('UP')
    if d == (x,y+1): action_ideas.append('DOWN')
    if d == (x-1,y): action_ideas.append('LEFT')
    if d == (x+1,y): action_ideas.append('RIGHT')
    if d == (x,y): action_ideas.append('BOMB')

    # Add proposal to drop a bomb if at dead end or next to opponent
    if len(valid) == 2 or (min(abs(xy[0] - x) + abs(xy[1] - y) for xy in others)) <= 1:
        action_ideas.append('BOMB')

    # Add proposal to run away from any nearby bomb about to blow
    for xb,yb,t in bombs:
        if (xb == x) and (abs(yb-y) < 4):
            # Run away
            if (yb > y): action_ideas.append('UP')
            if (yb < y): action_ideas.append('DOWN')
            # If possible, turn a corner
            action_ideas.append('LEFT')
            action_ideas.append('RIGHT')
        if (yb == y) and (abs(xb-x) < 4):
            # Run away
            if (xb > x): action_ideas.append('LEFT')
            if (xb < x): action_ideas.append('RIGHT')
            # If possible, turn a corner
            action_ideas.append('UP')
            action_ideas.append('DOWN')

    # Pick last action added to the proposals list that is also valid
    while len(action_ideas) > 0:
        a = action_ideas.pop()
        if a in valid_actions:
            agent.next_action = a
            break

    # Keep track of chosen action for cycle detection
    agent.history.appendleft(agent.next_action)


def reward_update(agent):
    """Called once per step to allow intermediate rewards based on game events.

    When this method is called, agent.events will contain a list of all game
    events relevant to your agent that occured during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state. In
    contrast to act, this method has no time limit.
    """
    agent.logger.debug(f'Encountered {len(agent.events)} game events.')


def end_of_episode(agent):
    """Called at the end of each game to hand out final rewards and do training.

    This is similar to reward_update, except it is only called at the end of a
    game. agent.events will contain all events that occured during your agent's
    final step. You should place your actual learning code in this method.
    """
    agent.logger.debug(f'Encountered {len(agent.events)} game events in final step.')
