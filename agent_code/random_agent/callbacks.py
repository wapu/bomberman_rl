
import numpy as np
from time import sleep


def setup(agent):
    agent.logger.debug('Successfully entered setup code')
    np.random.seed()

def act(agent):
    agent.logger.info('Pick action at random')
    # if np.random.rand() > 0.9:
    #     sleep(2.1)
    sleep(0.1 * np.random.rand())
    agent.next_action = np.random.choice(['RIGHT', 'LEFT', 'UP', 'DOWN', 'BOMB'])

def reward_update(agent):
    agent.logger.debug(f'Found reward of {agent.reward}')

def end_of_episode(agent):
    agent.logger.debug(f'Found final reward of {agent.reward}')
