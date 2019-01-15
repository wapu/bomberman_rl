
import os.path

import numpy as np
import torch
from settings import s, e
from torch.nn import functional
from torch.optim import Adam

from agent_code.rl_agent.model import DQN
from agent_code.rl_agent.schedule import LinearSchedule

from agent_code.simple_agent import callbacks


def setup(agent):
    agent.logger.debug('Successfully entered setup code')
    # hyperparameters
    np.random.seed()
    agent.test = s.gui
    agent.exploration = LinearSchedule(1000000, 0.1) if not agent.test else LinearSchedule(100, 0)
    agent.replay_buffer_size = 300000
    agent.batch_size = 32
    agent.num_param_updates = 0
    agent.target_update_freq = 2500
    agent.gamma = .95
    agent.learning_start_step = 50000
    agent.learning_interval = 8
    agent.save_interval = 100000
    import_file = './models/xxx.pth'
    lr = 0.0001

    agent.last_state = None
    agent.last_action = None

    callbacks.setup(agent)

    if os.path.isfile(import_file):
        data = torch.load(import_file)
        agent.model = DQN()
        agent.target_model = DQN()
        agent.optimizer = Adam(agent.model.parameters(), lr=lr)
        agent.model.load_state_dict(data['model'])
        agent.target_model.load_state_dict(data['model'])
        agent.optimizer.load_state_dict(data['optimizer'])
        agent.action_buffer = data['action_buffer']
        agent.state_buffer = data['state_buffer']
        agent.next_state_buffer = data['next_state_buffer']
        agent.reward_buffer = data['reward_buffer']
        agent.step = data['step']
        agent.logger.info(f'Loaded model at step {agent.step}')
    else:
        agent.model = DQN()
        agent.target_model = DQN()
        agent.optimizer = Adam(agent.model.parameters(), lr=lr)
        agent.action_buffer = torch.zeros((agent.replay_buffer_size, 1)).long()
        agent.state_buffer = torch.zeros((agent.replay_buffer_size, 6, 17, 17))
        agent.reward_buffer = torch.zeros((agent.replay_buffer_size, 1))
        agent.next_state_buffer = torch.zeros((agent.replay_buffer_size, 6, 17, 17))
        agent.step = 0


def act(agent):
    try:
        state = torch.zeros((1, 6, 17, 17))
        state[0, 0] = torch.from_numpy(agent.game_state['arena'])
        state[0, 1, agent.game_state['self'][0], agent.game_state['self'][1]] = 1
        for other in agent.game_state['others']:
            state[0, 2, other[0], other[1]] = 1
        for bomb in agent.game_state['bombs']:
            state[0, 3, bomb[0], bomb[1]] = bomb[2]
        state[0, 4] = torch.from_numpy(agent.game_state['explosions'])
        for coin in agent.game_state['coins']:
            state[0, 5, coin[0], coin[1]] = 1

        if agent.step < 150000:
            callbacks.act(agent)
            action = torch.zeros(1)
            action[0] = ['RIGHT', 'LEFT', 'UP', 'DOWN', 'BOMB', 'WAIT'].index(agent.next_action)
        else:
            action = select_action(state, agent)
            agent.next_action = ['RIGHT', 'LEFT', 'UP', 'DOWN', 'BOMB', 'WAIT'][action[0]]

        reward = compute_reward(agent)

        if agent.last_action is not None:
            agent.action_buffer[agent.step % agent.replay_buffer_size] = torch.LongTensor([[agent.last_action]])
            agent.state_buffer[agent.step % agent.replay_buffer_size] = agent.last_state
            agent.reward_buffer[agent.step % agent.replay_buffer_size] = torch.tensor([reward]).float()
            agent.next_state_buffer[agent.step % agent.replay_buffer_size] = state

        agent.last_state = state
        agent.last_action = action

        if agent.step % agent.learning_interval == 0 and agent.step > agent.learning_start_step and not agent.test:
            agent.logger.info('Doing optimizer step')
            idxs = []
            for i in range(agent.batch_size):
                idxs.append(np.random.randint(0, agent.replay_buffer_size if agent.step > agent.replay_buffer_size else agent.step))

            state_batch = agent.state_buffer[idxs]
            action_batch = agent.action_buffer[idxs]
            reward_batch = agent.reward_buffer[idxs]

            next_state_batch = agent.next_state_buffer[idxs]
            non_final_mask = torch.LongTensor([i for i in range(agent.batch_size) if (next_state_batch[i] == 0).sum().item() != 6 * 17 * 17])
            non_final_next_states = next_state_batch[non_final_mask]

            current_state_q_values = agent.model(state_batch)
            current_state_q_values = current_state_q_values.gather(1, action_batch)

            next_state_q_values = torch.zeros((agent.batch_size, 6)).type(torch.FloatTensor)
            # non_valid_action_mask = torch.FloatTensor(len(non_final_next_states), 6)
            # for i in range(len(non_final_next_states)):
            #     for j in range(6):
            #         if is_valid(non_final_next_states[i].unsqueeze_(0), j):
            #             non_valid_action_mask[i, j] = 0
            #         else:
            #             non_valid_action_mask[i, j] = float('-inf')

            non_final_next_state_q_values = agent.target_model(non_final_next_states)
            next_state_q_values.index_copy_(0, non_final_mask, non_final_next_state_q_values)
            # next_state_q_values[non_final_mask] += non_valid_action_mask
            next_state_q_values = next_state_q_values.max(1)[0]

            expected_current_state_q_values = (next_state_q_values * agent.gamma) + reward_batch
            loss = functional.smooth_l1_loss(current_state_q_values, expected_current_state_q_values)
            agent.logger.info(f'Loss was {loss}')
            agent.optimizer.zero_grad()
            loss.backward()
            agent.optimizer.step()

            agent.num_param_updates += 1

            if agent.num_param_updates % agent.target_update_freq == 0:
                print('Updating target model')
                agent.target_model.load_state_dict(agent.model.state_dict())

        if agent.step % agent.save_interval == 0 and agent.step > agent.learning_start_step and not agent.test:
            torch.save({
                'model': agent.model.state_dict(),
                'optimizer': agent.optimizer.state_dict(),
                'action_buffer': agent.action_buffer,
                'state_buffer': agent.state_buffer,
                'next_state_buffer': agent.next_state_buffer,
                'reward_buffer': agent.reward_buffer,
                'step': agent.step
            }, './models/' + str(agent.step) + '.pth')
            print('Saved model')

        if agent.step % 1000 == 0:
            print(agent.step)

        agent.step += 1
    except Exception as e:
        print(str(e))


def reward_update(agent):
    pass
    # agent.logger.debug(f'Found reward of {agent.reward}')


def end_of_episode(agent):
    reward = compute_reward(agent)

    agent.action_buffer[agent.step % agent.replay_buffer_size] = torch.LongTensor([[agent.last_action]])
    agent.state_buffer[agent.step % agent.replay_buffer_size] = agent.last_state
    agent.reward_buffer[agent.step % agent.replay_buffer_size] = torch.tensor([reward]).float()
    agent.next_state_buffer[agent.step % agent.replay_buffer_size] = torch.zeros((1, 6, 17, 17))

    agent.last_state = None
    agent.last_action = None


def compute_reward(agent):
    reward = -1
    for event in agent.game_state['events']:
        if event == e.BOMB_DROPPED:
            reward += 0
        elif event == e.COIN_COLLECTED:
            reward += 100
        elif event == e.CRATE_DESTROYED:
            reward += 30
        elif event == e.KILLED_SELF:
            reward -= 0
        elif event == e.KILLED_OPPONENT:
            reward += 100
        elif event == e.GOT_KILLED:
            reward -= 300
        elif event == e.WAITED:
            reward -= 2
        elif event == e.INVALID_ACTION:
            reward -= 2

    agent.logger.info(f'Found reward of {reward}')
    return reward


def select_action(state, agent):
    sample = np.random.random()

    if sample > agent.exploration.value(agent.step):
        output = agent.model.forward(state)
        # for i in range(len(output[0])):
        #     if not is_valid(state, i):
        #         output[0][i] = float('-inf')

        agent.logger.debug(f'Model output was {output}')

        # return Categorical(logits=output).sample().long()
        return output.max(1)[1].long()
    else:
        arr = np.random.permutation([0, 1, 2, 3, 4, 5])
        for i in range(6):
            if is_valid(state, i):
                return torch.from_numpy(np.array([arr[i]])).long()


def tile_is_free(arena, bombs, x, y):
    return (arena[x, y] == 0) and (bombs[x, y] == 0)


def is_valid(state, i):
    position = np.nonzero(state[0, 1])[0]

    if i == 0:
        return tile_is_free(state[0, 0], state[0, 3], position[0] + 1, position[1])
    elif i == 1:
        return tile_is_free(state[0, 0], state[0, 3], position[0] - 1, position[1])
    elif i == 2:
        return tile_is_free(state[0, 0], state[0, 3], position[0], position[1] - 1)
    elif i == 3:
        return tile_is_free(state[0, 0], state[0, 3], position[0], position[1] + 1)
    elif i == 4:
        # return state[0, 3][position[0], position[1]] == 0  # check if already a bomb is placed at the current spot
        return state[0, 3].sum() == 0
    else:
        return True  # wait is always allowed
