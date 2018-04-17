import numpy as np
import math
import random
import utils

from config import Config
from network import Network
from connect4_env import Connect4Env
from logger import logger

config = Config()

class MCT(object):

    def __init__(self, network):
        self.network = network
        self.tree = []
        self.A_s = {}
        self.N_s = {}
        self.P_s = {}
        self.Q_sa = {}
        self.N_sa = {}
        self.Cpuct = config.Cpuct

    def search(self, state, reward, result, env, is_search_root=True):
        if is_search_root:
            logger.debug('-= A =-')
            logger.debug('-= NEW =-')
            logger.debug('-= SEARCH =-')
        logger.debug('SEARCHING STATE AS PLAYER {}:'.format(env.get_current_player(state=state)))
        logger.debug(env.to_str(state))
        # if the game has reached the end state, return the reward as V, +1 or -1
        if result > 0:
            logger.debug('..........Reached end state return V = {}..........'.format(reward))
            return reward

        state_id = self._state2id(state)
        # the state is not on the tree, init the node by network and add it onto the tree
        if state_id not in self.tree:
            logger.debug('++++++++++Reach a new state++++++++++')
            v, p = self.network.predict(utils.format_state(state=state, env=env))
            valid_action_mask = env.get_valid_actions(state=state)
            self.A_s[state_id] = valid_action_mask
            self.P_s[state_id] = p[0] * valid_action_mask
            self.N_s[state_id] = 0
            self.tree.append(state_id)
            logger.debug('  valid action mask: {}'.format(valid_action_mask))
            logger.debug('  masked probabilities: {}'.format(self.P_s[state_id]))
            logger.debug('  return V = -{}'.format(v))
            logger.debug('added this state onto the tree')
            return -v
        # if the state is already on the tree, expand the node
        else:
            logger.debug('!!!!!!!!!!Expanding an existing state!!!!!!!!!!')
            # first, select the best action that maximize U value
            max_u = -float('inf')
            best_action = -1
            # TODO may need to shuffle the actions, so that it will not always expand the first when all U are zero
            actions = env.get_all_next_actions()
            if is_search_root:
                epsilon = config.Dir_Epsilon
                nu = np.random.dirichlet([config.Dir_Alpha] * len(actions))
            else:
                epsilon = 0
                nu = [0] * len(actions)
            for action in actions:
                if self.A_s[state_id][action] == 1:
                    state_action_id = self._state_action2id(state, action)
                    if state_action_id in self.Q_sa:
                        logger.debug('  action {} of the current state has been visited before'.format(action))
                        u = self.Q_sa[state_action_id] + self.Cpuct * (
                                (1 - epsilon) * self.P_s[state_id][action] + epsilon * nu[action]) * math.sqrt(
                            self.N_s[state_id]) / (1 + self.N_sa[state_action_id])
                        logger.debug(
                            '  U = Q(s,a) + Cpuct * ((1 - epsilon) * P(s,a) + epsilon * dir(alpha)) * sqrt(N(s)) / (1 + N(s,a)) = {} + {} * ((1 - {}) * {} + {} * {}) * sqrt({}) / (1 + {}) = {}'.format(
                                self.Q_sa[state_action_id], self.Cpuct, epsilon, self.P_s[state_id][action], epsilon,
                                nu[action], self.N_s[state_id],
                                self.N_sa[state_action_id], u))
                    elif self.N_s[state_id] > 0:
                        logger.debug('  action {} of the current state has never been visited before'.format(action))
                        u = self.Cpuct * (
                                (1 - epsilon) * self.P_s[state_id][action] + epsilon * nu[action]) * math.sqrt(
                            self.N_s[state_id])
                        logger.debug(
                            '  U = Cpuct * ((1 - epsilon) * P(s,a) + epsilon * dir(alpha)) * sqrt(N(s)) = {} * ((1 - {}) * {} + {} * {}) * sqrt({}) = {}'.format(
                                self.Cpuct, epsilon, self.P_s[state_id][action], epsilon, nu[action],
                                self.N_s[state_id], u
                            ))
                    else:
                        logger.debug('  action {} of the current state has never been visited before'.format(action))
                        u = self.Cpuct * (
                                (1 - epsilon) * self.P_s[state_id][action] + epsilon * nu[action])
                        logger.debug(
                            '  U = Cpuct * ((1 - epsilon) * P(s,a) + epsilon * dir(alpha)) = {} * ((1 - {}) * {} + {} * {}) = {}'.format(
                                self.Cpuct, epsilon, self.P_s[state_id][action], epsilon, nu[action], u
                            ))
                    if u > max_u:
                        max_u = u
                        best_action = action
                else:
                    logger.debug('  action {} is invalid'.format(action))
            logger.debug('  the best action is {}'.format(best_action))
            # then, take the best action and continue traversing the tree by invoking the search method recursively,
            # which also back fills the Q(s,a) and N(s,a)
            next_state, reward, result = env.simulate(test_state=state, col_idx=best_action)
            logger.debug('//////////Traverse the state of the best action recursively//////////')
            v = self.search(state=next_state, reward=reward, result=result, env=env, is_search_root=False)
            logger.debug('//////////Traverse is done//////////')
            # part of back fill process, update Q(s,a) and N(s,a)
            state_action_id = self._state_action2id(state, best_action)
            # if Q(s,a) already exists
            if state_action_id in self.Q_sa:
                old_N_sa = self.N_sa[state_action_id]
                old_Q_sa = self.Q_sa[state_action_id]
                # increase N(s,a)
                self.N_sa[state_action_id] += 1
                logger.debug('  increased N(s,a) from {} to {}'.format(old_N_sa, self.N_sa[state_action_id]))
                # re-calculate the mean of V as Q(s,a)
                self.Q_sa[state_action_id] = (self.Q_sa[state_action_id] * old_N_sa + v) / self.N_sa[state_action_id]
                logger.debug('  increased Q(s,a) from {} to {}'.format(old_Q_sa, self.Q_sa[state_action_id]))
            # if Q(s,a) does not exist, meaning the 'next state' generated by (s,a) is not on the tree
            else:
                # init Q(s,a) as the prediction of the network
                self.Q_sa[state_action_id] = v
                logger.debug('  init Q(s,a) as {}'.format(v))
                self.N_sa[state_action_id] = 1
                logger.debug('  init N(s,a) as 1')
            # increase N(s) because we just expanded state s
            old_N_s = self.N_s[state_id]
            self.N_s[state_id] += 1
            logger.debug('  increased N(s) from {} to {}'.format(old_N_s, self.N_s[state_id]))
            return -v

    def get_actions_probability(self, state, env, temperature=config.Temperature):
        logger.debug('Getting action probabilities for the following state with temperature {}:'.format(temperature))
        logger.debug(env.to_str(state))
        N_sa_list = []
        for action in env.get_all_next_actions():
            state_action_id = self._state_action2id(state, action)
            N_sa_list.append(self.N_sa[state_action_id] if state_action_id in self.N_sa else 0)
        logger.debug('N(s,a) list for the give state is {}'.format(N_sa_list))
        if temperature == 0:
            best_a = random.choice(np.argwhere(N_sa_list == np.amax(N_sa_list)))[0]
            prob = [0 for _ in range(len(env.get_all_next_actions()))]
            prob[best_a] = 1
        else:
            N_sa_list = [x ** (1 / temperature) for x in N_sa_list]
            sum_N_sa = sum(N_sa_list)
            prob = [float(N_sa / sum_N_sa) for N_sa in N_sa_list]
        logger.debug('the probabilities are {}'.format(prob))
        return prob

    def _train(self, steps, state, reward, result, env):
        for i in range(steps):
            self.search(state, reward, result, env)

    def _predict_action(self, state, env):
        logger.debug('Predict action based on action probabilities for state: {}'.format(env.to_str(state)))
        prob = self.get_actions_probability(state, env)
        logger.debug('The probabilities are {} and the best one is {}'.format(prob, np.argmax(prob)))
        return np.argmax(prob)

    def _state2id(self, state):
        return ''.join(str(int(s)) for s in state.flatten())

    def _state_action2id(self, state, action):
        return ''.join((self._state2id(state), str(action)))


if __name__ == '__main__':
    network = Network(input_dim=(7, 6, 1), output_dim=7,
                      layers_metadata=[{'filters': 42, 'kernel_size': (4, 4)}, {'filters': 42, 'kernel_size': (4, 4)},
                                       {'filters': 42, 'kernel_size': (4, 4)}], reg_const=0.6, learning_rate=0.0005,
                      root_path=None)
    env = Connect4Env(width=7, height=6)
    mct = MCT(network=network)

    player = 1
    try:
        human_player = int(input('Would you like to be the 1st player or the 2nd player (answer 1 or 2): '))
        if human_player not in (1, 2):
            print('Sorry, I don''t understand your answer. I will play with myself.')
    except:
        print('Sorry, I don''t understand your answer. I will play with myself.')
        human_player = 3

    reward = 0
    result = 0
    while True:
        state = env.get_state()
        mct._train(env.width * 2, state, -reward, -result, env)

        env.print()
        if player == human_player:
            col_idx = int(input(
                'Player {}\'s turn. Please input the col number (1 to {}) you want to place your chip:'.format(
                    player,
                    env.width)))
        else:
            col_idx = mct._predict_action(state, env) + 1
            print('Player {} is placing the chip to'.format(player), col_idx)
        _, reward, result = env.step(col_idx - 1)
        if result < 0:
            print('Your input is invalid.')
            if player != human_player:
                break
        elif result == 0:
            if player == 1:
                player = 2
            else:
                player = 1
        elif result == 3:
            print('Draw game!!!!!')
            break
        else:
            print('Player', player, 'won!!!!!')
            break

    env.print()
