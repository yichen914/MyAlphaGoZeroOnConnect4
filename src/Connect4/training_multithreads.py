import numpy as np
import random
import utils
import time
import pickle
import os

from config import Config
from connect4_env import Connect4Env
from monte_carlo_tree import MCT
from network import Network
from logger import logger
from collections import deque
from threading import Thread

config = Config(is_multithread=True)


class Training(object):

    def __init__(self, best_network):
        # init memory
        self.memory = deque(maxlen=config.Memory_Size)
        self._load_memory()
        # init best network, this network can be loaded from disk
        self.best_network = best_network
        # init current network for training and make it as same as the best network
        self.current_network = Network('Current')
        # self.current_network.replace_by(self.best_network)
        # init test network for comparision
        self.test_network = Network('Test')
        self.self_play_switch = True
        self.fit_switch = True
        self.comparison_switch = False
    
    def run_episode(self):
        steps = []
        env = Connect4Env(width=config.Width, height=config.Height)
        mct = MCT(network=self.best_network)
        state = env.get_state()
        reward = 0
        result = 0
        while True:
            # MCTS
            for i in range(config.MCTS_Num):
                mct.search(state=state, reward=reward, result=result, env=env)
            # get PI from MCT
            if len(steps) < 10:
                pi = mct.get_actions_probability(state=state, env=env, temperature=1)
            else:
                pi = mct.get_actions_probability(state=state, env=env, temperature=0)
            # add (state, PI and placeholder of W) to memory
            steps.append([state, pi, None, env.get_current_player()])
            # choose an action based on PI
            action = np.random.choice(len(pi), p=pi)
            # take the action
            state, reward, result = env.step(action)
            logger.debug(action + 1)
            logger.debug(env.to_str(state))
            # if game is finished, back fill the W placeholder
            if result != 0:
                steps = self._assign_w(steps=steps, winner=result)
                steps = self._symmetrize_steps(steps=steps)
                # logger.info(self.self_play_env.to_str(state))
                break
        for step in steps:
            self.memory.append(step)
            logger.debug('================================')
            logger.debug(env.to_str(step[0]))
            logger.debug('player: {}'.format(step[3]))
            logger.debug('probabilities: {}'.format(step[1]))
            logger.debug('value: {}'.format(step[2]))

    def _assign_w(self, steps, winner):
        for i in range(len(steps)):
            if i % 2 == 0:
                if winner == 1:
                    steps[i][2] = 1
                else:
                    steps[i][2] = -1
            else:
                if winner == 2:
                    steps[i][2] = 1
                else:
                    steps[i][2] = -1
        logger.debug('back fill result: {}'.format(steps))
        return steps

    def _symmetrize_steps(self, steps):
        env = Connect4Env(width=config.Width, height=config.Height)
        for i in range(len(steps)):
            state = steps[i][0]
            prob = steps[i][1]
            symmetrical_state = env.get_mirror_state(state)
            symmetrical_prob = prob[::-1]
            steps.append([symmetrical_state, symmetrical_prob, steps[i][2], steps[i][3]])
        return steps

    def _prepare_training_data(self, samples):
        inputs = []
        targets_w = []
        targets_pi = []
        env = Connect4Env(width=config.Width, height=config.Height)
        for sample in samples:
            inputs.append(utils.format_state(sample[0], env))
            targets_pi.append(sample[1])
            targets_w.append(sample[2])
        return np.vstack(inputs), [np.vstack(targets_w), np.vstack(targets_pi)]

    # def compete_for_best_network(self, new_network, best_network):
    #     logger.info('Comparing network....')
    #     mct_new = MCT(network=new_network)
    #     mct_best = MCT(network=best_network)
    #     players = [[mct_new, 0], [mct_best, 0]]
    #     env = Connect4Env(width=config.Width, height=config.Height)
    #
    #     for i in range(config.Compete_Game_Num):
    #         env.reset()
    #         state = env.get_state()
    #         reward = 0
    #         result = 0
    #         step = 0
    #         draw_games = 0
    #         logger.debug('{} network get the upper hand for this game.'.format(players[step % 2][0].network.name))
    #         while True:
    #             for _ in range(config.Test_MCTS_Num):
    #                 players[step % 2][0].search(state=state, reward=reward, result=result, env=env)
    #             prob = players[step % 2][0].get_actions_probability(state=state, env=env, temperature=0)
    #             action = np.random.choice(len(prob), p=prob)
    #             state, reward, result = env.step(col_idx=action)
    #             if result == 1:
    #                 players[0][1] += 1
    #                 break
    #             elif result == 2:
    #                 players[1][1] += 1
    #                 break
    #             elif result == 3:
    #                 draw_games += 1
    #                 break
    #             else:
    #                 step += 1
    #         logger.debug(env.to_str())
    #         logger.debug(result)
    #         logger.info(''.join(('#' * (i + 1), '-' * (config.Compete_Game_Num - i - 1))))
    #         players.reverse()
    #
    #     if players[0][0] == mct_new:
    #         mct_new_wins = players[0][1]
    #         mct_best_wins = players[1][1]
    #     else:
    #         mct_new_wins = players[1][1]
    #         mct_best_wins = players[0][1]
    #
    #     compete_result = mct_new_wins / (mct_best_wins + mct_new_wins)
    #     logger.debug(
    #         'new network won {} games, best network won {} games, draw games are {}'.format(mct_new_wins,
    #                                                                                              mct_best_wins,
    #                                                                                              draw_games))
    #     logger.info('new network winning ratio is {}'.format(compete_result))
    #
    #     is_update = compete_result > config.Best_Network_Threshold
    #     if is_update:
    #         self.best_network.replace_by(new_network)
    #         logger.info('Updated best network!!!!')
    #     else:
    #         # self.current_network.replace_by(self.best_network)
    #         logger.info('Discarded current network....')
    #     return is_update

    def compete_for_best_network(self, new_network, best_network):
        logger.info('Comparing network....')
        mct_new = MCT(network=new_network)
        mct_best = MCT(network=best_network)
        players = [[mct_new, 0], [mct_best, 0]]
        env = Connect4Env(width=config.Width, height=config.Height)

        mct_new_wins = 0
        mct_best_wins = 0
        draw_games = 0
        for i in range(config.Compete_Game_Num):
            env.reset()
            state = env.get_state()
            reward = 0
            result = 0
            step = 0

            logger.debug('{} network get the upper hand for this game.'.format(players[step % 2][0].network.name))
            while True:
                for _ in range(config.Test_MCTS_Num):
                    players[step % 2][0].search(state=state, reward=reward, result=result, env=env)
                prob = players[step % 2][0].get_actions_probability(state=state, env=env, temperature=0)
                action = np.random.choice(len(prob), p=prob)
                state, reward, result = env.step(col_idx=action)
                if result == 1:
                    players[0][1] += 1
                    break
                elif result == 2:
                    players[1][1] += 1
                    break
                elif result == 3:
                    draw_games += 1
                    break
                else:
                    step += 1
            logger.debug(env.to_str())
            logger.debug(result)

            if players[0][0] == mct_new:
                mct_new_wins = players[0][1]
                mct_best_wins = players[1][1]
            else:
                mct_new_wins = players[1][1]
                mct_best_wins = players[0][1]

            logger.info(''.join(
                ('O' * mct_new_wins, 'X' * mct_best_wins, '-' * draw_games, '.' * (config.Compete_Game_Num - i - 1))))

            if mct_best_wins / (mct_new_wins + mct_best_wins + (config.Compete_Game_Num - i - 1)) >= (1 - config.Best_Network_Threshold):
                logger.info('new network has no hope to win in the comparison, so stop the comparison early.')
                break
            elif mct_new_wins / (mct_new_wins + mct_best_wins + (config.Compete_Game_Num - i - 1)) > config.Best_Network_Threshold:
                logger.info('new network has already won in the comparison, so stop the comparison early.')
                break
            else:
                players.reverse()

        compete_result = mct_new_wins / (mct_best_wins + mct_new_wins)
        logger.debug(
            'new network won {} games, best network won {} games, draw games are {}'.format(mct_new_wins,
                                                                                                 mct_best_wins,
                                                                                                 draw_games))
        logger.info('new network winning ratio is {}'.format(compete_result))

        is_update = compete_result > config.Best_Network_Threshold
        if is_update:
            self.best_network.replace_by(new_network)
            logger.info('Updated best network!!!!')
        else:
            # self.current_network.replace_by(self.best_network)
            logger.info('Discarded current network....')
        return is_update

    def self_play(self):
        while self.self_play_switch:
            self.run_episode()
            logger.info('generated new samples. total sample number is {}'.format(len(self.memory)))
            self._save_memory()

    def fit(self):
        while self.fit_switch:
            if len(self.memory) >= config.Min_Memory_Size_Before_Fit:
                inputs, targets = self._prepare_training_data(random.sample(self.memory, config.Sample_Size))
                val_inputs, val_targets = self._prepare_training_data(random.sample(self.memory, config.Sample_Size))
                logger.info('Fitting network....')
                self.current_network.fit(inputs=inputs, targets=targets, epochs=config.Epochs_Num,
                                         batch_size=config.Batch_Size, validation_data=(val_inputs, val_targets))
                logger.info('Fitted network')
                self.comparison_switch = True
            time.sleep(config.Fit_Interval)

    def comparison(self):
        while not self.comparison_switch:
            time.sleep(config.Comparison_Interval)
        update_count = 0
        long_wait = config.Comparison_Long_Wait
        while self.comparison_switch:
            if update_count >= config.Iteration_Num:
                break
            elif len(self.memory) >= config.Min_Memory_Size_Before_Fit:
                if long_wait > 0:
                    logger.info(
                        'Comparison thread will fall into a long sleep to give fit thread some time to fit the network....')
                    time.sleep(long_wait)
                    long_wait = 0
                # take a snapshot of the current network as test network for comparison
                self.test_network.replace_by(self.current_network)
                logger.info(
                    'took a snapshot of the current network as test network for No. {} update.'.format(update_count + 1))
                is_update = self.compete_for_best_network(self.test_network, self.best_network)
                if is_update:
                    update_count += 1
                    self.best_network.save()
                    # clean up certain percentage of memory due to the creation of new best network
                    for _ in range(int(len(self.memory) * config.New_Best_Network_Memory_Clean_Rate)):
                        self.memory.pop()
                    # set long wait time to 1 hour
                    long_wait = config.Comparison_Long_Wait
                # save the current work after every comparison
                self.current_network.save()
            else:
                time.sleep(config.Comparison_Interval)
        self.self_play_switch = False
        self.fit_switch = False

    def _save_memory(self):
        with open('{}\models\connect4agent_memory.dump'.format(config.Root_Path), 'wb+') as file_handler:
            pickle.dump(self.memory, file_handler)

    def _load_memory(self):
        filepath = '{}\models\connect4agent_memory.dump'.format(config.Root_Path)
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as file_handler:
                self.memory = pickle.load(file_handler)
            logger.info('load {} data into memory'.format(len(self.memory)))
                
    def train(self):
        self_play_thread = Thread(target=self.self_play)
        fit_thread = Thread(target=self.fit)
        comparison_thread = Thread(target=self.comparison)

        self_play_thread.setDaemon(True)
        fit_thread.setDaemon(True)
        comparison_thread.setDaemon(True)

        self_play_thread.start()
        fit_thread.start()
        comparison_thread.start()

        self_play_thread.join()
        fit_thread.join()
        comparison_thread.join()


if __name__ == '__main__':

    training_flag = str(input('Would you like to train the network before test it (answer Y or N): ')).upper() == 'Y'

    best_network = Network('Best')

    if training_flag:
        training = Training(best_network)
        time.sleep(10)
        training.train()
    # ==========================================
    player = 1
    env = Connect4Env(width=config.Width, height=config.Height)
    mct = MCT(network=best_network)
    reward = 0
    result = 0
    try:
        human_player = int(input('Would you like to be the 1st player or the 2nd player (answer 1 or 2): '))
        if human_player not in (1, 2):
            print('Sorry, I don''t understand your answer. I will play with myself.')
    except:
        print('Sorry, I don''t understand your answer. I will play with myself.')
        human_player = 3

    while True:
        env.print()
        if player == human_player:
            col_idx = int(input(
                'Player {}\'s turn. Please input the col number (1 to {}) you want to place your chip:'.format(
                    player,
                    env.width)))
        else:
            state = env.get_state()
            for _ in range(config.Test_MCTS_Num):
                mct.search(state=state, reward=reward, result=result, env=env)
            prob = mct.get_actions_probability(state=state, env=env, temperature=0)
            action = np.random.choice(len(prob), p=prob)
            col_idx = action + 1
            print('Player {} is placing the chip to'.format(player), col_idx)
        state, reward, result = env.step(col_idx - 1)
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

