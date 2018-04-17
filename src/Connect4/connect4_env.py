import numpy as np
import os


class Connect4Env:

    def __init__(self, width=7, height=6):
        self.width = width
        self.height = height
        self.board = np.full((self.width, self.height), 0)

    def step(self, col_idx):

        step_player = self.get_current_player()

        # invalid column index is provided
        if col_idx >= self.width:
            state = self.board.copy()
            # reward = -100
            reward = -1
            result = -1
            return state, reward, result

        # the provided column is full
        row_idx = np.argmin(self.board, 1)[col_idx]
        if self.board[col_idx, row_idx] != 0:
            state = self.board.copy()
            # reward = -100
            reward = -1
            result = -1
            return state, reward, result

        # the provided column is valid
        # drop the chip to this column
        self.board[col_idx, row_idx] = step_player

        state = self.board.copy()
        result = self._judge(col_idx, row_idx)
        if result == step_player:
            reward = 1
        else:
            reward = 0

        # print(reward)

        return state, reward, result

    def simulate(self, test_state, col_idx):
        # backup the current state and current player
        snapshot = self.board.copy()
        # play on the given state
        self.board = test_state.copy()
        state, reward, result = self.step(col_idx)
        # restore the state and current player
        self.board = snapshot

        return state, reward, result

    def _judge(self, col_idx, row_idx):
        result = 0
        player = self.board[col_idx, row_idx]

        # vertical
        if row_idx >= 3:
            check_range = self.board[col_idx, np.arange(row_idx - 3, row_idx + 1)]
            if len(check_range[check_range != player]) == 0:
                result = player
                # print('Player', player, 'won vertically!')

        # horizontal
        for idx in range(col_idx - 3, col_idx + 4):
            if 0 <= idx < self.width:
                new_range = np.arange(max(idx, idx - 3), min(self.width, idx + 4))
                if len(new_range) >= 4:
                    check_range = self.board[new_range, row_idx]
                    if len(check_range[check_range != player]) == 0:
                        result = player
                        # print('Player', player, 'won horizontally!')
                        break
        # diagonal
        diag_bwd = np.diagonal(np.rot90(self.board), offset=((row_idx + 1) - (self.width - (col_idx + 1))))
        diag_fwd = np.diagonal(np.flipud(np.rot90(self.board)), offset=col_idx - row_idx)
        # print(diag_bwd)
        # print(diag_fwd)
        if len(diag_bwd) >= 4:
            for idx in range(0, len(diag_bwd) - 3):
                check_range = diag_bwd[idx:idx + 4]
                # print(idx, idx + 4)
                if len(check_range[check_range != player]) == 0:
                    result = player
                    # print('Player', player, 'won backward diagonally!')
                    break
        if len(diag_fwd) >=4:
            for idx in range(0, len(diag_fwd) - 3):
                check_range = diag_fwd[idx:idx + 4]
                # print(idx, idx + 4)
                if len(check_range[check_range != player]) == 0:
                    result = player
                    # print('Player', player, 'won forward diagonally!')
                    break
        # draw
        if len(self.board.reshape(self.width * self.height).nonzero()[0]) == self.width * self.height:
            # print('Draw game!')
            result = 3

        return result

    def get_all_next_actions(self):
        return [action for action in range(self.width)]

    def get_valid_actions(self, state=None):
        if state is None:
            state = self.board.copy()
        actions = []
        for col_idx in range(self.width):
            if np.min(state, 1)[col_idx] > 0:
                actions.append(0)
            else:
                actions.append(1)
        return actions

    def reset(self):
        self.board = np.full((self.width, self.height), 0)

    def to_str(self, board=None):
        string = os.linesep
        if board is None:
            board = self.board
        b = np.rot90(board).reshape(self.width * self.height)
        for idx, c in enumerate(b):
            c = int(c)
            if (idx + 1) % self.width > 0:
                string = '{}{} '.format(string, c)
            else:
                string = '{}{}{}'.format(string, c, os.linesep)
        return string

    def print(self, board=None):
        print(self.to_str(board))

    def get_state(self):
        return self.board.copy().astype(dtype=np.float32)

    def get_mirror_state(self, board=None):
        if board is None:
            board = self.board
        mirror = np.array(board)
        for col_idx in range(self.width):
            for row_idx in range(self.height):
                mirror[self.width - col_idx - 1, row_idx] = board[col_idx, row_idx]
        # return mirror.astype(dtype=np.float32)
        return mirror

    def get_inv_state(self, board=None):
        if board is None:
            board = self.board
        # inv = board.copy().astype(dtype=np.float32)
        inv = board.copy()
        for col_idx in range(self.width):
            for row_idx in range(self.height):
                if inv[col_idx, row_idx] == 1:
                    inv[col_idx, row_idx] = 2
                elif inv[col_idx, row_idx] == 2:
                    inv[col_idx, row_idx] = 1
        return inv

    def get_current_player(self, state=None):
        if state is None:
            state = self.board.copy()
        unique, counts = np.unique(state, return_counts=True)
        player1_count = counts[np.where(unique == 1)]
        player2_count = counts[np.where(unique == 2)]
        if len(player1_count) == 0:
            player1_count = 0
        if len(player2_count) == 0:
            player2_count = 0
        if player1_count > player2_count:
            return 2
        else:
            return 1


def main():
    b = Connect4Env()

    while True:
        b.print()
        col_idx = int(input('Player {}\'s turn. Please input the col number (1 to {}) you want to place your chip:'.format(b.get_current_player(), b.width)))
        state, reward, result = b.step(col_idx - 1)
        if result < 0:
            print('Your input is invalid.')
        elif result == 0:
            pass
        elif result == 3:
            print('Draw game!!!!!')
        else:
            print('Player', b.get_current_player(), 'won!!!!!')
            b.print()
            break


if __name__ == '__main__':
    main()