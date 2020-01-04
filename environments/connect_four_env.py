import numpy as np
import tensorflow as tf

from .environment import Environment

class ActionNotAvailableException(Exception):
    '''Error: column already full. Action not available. '''
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return str(self.value)


class ConnectFourEnvEncoder():
    def __init__(self):
        self.n_rows = 6
        self.n_cols = 7
    
    @staticmethod
    def encode(state):
        # TODO - more efficient representation
        return state

    @staticmethod
    def decode_action_index(index):
        # TODO - placeholder for more efficient string methods
        return index

    def n_actions(self):
        return self.n_cols
    

class ConnectFourEnv(Environment):
    '''
    Class representing ConnectFour environment.
    '''
    def __init__(self):
        super(ConnectFourEnv, self).__init__()
        
        self.n_rows = 6
        self.n_cols = 7
        
    def reset(self):
        # [player1_pieces, player2_pieces, whose_turn] [rows] [columns]
        # 0th dimension - 'one-hot' style encoding. Player_1 = '0', Player_2 = '1'.
        self.board_shape = (3, self.n_rows, self.n_cols)

        # state
        return np.zeros(self.board_shape)
        
    def step(self, state, action, current_player):
        '''
        Takes an action in state_t, yielding state_(t+1) and toggles which player's turn
        it is to move next.

        Args:
          state (np.array): current state.
          action (int): next column to place a move.
        Returns:
          next_state.
        '''
        if not self.is_action_available(state, action):
            raise ActionNotAvailableException('Error: column already full. Action not available.')

        # Adjust for zero indexing
        row = self.n_rows - 1

        # Find the lowest row in chosen column to place player's piece
        while row >= 0:
            if (state[current_player, row, action] == 0) and (state[current_player ^ 1, row, action] == 0):
                break
            row -= 1

        # Take the action
        next_state = state.copy()
        next_state[current_player, row, action] = 1

        # Update the model's view of whose turn it is in the next_state
        next_state[-1] = np.full((self.n_rows, self.n_cols), current_player)

        # All non-winning moves get 0
        reward = 0

        # Check if no more space on board
        is_game_over = (np.sum(state[0:-1]) == self.n_rows * self.n_rows)

        # Check if action in most recent column and row caused for current player
        # For now, loss is handled as -1 by toggling in the MCTS.
        is_winner = self.check_winner(state[current_player], row, action)
        if is_winner is True:
            reward = 1
            is_game_over = True

        return next_state, reward, is_game_over

    
    def is_action_available(self, state, action):
        '''
        Checks whether an action / column is available.
        Does so by checking if the cell in the top row of each column is available.
        Player 1 plays on the 0th plane of the 3rd board state dimension.
        Player 2 plays on the 1st plane of the 3rd board state dimension.
        Therefore check neither player has a piece in the top cell for the column.
        '''
        # For clarity:
        player_one_index = 0
        player_two_index = 1
        # [3,6,7]
        return (state[player_one_index, 0, action] == 0) and (state[player_two_index, 0, action] == 0)

    def check_winner(self, state, row, column):
        if self._horizontal_win(state, row, column):
            return True
        if self._vertical_win(state, row, column):
            return True
        if self._diagonal_win(state):
            return True
        return False
    
    def _horizontal_win(self, state, r, c):
        ''' Check up to 3 pieces either side of placed piece. '''

        left = max(c-3, 0)
        right = min(c+3, self.n_cols-1)  # n_c-1: for zero-indexing

        # Loop up to right-2: loop range does not execute 3rd and final anyway
        for i in range(left, right-2):
            if state[r][i] == 1 and state[r][i+1] == 1 and state[r][i+2] == 1 and state[r][i+3] == 1:
                return True

        return False

    def _vertical_win(self, state, r, c):
        ''' Check up to 3 pieces either side of placed piece. '''

        low = max(r-3, 0)
        high = min(r+3, self.n_rows - 1)  # # n_r-1: for zero-indexing

        # Loop up to high-2: loop range does not execute 3rd and final anyway
        for i in range(low, high-2):
            if state[i][c] == 1 and state[i+1][c] == 1 and state[i+2][c] == 1 and state[i+3][c] == 1:
                return True

        return False


    def _diagonal_win(self, state):
        h = len(state)
        w = len(state[0])

        for p in range(h + w - 1):
            # Diagonals
            diags = 0
            for q in range(max(p-h+1, 0), min(p+1, w)):
                d = state[h - p + q - 1][q]
                diags += d
                if diags >= 4:
                    return True

            # Anti-diagonals
            antis = 0
            for q in range(max(p-h+1,0), min(p+1, w)):
                a = state[p - q][q]
                antis += a
                if antis >= 4:
                    return True
        
        return False
